from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_trace_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix == ".jsonl":
            current = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = read_json(path)
            if isinstance(payload, list):
                current = payload
            elif isinstance(payload, dict) and isinstance(payload.get("step_records"), list):
                current = payload["step_records"]
            else:
                raise ValueError(f"Unsupported trace JSON shape: {path}")
        for index, row in enumerate(current):
            if not isinstance(row, dict):
                raise TypeError(f"Trace row {index} in {path} is not an object")
            item = dict(row)
            item["__trace_source"] = str(path)
            rows.append(item)
    if not rows:
        raise ValueError("No trace rows were loaded")
    return rows


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _component_flops(cost_table: Mapping[str, Any]) -> dict[str, float]:
    components = cost_table.get("components")
    if not isinstance(components, dict) or not components:
        raise ValueError("cost table must contain a non-empty components object")
    result = {}
    for name, payload in components.items():
        if not isinstance(payload, dict):
            raise TypeError(f"component {name} must be an object")
        result[str(name)] = _finite_nonnegative(payload.get("flops"), f"{name}.flops")
    return result


def _component_names(
    action_components: Mapping[str, Any], action: str, stage: str
) -> list[str]:
    by_stage = action_components.get(action)
    if not isinstance(by_stage, dict):
        raise KeyError(f"No component mapping for action={action!r}")
    names = by_stage.get(stage, by_stage.get("$default"))
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise KeyError(f"No component list for action={action!r}, stage={stage!r}")
    return names


def aggregate_trace(
    *,
    cost_table: Mapping[str, Any],
    mapping: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    component_costs = _component_flops(cost_table)
    action_components = mapping.get("action_components")
    if not isinstance(action_components, dict):
        raise ValueError("mapping.action_components must be an object")

    sample_field = str(mapping.get("sample_field", "sample_id"))
    step_field = str(mapping.get("step_field", "step_index"))
    stage_field = str(mapping.get("stage_field", "model_stage"))
    action_field = str(mapping.get("action_field", "decision"))
    baseline_action = str(mapping.get("baseline_action", "baseline"))
    aliases = mapping.get("action_aliases", {})
    if not isinstance(aliases, dict):
        raise ValueError("mapping.action_aliases must be an object")

    per_sample: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "steps": 0,
            "candidate_flops": 0.0,
            "baseline_flops": 0.0,
            "actions": defaultdict(int),
            "stages": defaultdict(int),
        }
    )

    for row_index, row in enumerate(rows):
        source = str(row.get("__trace_source", "trace"))
        sample_id = str(row.get(sample_field) or Path(source).stem)
        if stage_field not in row or action_field not in row:
            raise KeyError(
                f"Trace row {row_index} must contain {stage_field!r} and {action_field!r}"
            )
        stage = str(row[stage_field])
        raw_action = str(row[action_field])
        action = str(aliases.get(raw_action, raw_action))
        candidate_names = _component_names(action_components, action, stage)
        baseline_names = _component_names(action_components, baseline_action, stage)

        def total(names: list[str]) -> float:
            missing = [name for name in names if name not in component_costs]
            if missing:
                raise KeyError(f"Unknown components in mapping: {missing}")
            return sum(component_costs[name] for name in names)

        item = per_sample[sample_id]
        item["steps"] += 1
        item["candidate_flops"] += total(candidate_names)
        item["baseline_flops"] += total(baseline_names)
        item["actions"][action] += 1
        item["stages"][stage] += 1
        if step_field in row:
            item.setdefault("step_indices", []).append(row[step_field])

    samples = []
    for sample_id in sorted(per_sample):
        item = per_sample[sample_id]
        candidate = item["candidate_flops"]
        baseline = item["baseline_flops"]
        samples.append(
            {
                "sample_id": sample_id,
                "steps": item["steps"],
                "candidate_flops": candidate,
                "candidate_tflops": candidate / 1_000_000_000_000,
                "baseline_flops": baseline,
                "baseline_tflops": baseline / 1_000_000_000_000,
                "flops_speedup": baseline / candidate if candidate > 0 else None,
                "actions": dict(sorted(item["actions"].items())),
                "stages": dict(sorted(item["stages"].items())),
            }
        )

    candidate_total = sum(item["candidate_flops"] for item in samples)
    baseline_total = sum(item["baseline_flops"] for item in samples)
    return {
        "schema_version": 1,
        "scope": "path-aware forward FLOPs from component table and actual trace",
        "sample_count": len(samples),
        "step_count": sum(item["steps"] for item in samples),
        "candidate_total_flops": candidate_total,
        "candidate_total_tflops": candidate_total / 1_000_000_000_000,
        "candidate_mean_tflops_per_sample": (
            candidate_total / len(samples) / 1_000_000_000_000
        ),
        "baseline_total_flops": baseline_total,
        "baseline_total_tflops": baseline_total / 1_000_000_000_000,
        "baseline_mean_tflops_per_sample": (
            baseline_total / len(samples) / 1_000_000_000_000
        ),
        "flops_speedup_ratio_of_sums": (
            baseline_total / candidate_total if candidate_total > 0 else None
        ),
        "per_sample": samples,
    }
