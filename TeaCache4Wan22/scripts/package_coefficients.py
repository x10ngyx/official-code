#!/usr/bin/env python3
"""Package validated raw calibration output into the public runtime schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


PROTOCOL_KEYS = (
    "task",
    "size_wh",
    "frame_num",
    "sampling_steps",
    "sample_solver",
    "shift",
    "guide_scale_low_high",
    "boundary",
    "param_dtype",
    "use_ret_steps",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_git_repository(value: Any) -> str:
    """Normalize the optional transport suffix without weakening repository identity."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Git repository provenance must be a non-empty string.")
    normalized = value.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    project_root = Path(__file__).resolve().parents[1]

    coefficients_path = result_root / "coefficients.json"
    fit_report_path = result_root / "FIT_REPORT.md"
    manifest_path = result_root / "prompts.jsonl"
    selection_path = result_root / "prompt_selection.json"
    for path in (coefficients_path, fit_report_path, manifest_path, selection_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw = load_json(coefficients_path)
    selection = load_json(selection_path)
    manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(manifest) != 70 or raw.get("prompt_count") != 70:
        raise ValueError("Coefficient packaging requires exactly 70 prompts.")
    if len({row.get("sample_id") for row in manifest}) != 70:
        raise ValueError("Calibration manifest sample IDs must be unique.")
    if raw.get("raw_forward_record_count") != 7000:
        raise ValueError("Expected 7000 raw forward records.")
    if raw.get("within_stage_transition_count") != 6720:
        raise ValueError("Expected 6720 within-stage transition records.")
    if raw.get("primary_fit_record_count") != 6580:
        raise ValueError("Expected 6580 primary fit records.")
    fit_software = raw.get("fit_software", {})
    if not fit_software.get("python") or not fit_software.get("numpy"):
        raise ValueError("Raw fit is missing Python/NumPy provenance.")
    if set(fit_software.get("thread_environment", {}).values()) != {"1"}:
        raise ValueError("Raw fit was not constrained to one BLAS thread.")
    fit_script_path = (
        project_root
        / "experiments"
        / "fit_t2vcompbench70_wan22_t2v_a14b"
        / "fit_polynomials.py"
    )
    if fit_software.get("fit_script_sha256") != sha256(fit_script_path):
        raise ValueError("Raw fit was produced by a different fit_polynomials.py.")

    sample_paths = sorted((result_root / "samples").glob("*.json"))
    if len(sample_paths) != 70:
        raise ValueError(f"Expected 70 sample files, found {len(sample_paths)}")
    sample_hashes = {}
    observed_ids = set()
    for path in sample_paths:
        payload = load_json(path)
        if payload.get("status") != "complete" or len(payload.get("records", [])) != 100:
            raise ValueError(f"Invalid calibration sample: {path}")
        values = (
            value
            for row in payload["records"]
            for value in (
                row.get("e_rel_l1"),
                row.get("h_rel_l1"),
                row.get("block_residual_rel_l1"),
            )
            if value is not None
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"Non-finite calibration statistic in {path}")
        observed_ids.add(payload["prompt"]["sample_id"])
        sample_hashes[path.name] = sha256(path)
    expected_ids = {row["sample_id"] for row in manifest}
    if observed_ids != expected_ids:
        raise ValueError("Calibration sample IDs do not match prompts.jsonl.")

    lock = load_json(project_root / "upstream_lock.json")
    if canonical_git_repository(selection.get("source_repository")) != canonical_git_repository(
        lock["t2v_compbench"]["repository"]
    ):
        raise ValueError("Prompt selection repository differs from upstream_lock.json.")
    if selection.get("source_commit") != lock["t2v_compbench"]["commit"]:
        raise ValueError("Prompt selection commit differs from upstream_lock.json.")
    if selection.get("selection_seed") != 42:
        raise ValueError("Public coefficient packaging requires selection_seed=42.")
    if selection.get("selected_prompt_count") != 70:
        raise ValueError("Prompt selection metadata does not record 70 prompts.")
    if selection.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("Prompt selection manifest SHA256 mismatch.")
    shard_paths = sorted((result_root / "shards").glob("shard_*.json"))
    if [path.name for path in shard_paths] != [f"shard_{index}.json" for index in range(4)]:
        raise ValueError("Expected exactly shard_0.json through shard_3.json.")
    shard_payloads = [load_json(path) for path in shard_paths]
    if any(payload.get("status") != "complete" for payload in shard_payloads):
        raise ValueError("All four calibration shards must have status=complete.")
    if {
        (payload.get("shard_index"), payload.get("num_shards"))
        for payload in shard_payloads
    } != {(index, 4) for index in range(4)}:
        raise ValueError("Calibration shard indices or num_shards are invalid.")
    selected_ids = [
        sample_id
        for payload in shard_payloads
        for sample_id in payload.get("selected_sample_ids", [])
    ]
    if len(selected_ids) != 70 or set(selected_ids) != expected_ids:
        raise ValueError("Calibration shard selections do not partition the 70 prompts.")
    if any(payload.get("manifest_sha256") != sha256(manifest_path) for payload in shard_payloads):
        raise ValueError("Calibration shards used a different prompt manifest.")
    canonical_run_config = json.dumps(raw["run_config"], sort_keys=True)
    if any(
        json.dumps(payload.get("run_config"), sort_keys=True) != canonical_run_config
        for payload in shard_payloads
    ):
        raise ValueError("Calibration shard run_config differs from coefficients.json.")
    collection_sources = {
        (
            payload.get("wan_model_py_sha256"),
            payload.get("wan_text2video_py_sha256"),
        )
        for payload in shard_payloads
    }
    if len(collection_sources) != 1:
        raise ValueError("Calibration shards used different Wan2.2 source files.")
    model_sha, text2video_sha = next(iter(collection_sources))
    observed_source = {
        "wan/modules/model.py": model_sha,
        "wan/text2video.py": text2video_sha,
    }
    approved_sources = lock["wan22"]["approved_calibration_source_sha256"]
    source_variant = next(
        (
            name
            for name, expected in approved_sources.items()
            if observed_source == expected
        ),
        None,
    )
    equivalence_record = None
    if source_variant is None:
        equivalence_path = result_root / "SOURCE_EQUIVALENCE_REPORT.json"
        if not equivalence_path.is_file():
            raise ValueError(
                "Calibration used a noncanonical Wan2.2 source; a validated "
                "SOURCE_EQUIVALENCE_REPORT.json is required."
            )
        equivalence = load_json(equivalence_path)
        if (
            equivalence.get("schema")
            != "teacache4wan22_calibration_source_equivalence_v1"
            or equivalence.get("status") != "pass"
            or equivalence.get("calibration_source_sha256") != observed_source
            or equivalence.get("reference_source_sha256") not in approved_sources.values()
        ):
            raise ValueError("Invalid calibration source equivalence report.")
        equivalence_record = {
            "sha256": sha256(equivalence_path),
            "payload": equivalence,
        }

    protocol = {key: raw["run_config"][key] for key in PROTOCOL_KEYS}
    if protocol["use_ret_steps"] is not False:
        raise ValueError("Only non-retention TeaCache coefficients may be packaged.")
    stages = {}
    expected_stage_counts = {"high": 4340, "low": 2240}
    expected_unique_x = {"high": 31, "low": 16}
    for stage in ("high", "low"):
        fit = raw["primary"][stage]
        coefficients = [float(value) for value in fit["coefficients_descending"]]
        if len(coefficients) != 5 or not all(math.isfinite(value) for value in coefficients):
            raise ValueError(f"Invalid {stage} quartic coefficients.")
        scalar_fit_values = (fit["x_unique_count"], fit["x_min"], fit["x_max"])
        if not all(math.isfinite(float(value)) for value in scalar_fit_values):
            raise ValueError(f"Invalid {stage} fit-domain statistic.")
        if fit["overall_metrics"].get("count") != expected_stage_counts[stage]:
            raise ValueError(f"Unexpected {stage} primary fit count.")
        if fit["x_unique_count"] != expected_unique_x[stage]:
            raise ValueError(f"Unexpected {stage} unique-x count.")
        if {
            metrics.get("count") for metrics in fit["branch_metrics"].values()
        } != {expected_stage_counts[stage] // 2}:
            raise ValueError(f"Unexpected {stage} branch fit counts.")
        if len(fit["category_metrics"]) != 7 or {
            metrics.get("count") for metrics in fit["category_metrics"].values()
        } != {expected_stage_counts[stage] // 7}:
            raise ValueError(f"Unexpected {stage} category fit counts.")
        for metric_group in (
            [fit["overall_metrics"]],
            fit["branch_metrics"].values(),
            fit["category_metrics"].values(),
        ):
            for metrics in metric_group:
                if not all(
                    isinstance(value, (int, float)) and math.isfinite(float(value))
                    for value in metrics.values()
                ):
                    raise ValueError(f"Non-finite {stage} fit metric.")
        stages[stage] = {
            "coefficient_order": ["x^4", "x^3", "x^2", "x", "constant"],
            "coefficients_descending": coefficients,
            "x_unique_count": fit["x_unique_count"],
            "x_min": fit["x_min"],
            "x_max": fit["x_max"],
            "overall_metrics": fit["overall_metrics"],
            "branch_metrics": fit["branch_metrics"],
            "category_metrics": fit["category_metrics"],
        }

    category_counts = Counter(row["category"] for row in manifest)
    if len(category_counts) != 7 or set(category_counts.values()) != {10}:
        raise ValueError("Calibration manifest must contain seven categories of 10 prompts.")
    payload = {
        "schema": "teacache4wan22_coefficients_v1",
        "method": "ordinary quartic least squares on raw pooled cond/uncond observations",
        "formula": "y_hat = a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0",
        "x_definition": "adjacent relative-L1 of non-retention timestep embedding e",
        "y_definition": "adjacent relative-L1 of full-compute post-block/pre-head hidden state H",
        "protocol": protocol,
        "stages": stages,
        "calibration": {
            "prompt_count": 70,
            "category_counts": dict(sorted(category_counts.items())),
            "raw_forward_record_count": 7000,
            "within_stage_transition_count": 6720,
            "primary_fit_record_count": 6580,
            "prompt_source_repository": selection["source_repository"],
            "prompt_source_commit": selection["source_commit"],
            "selection_seed": selection["selection_seed"],
            "prompts_jsonl_sha256": sha256(manifest_path),
            "prompt_selection_sha256": sha256(selection_path),
            "raw_coefficients_sha256": sha256(coefficients_path),
            "fit_report_sha256": sha256(fit_report_path),
            "fit_software": fit_software,
            "sample_sha256": sample_hashes,
            "shard_metadata_sha256": {
                path.name: sha256(path) for path in shard_paths
            },
            "collection_source": {
                "variant": source_variant or "validated_equivalent_noncanonical",
                "file_sha256": observed_source,
                "torch_versions": sorted(
                    {payload.get("torch_version") for payload in shard_payloads}
                ),
                "equivalence_report": equivalence_record,
            },
        },
        "source_locks": {
            "wan22_commit": lock["wan22"]["commit"],
            "teacache_commit": lock["teacache"]["commit"],
            "t2v_compbench_commit": lock["t2v_compbench"]["commit"],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(args.output.resolve()),
                "sha256": sha256(args.output),
                "high_coefficients": stages["high"]["coefficients_descending"],
                "low_coefficients": stages["low"]["coefficients_descending"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
