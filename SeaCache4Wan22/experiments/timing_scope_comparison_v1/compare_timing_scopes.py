#!/usr/bin/env python3
"""Compare legacy compute-only and current full-generate timing scopes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXP_ROOT = Path("/all/yiran07-disk3/huteng_data/exp").resolve()
COMPUTE_RE = re.compile(r"inference_compute_elapsed_seconds=([0-9.]+)")
TRANSFER_RE = re.compile(r"inference_weight_transfer_elapsed_seconds=([0-9.]+)")
WALL_RE = re.compile(r"generation_wall_elapsed_seconds=([0-9.]+)")
BASELINE_RE = re.compile(r"baseline_(?P<sample>.+)\.log$")
CANDIDATE_RE = re.compile(
    r"seacache_th_(?P<label>[0-9]+p[0-9]+)_(?P<sample>.+)\.log$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-baselines", type=int, default=30)
    parser.add_argument("--expected-thresholds", type=int, default=9)
    parser.add_argument("--targets", type=float, nargs="+", default=[1.8, 2.4, 3.0])
    return parser.parse_args()


def external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as exc:
        raise ValueError(f"output must be under {EXP_ROOT}: {resolved}") from exc
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one(pattern: re.Pattern[str], text: str, path: Path) -> float:
    values = pattern.findall(text)
    if len(values) != 1:
        raise ValueError(f"expected one {pattern.pattern!r} match in {path}, found {len(values)}")
    value = float(values[0])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid time {value} in {path}")
    return value


def threshold_from_label(label: str) -> float:
    return float(label.replace("p", ".", 1))


def parse_log(path: Path) -> dict[str, Any] | None:
    baseline = BASELINE_RE.fullmatch(path.name)
    candidate = CANDIDATE_RE.fullmatch(path.name)
    if baseline is None and candidate is None:
        return None
    text = path.read_text(encoding="utf-8", errors="strict")
    compute = one(COMPUTE_RE, text, path)
    transfer = one(TRANSFER_RE, text, path)
    wall = one(WALL_RE, text, path)
    residual = wall - compute - transfer
    if residual < -0.01:
        raise ValueError(
            f"wall time is smaller than compute+transfer by {-residual:.6f}s: {path}"
        )
    if baseline is not None:
        condition = "baseline"
        sample_id = baseline.group("sample")
        threshold_label = None
        threshold = None
    else:
        assert candidate is not None
        condition = "seacache"
        sample_id = candidate.group("sample")
        threshold_label = candidate.group("label")
        threshold = threshold_from_label(threshold_label)
    return {
        "condition": condition,
        "sample_id": sample_id,
        "threshold_label": threshold_label,
        "threshold": threshold,
        "legacy_compute_seconds": compute,
        "transfer_seconds": transfer,
        "full_generate_wall_seconds": wall,
        "wall_minus_compute_seconds": wall - compute,
        "residual_seconds": residual,
        "wall_over_compute_fraction": wall / compute - 1.0,
        "log_path": str(path.resolve()),
        "log_sha256": sha256(path),
    }


def quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute quantile of empty values")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    compute = [float(row["legacy_compute_seconds"]) for row in rows]
    transfer = [float(row["transfer_seconds"]) for row in rows]
    wall = [float(row["full_generate_wall_seconds"]) for row in rows]
    residual = [float(row["residual_seconds"]) for row in rows]
    relative = [float(row["wall_over_compute_fraction"]) for row in rows]
    return {
        "n": len(rows),
        "legacy_compute_seconds": {
            "total": sum(compute), "mean": sum(compute) / len(rows),
        },
        "transfer_seconds": {
            "total": sum(transfer), "mean": sum(transfer) / len(rows),
        },
        "full_generate_wall_seconds": {
            "total": sum(wall), "mean": sum(wall) / len(rows),
        },
        "wall_minus_compute_seconds": {
            "total": sum(wall) - sum(compute),
            "mean": (sum(wall) - sum(compute)) / len(rows),
            "p50": quantile((w - c for w, c in zip(wall, compute)), 0.5),
            "p95": quantile((w - c for w, c in zip(wall, compute)), 0.95),
        },
        "wall_over_compute_fraction": {
            "ratio_of_sums": sum(wall) / sum(compute) - 1.0,
            "mean": sum(relative) / len(relative),
            "p50": quantile(relative, 0.5),
            "p95": quantile(relative, 0.95),
        },
        "residual_seconds": {
            "mean": sum(residual) / len(residual),
            "p95": quantile(residual, 0.95),
        },
    }


def interpolate_threshold(points: list[dict[str, Any]], target: float, field: str) -> dict[str, Any]:
    ordered = sorted(points, key=lambda row: float(row["threshold"]))
    for left, right in zip(ordered, ordered[1:]):
        lo = float(left[field])
        hi = float(right[field])
        if lo <= target <= hi and hi > lo:
            fraction = (target - lo) / (hi - lo)
            threshold = float(left["threshold"]) + fraction * (
                float(right["threshold"]) - float(left["threshold"])
            )
            return {
                "status": "interpolated",
                "threshold": threshold,
                "bracket_thresholds": [left["threshold"], right["threshold"]],
                "bracket_speedups": [lo, hi],
            }
    return {
        "status": "outside_observed_range",
        "threshold": None,
        "observed_speedup_range": [float(ordered[0][field]), float(ordered[-1][field])],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main() -> None:
    args = parse_args()
    source = args.source_root.expanduser().resolve(strict=True)
    output = external(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    if args.expected_baselines < 1 or args.expected_thresholds < 1:
        raise ValueError("expected counts must be positive")
    if not args.targets or any(not math.isfinite(value) or value <= 1 for value in args.targets):
        raise ValueError("targets must be finite speedups greater than one")

    log_paths = sorted(source.glob("shard_gpu*/logs/*.log"))
    parsed = [row for path in log_paths if (row := parse_log(path)) is not None]
    baselines = [row for row in parsed if row["condition"] == "baseline"]
    candidates = [row for row in parsed if row["condition"] == "seacache"]
    baseline_by_id = {str(row["sample_id"]): row for row in baselines}
    if len(baseline_by_id) != len(baselines):
        raise ValueError("duplicate baseline sample IDs")
    if len(baselines) != args.expected_baselines:
        raise ValueError(f"expected {args.expected_baselines} baselines, found {len(baselines)}")

    by_threshold: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_threshold[float(row["threshold"])].append(row)
    if len(by_threshold) != args.expected_thresholds:
        raise ValueError(f"expected {args.expected_thresholds} thresholds, found {len(by_threshold)}")

    threshold_rows: list[dict[str, Any]] = []
    for threshold in sorted(by_threshold):
        rows = by_threshold[threshold]
        ids = {str(row["sample_id"]) for row in rows}
        if len(rows) != args.expected_baselines or ids != set(baseline_by_id):
            raise ValueError(f"threshold {threshold} is not paired to every baseline")
        paired_baselines = [baseline_by_id[str(row["sample_id"])] for row in rows]
        old_base = sum(float(row["legacy_compute_seconds"]) for row in paired_baselines)
        old_sea = sum(float(row["legacy_compute_seconds"]) for row in rows)
        wall_base = sum(float(row["full_generate_wall_seconds"]) for row in paired_baselines)
        wall_sea = sum(float(row["full_generate_wall_seconds"]) for row in rows)
        old_speedup = old_base / old_sea
        wall_speedup = wall_base / wall_sea
        run_summary = summarize_runs(rows)
        threshold_rows.append({
            "threshold": threshold,
            "n": len(rows),
            "legacy_compute_speedup": old_speedup,
            "full_generate_wall_speedup": wall_speedup,
            "speedup_absolute_delta": wall_speedup - old_speedup,
            "speedup_relative_delta_fraction": wall_speedup / old_speedup - 1.0,
            "baseline_legacy_compute_total_seconds": old_base,
            "baseline_full_generate_wall_total_seconds": wall_base,
            "seacache_legacy_compute_total_seconds": old_sea,
            "seacache_full_generate_wall_total_seconds": wall_sea,
            "seacache_wall_minus_compute_mean_seconds": run_summary["wall_minus_compute_seconds"]["mean"],
            "seacache_wall_over_compute_ratio_of_sums_fraction": run_summary["wall_over_compute_fraction"]["ratio_of_sums"],
        })

    target_rows = []
    for target in args.targets:
        legacy = interpolate_threshold(threshold_rows, target, "legacy_compute_speedup")
        current = interpolate_threshold(threshold_rows, target, "full_generate_wall_speedup")
        legacy_threshold = legacy["threshold"]
        current_threshold = current["threshold"]
        target_rows.append({
            "target_speedup": target,
            "legacy_compute_threshold_estimate": legacy_threshold,
            "full_generate_wall_threshold_estimate": current_threshold,
            "threshold_absolute_shift": (
                current_threshold - legacy_threshold
                if current_threshold is not None and legacy_threshold is not None
                else None
            ),
            "threshold_relative_shift_fraction": (
                current_threshold / legacy_threshold - 1.0
                if current_threshold is not None and legacy_threshold is not None
                else None
            ),
            "legacy_interpolation": legacy,
            "full_generate_wall_interpolation": current,
        })

    baseline_summary = summarize_runs(baselines)
    candidate_summary = summarize_runs(candidates)
    payload = {
        "schema": "seacache4wan22_timing_scope_comparison_v1",
        "status": "pass",
        "source_root": str(source),
        "scope_definitions": {
            "legacy_compute_seconds": (
                "T5, denoising/scheduler, and VAE compute spans synchronized inside WanT2V.generate; "
                "explicitly excludes model weight transfers/offload"
            ),
            "full_generate_wall_seconds": (
                "wall time around WanT2V.generate; includes T5, denoising/cache/CFG/scheduler, "
                "in-generate transfers/offload, and VAE decode"
            ),
        },
        "counts": {
            "baseline_runs": len(baselines),
            "seacache_runs": len(candidates),
            "thresholds": len(by_threshold),
            "total_runs": len(parsed),
        },
        "baseline_summary": baseline_summary,
        "seacache_all_thresholds_summary": candidate_summary,
        "per_threshold": threshold_rows,
        "target_threshold_interpolation": target_rows,
        "limitations": [
            "The archived generation_wall_elapsed_seconds has the same inclusion boundary as current pipeline_generate_wall_seconds but predates current component/DiT instrumentation.",
            "Threshold estimates use piecewise-linear interpolation of the archived 30-prompt threshold grid and require final calibration under the current runner before Vbench200.",
            "This audit compares latency definitions on archived runs; it does not claim code-version runtime equivalence.",
        ],
    }

    output.mkdir(parents=True)
    (output / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    run_fields = [
        "condition", "sample_id", "threshold_label", "threshold",
        "legacy_compute_seconds", "transfer_seconds", "full_generate_wall_seconds",
        "wall_minus_compute_seconds", "residual_seconds", "wall_over_compute_fraction",
        "log_path", "log_sha256",
    ]
    write_csv(output / "per_run.csv", parsed, run_fields)
    threshold_fields = [
        "threshold", "n", "legacy_compute_speedup", "full_generate_wall_speedup",
        "speedup_absolute_delta", "speedup_relative_delta_fraction",
        "baseline_legacy_compute_total_seconds", "baseline_full_generate_wall_total_seconds",
        "seacache_legacy_compute_total_seconds", "seacache_full_generate_wall_total_seconds",
        "seacache_wall_minus_compute_mean_seconds",
        "seacache_wall_over_compute_ratio_of_sums_fraction",
    ]
    write_csv(output / "per_threshold.csv", threshold_rows, threshold_fields)
    target_fields = [
        "target_speedup", "legacy_compute_threshold_estimate",
        "full_generate_wall_threshold_estimate", "threshold_absolute_shift",
        "threshold_relative_shift_fraction",
    ]
    write_csv(output / "target_thresholds.csv", target_rows, target_fields)

    lines = [
        "# SeaCache4Wan22 timing-scope comparison",
        "",
        f"Source: `{source}`",
        "",
        "The legacy headline excludes in-generate model transfers/offload. The current",
        "headline covers the complete CUDA-synchronized `WanT2V.generate()` wall time.",
        "",
        "## Aggregate scope difference",
        "",
        "| Condition | n | Compute mean (s) | Full wall mean (s) | Added mean (s) | Added ratio-of-sums |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, summary in (("Baseline", baseline_summary), ("SeaCache, all thresholds", candidate_summary)):
        lines.append(
            f"| {label} | {summary['n']} | {summary['legacy_compute_seconds']['mean']:.3f} | "
            f"{summary['full_generate_wall_seconds']['mean']:.3f} | "
            f"{summary['wall_minus_compute_seconds']['mean']:.3f} | "
            f"{summary['wall_over_compute_fraction']['ratio_of_sums'] * 100:.3f}% |"
        )
    lines.extend([
        "",
        "## Speedup by threshold",
        "",
        "| Threshold | Legacy compute-only | Full generate wall | Absolute change | Relative change |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in threshold_rows:
        lines.append(
            f"| {row['threshold']:.2f} | {row['legacy_compute_speedup']:.4f}x | "
            f"{row['full_generate_wall_speedup']:.4f}x | "
            f"{row['speedup_absolute_delta']:+.4f}x | "
            f"{row['speedup_relative_delta_fraction'] * 100:+.3f}% |"
        )
    lines.extend([
        "",
        "## Requested-speed threshold interpolation",
        "",
        "| Target | Legacy threshold | Full-wall threshold | Shift |",
        "|---:|---:|---:|---:|",
    ])
    for row in target_rows:
        lines.append(
            f"| {row['target_speedup']:.1f}x | {row['legacy_compute_threshold_estimate']:.5f} | "
            f"{row['full_generate_wall_threshold_estimate']:.5f} | "
            f"{row['threshold_absolute_shift']:+.5f} "
            f"({row['threshold_relative_shift_fraction'] * 100:+.2f}%) |"
        )
    lines.extend([
        "",
        "These threshold values are interpolation estimates from the archived 30-prompt grid,",
        "not frozen Vbench200 settings. A small current-runner calibration is still required.",
        "The archive predates current component/DiT instrumentation, so this result isolates the",
        "timing-boundary change but does not measure instrumentation overhead.",
        "",
        "## Files",
        "",
        "- `comparison.json`: canonical result and limitations.",
        "- `per_run.csv`: all paired timing fields and source-log hashes.",
        "- `per_threshold.csv`: ratio-of-sums speedups.",
        "- `target_thresholds.csv`: requested-speed interpolation.",
    ])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = []
    for name in ("README.md", "comparison.json", "per_run.csv", "per_threshold.csv", "target_thresholds.csv"):
        hashes.append(f"{sha256(output / name)}  {name}")
    (output / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output_dir": str(output), "counts": payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()
