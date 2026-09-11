#!/usr/bin/env python3
"""Validate and fit the ten-sample SeaCache speedup calibration scan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PROJECT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(DATA_PROJECT / "src"))

from ours4wan21_data.manifest import PROTOCOL  # noqa: E402
from ours4wan21_data.performance import read_json  # noqa: E402
from run_calibration import (  # noqa: E402
    COMPLETE_SCHEMA,
    NUM_SAMPLES,
    completion_valid,
    load_jsonl,
    result_paths,
    validate_rows,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def linear_fit(points: list[tuple[float, float]]) -> dict[str, float]:
    if len(points) < 2:
        raise ValueError("linear fit requires at least two points")
    x_mean = sum(x for x, _ in points) / len(points)
    y_mean = sum(y for _, y in points) / len(points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator <= 0:
        raise ValueError("linear fit x values have no variance")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    intercept = y_mean - slope * x_mean
    residual_ss = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    total_ss = sum((y - y_mean) ** 2 for _, y in points)
    r_squared = 1.0 if total_ss == 0 and residual_ss == 0 else 1.0 - residual_ss / total_ss
    rmse = math.sqrt(residual_ss / len(points))
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "rmse": rmse,
        "point_count": len(points),
    }


def select_fit_rows(rows: list[dict[str, float]], domain: tuple[float, float]) -> list[dict[str, float]]:
    ordered = sorted(rows, key=lambda row: row["speedup"])
    lower, upper = domain
    below = [row for row in ordered if row["speedup"] <= lower]
    above = [row for row in ordered if row["speedup"] >= upper]
    if not below or not above:
        raise ValueError(
            f"threshold scan does not bracket target speedup domain {domain}: "
            f"observed {ordered[0]['speedup']:.4f}-{ordered[-1]['speedup']:.4f}"
        )
    selected = [row for row in ordered if lower <= row["speedup"] <= upper]
    selected.extend((below[-1], above[0]))
    unique = {float(row["threshold"]): row for row in selected}
    result = sorted(unique.values(), key=lambda row: row["speedup"])
    if len(result) < 3:
        raise ValueError("fewer than three threshold levels support the requested fit domain")
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)


def analyze(root: Path, additional_roots: list[Path] | None = None) -> dict[str, Any]:
    root = root.resolve(strict=True)
    scan_roots = [root] + [path.resolve(strict=True) for path in (additional_roots or [])]
    if len(set(scan_roots)) != len(scan_roots):
        raise ValueError("calibration scan roots must be unique")

    scan_inputs: list[dict[str, Any]] = []
    sample_ids: list[str] | None = None
    seen_thresholds: set[float] = set()
    for scan_root in scan_roots:
        protocol = read_json(scan_root / "protocol.json")
        manifest_path = scan_root / "manifests" / "calibration_manifest.jsonl"
        rows = load_jsonl(manifest_path)
        scan_thresholds = tuple(float(value) for value in protocol["thresholds"])
        validate_rows(rows, scan_thresholds)
        current_sample_ids = [str(value) for value in protocol["sample_ids"]]
        if sample_ids is None:
            sample_ids = current_sample_ids
        elif current_sample_ids != sample_ids:
            raise ValueError(f"calibration sample IDs differ across scan roots: {scan_root}")
        duplicates = seen_thresholds.intersection(scan_thresholds)
        if duplicates:
            raise ValueError(f"duplicate thresholds across scan roots: {sorted(duplicates)}")
        seen_thresholds.update(scan_thresholds)
        failures = sorted((scan_root / "failures").glob("*.json")) if (scan_root / "failures").exists() else []
        if failures:
            raise RuntimeError(
                f"calibration contains failure records in {scan_root}: {[path.name for path in failures]}"
            )
        scan_inputs.append({
            "root": scan_root,
            "manifest_path": manifest_path,
            "rows": rows,
            "thresholds": scan_thresholds,
        })
    thresholds = tuple(sorted(seen_thresholds))

    per_sample: list[dict[str, Any]] = []
    for scan in scan_inputs:
        for row in scan["rows"]:
            paths = result_paths(scan["root"], row)
            if not completion_valid(paths, row):
                raise RuntimeError(f"incomplete calibration result: {row['trajectory_id']}")
            complete = read_json(paths["complete"])
            if complete.get("schema") != COMPLETE_SCHEMA:
                raise ValueError(f"unexpected completion schema: {paths['complete']}")
            timing = read_json(paths["timing"])
            performance = read_json(paths["performance"])
            trace = read_json(paths["trace"])
            baseline_performance = read_json(Path(complete["baseline_performance"]))
            baseline_seconds = float(baseline_performance["pipeline_generate_wall_seconds"])
            candidate_seconds = float(performance["pipeline_generate_wall_seconds"])
            reuse = int(trace["reuse"])
            recompute = int(trace["recompute"])
            if reuse + recompute != 100:
                raise ValueError(f"skip counts do not close: {row['trajectory_id']}")
            recomputed_speedup = baseline_seconds / candidate_seconds
            if not math.isclose(recomputed_speedup, float(complete["inference_latency_speedup"]), rel_tol=1e-12):
                raise ValueError(f"speedup mismatch: {row['trajectory_id']}")
            per_sample.append({
                "source_root": str(scan["root"]),
                "threshold": float(row["fixed_threshold"]),
                "sample_id": str(row["sample_id"]),
                "prompt_rank": int(row["prompt_rank"]),
                "shard_index": int(row["shard_index"]),
                "baseline_inference_seconds": baseline_seconds,
                "candidate_inference_seconds": candidate_seconds,
                "inference_speedup": recomputed_speedup,
                "baseline_dit_cuda_seconds": float(baseline_performance["dit_cuda_seconds"]),
                "candidate_dit_cuda_seconds": float(performance["dit_cuda_seconds"]),
                "baseline_t5_cuda_seconds": float(baseline_performance["t5_cuda_seconds"]),
                "candidate_t5_cuda_seconds": float(performance["t5_cuda_seconds"]),
                "baseline_vae_decode_cuda_seconds": float(baseline_performance["vae_decode_cuda_seconds"]),
                "candidate_vae_decode_cuda_seconds": float(performance["vae_decode_cuda_seconds"]),
                "baseline_dit_tflops": float(baseline_performance["estimated_dit_tflops_per_video"]),
                "candidate_dit_tflops": float(performance["estimated_dit_tflops_per_video"]),
                "t5_tflops": float(performance["estimated_t5_tflops_per_video"]),
                "vae_decode_tflops": float(performance["estimated_vae_decode_tflops_per_video"]),
                "reuse_branch_calls": reuse,
                "recompute_branch_calls": recompute,
                "equivalent_reuse_steps": reuse / 2.0,
                "both_reuse_steps": int(complete["both_reuse_steps"]),
                "both_recompute_steps": int(complete["both_recompute_steps"]),
                "mixed_steps": int(complete["mixed_steps"]),
            })

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in per_sample:
        grouped[float(row["threshold"])].append(row)
    summary: list[dict[str, Any]] = []
    for threshold in thresholds:
        group = grouped[threshold]
        if len(group) != NUM_SAMPLES:
            raise ValueError(f"threshold {threshold} has {len(group)} samples, expected {NUM_SAMPLES}")
        sum_baseline = sum(float(row["baseline_inference_seconds"]) for row in group)
        sum_candidate = sum(float(row["candidate_inference_seconds"]) for row in group)
        sum_baseline_dit = sum(float(row["baseline_dit_tflops"]) for row in group)
        sum_candidate_dit = sum(float(row["candidate_dit_tflops"]) for row in group)
        speedups = [float(row["inference_speedup"]) for row in group]
        mean_speedup = sum(speedups) / NUM_SAMPLES
        speedup_std = math.sqrt(sum((value - mean_speedup) ** 2 for value in speedups) / NUM_SAMPLES)
        summary.append({
            "threshold": threshold,
            "sample_count": NUM_SAMPLES,
            "baseline_inference_seconds_sum": sum_baseline,
            "candidate_inference_seconds_sum": sum_candidate,
            "speedup": sum_baseline / sum_candidate,
            "per_sample_speedup_mean": mean_speedup,
            "per_sample_speedup_std_population": speedup_std,
            "dit_flops_speedup": sum_baseline_dit / sum_candidate_dit,
            "candidate_dit_tflops_mean": sum_candidate_dit / NUM_SAMPLES,
            "candidate_t5_tflops_mean": sum(float(row["t5_tflops"]) for row in group) / NUM_SAMPLES,
            "candidate_vae_decode_tflops_mean": sum(float(row["vae_decode_tflops"]) for row in group) / NUM_SAMPLES,
            "candidate_t5_cuda_seconds_mean": sum(float(row["candidate_t5_cuda_seconds"]) for row in group) / NUM_SAMPLES,
            "candidate_dit_cuda_seconds_mean": sum(float(row["candidate_dit_cuda_seconds"]) for row in group) / NUM_SAMPLES,
            "candidate_vae_decode_cuda_seconds_mean": sum(float(row["candidate_vae_decode_cuda_seconds"]) for row in group) / NUM_SAMPLES,
            "reuse_branch_calls_mean": sum(float(row["reuse_branch_calls"]) for row in group) / NUM_SAMPLES,
            "equivalent_reuse_steps_mean": sum(float(row["equivalent_reuse_steps"]) for row in group) / NUM_SAMPLES,
            "both_reuse_steps_mean": sum(float(row["both_reuse_steps"]) for row in group) / NUM_SAMPLES,
            "mixed_steps_mean": sum(float(row["mixed_steps"]) for row in group) / NUM_SAMPLES,
        })

    fit_rows = select_fit_rows(summary, (1.5, 3.5))
    threshold_fit = linear_fit([(float(row["speedup"]), float(row["threshold"])) for row in fit_rows])
    branch_skip_fit = linear_fit([(float(row["speedup"]), float(row["reuse_branch_calls_mean"])) for row in fit_rows])
    step_skip_fit = linear_fit([(float(row["speedup"]), float(row["equivalent_reuse_steps_mean"])) for row in fit_rows])
    for fit in (threshold_fit, branch_skip_fit, step_skip_fit):
        fit["equation_y_of_speedup"] = f"y = {fit['slope']:.10f} * speedup + {fit['intercept']:.10f}"
    endpoints = [1.5, 3.5]
    mapped_thresholds = [threshold_fit["intercept"] + threshold_fit["slope"] * speedup for speedup in endpoints]
    mapped_branch_skips = [branch_skip_fit["intercept"] + branch_skip_fit["slope"] * speedup for speedup in endpoints]
    if threshold_fit["slope"] <= 0 or not all(thresholds[0] <= value <= thresholds[-1] for value in mapped_thresholds):
        raise ValueError(f"linear threshold fit is invalid for materialization: {mapped_thresholds}")

    analysis_dir = root / "analysis"
    per_sample_fields = list(per_sample[0])
    summary_fields = list(summary[0])
    write_csv(analysis_dir / "per_sample.csv", per_sample, per_sample_fields)
    summary_path = analysis_dir / "threshold_summary.csv"
    write_csv(summary_path, summary, summary_fields)
    fits = {
        "schema": "ours4wan21_seacache_linear_fits_v1",
        "analysis_script_sha256": sha256(Path(__file__)),
        "sample_count": NUM_SAMPLES,
        "scan_roots": [str(scan["root"]) for scan in scan_inputs],
        "threshold_level_count": len(summary),
        "fit_thresholds": [row["threshold"] for row in fit_rows],
        "fit_observed_speedup_range": [min(row["speedup"] for row in fit_rows), max(row["speedup"] for row in fit_rows)],
        "target_speedup_domain": endpoints,
        "speedup_to_threshold": threshold_fit,
        "speedup_to_reuse_branch_calls_out_of_100": branch_skip_fit,
        "speedup_to_equivalent_reuse_steps_out_of_50": step_skip_fit,
        "endpoint_predictions": {
            "speedups": endpoints,
            "mean_thresholds": mapped_thresholds,
            "reuse_branch_calls": mapped_branch_skips,
            "equivalent_reuse_steps": [value / 2.0 for value in mapped_branch_skips],
        },
        "aggregation": "threshold-level ratio of sums over ten matched samples",
    }
    atomic_json(analysis_dir / "linear_fits.json", fits)
    mapping = {
        "schema": "ours4wan21_speed_threshold_mapping_v1",
        "calibration_status": "calibrated",
        "model": "Wan2.1-T2V-1.3B",
        "protocol": {key: value for key, value in PROTOCOL.items() if key != "model"},
        "target_speedup_domain": endpoints,
        "fit_source": str(summary_path.resolve()),
        "fit_source_sha256": sha256(summary_path),
        "threshold_bounds": [thresholds[0], thresholds[-1]],
        "mapping": {
            "kind": "monotone_piecewise_linear",
            "speedups": endpoints,
            "mean_thresholds": mapped_thresholds,
            "provenance": "two endpoint knots evaluated from OLS threshold = a * speedup + b",
        },
        "skip_fit": {
            "count_unit": "reused CFG-branch DiT forward calls out of 100 per video",
            "slope": branch_skip_fit["slope"],
            "intercept": branch_skip_fit["intercept"],
            "r_squared": branch_skip_fit["r_squared"],
            "equivalent_50_step_scale": "reuse_branch_calls / 2",
        },
        "fit_details": str((analysis_dir / "linear_fits.json").resolve()),
        "notes": "Ten-sample fixed-threshold SeaCache calibration; full-pipeline speedup includes GPU T5, denoising, and VAE decode.",
    }
    mapping_path = analysis_dir / "speed_threshold_mapping.calibrated.json"
    atomic_json(mapping_path, mapping)
    quality = {
        "status": "pass",
        "analysis_script_sha256": sha256(Path(__file__)),
        "expected_candidates": sum(len(scan["rows"]) for scan in scan_inputs),
        "complete_candidates": len(per_sample),
        "samples_per_threshold": NUM_SAMPLES,
        "thresholds": list(thresholds),
        "scan_roots": [str(scan["root"]) for scan in scan_inputs],
        "failure_records": 0,
        "timing_calls_per_candidate": 100,
        "action_count_closure": "reuse + recompute = 100 for every candidate",
        "target_domain_bracketed": True,
        "fit_source_sha256": sha256(summary_path),
        "manifest_sha256": {
            str(scan["root"]): sha256(scan["manifest_path"])
            for scan in scan_inputs
        },
    }
    atomic_json(analysis_dir / "data_quality.json", quality)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        speed_grid = [1.5 + index * (3.5 - 1.5) / 100 for index in range(101)]
        axes[0].scatter([row["speedup"] for row in summary], [row["threshold"] for row in summary], color="#1665D8")
        axes[0].plot(speed_grid, [threshold_fit["intercept"] + threshold_fit["slope"] * x for x in speed_grid], color="#D94841")
        axes[0].set(xlabel="Full-pipeline speedup (x)", ylabel="SeaCache threshold", title=f"Threshold fit, R²={threshold_fit['r_squared']:.4f}")
        axes[0].grid(alpha=0.25)
        axes[1].scatter([row["speedup"] for row in summary], [row["equivalent_reuse_steps_mean"] for row in summary], color="#1665D8")
        axes[1].plot(speed_grid, [step_skip_fit["intercept"] + step_skip_fit["slope"] * x for x in speed_grid], color="#D94841")
        axes[1].set(xlabel="Full-pipeline speedup (x)", ylabel="Equivalent skipped steps / 50", title=f"Skip fit, R²={step_skip_fit['r_squared']:.4f}")
        axes[1].grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(analysis_dir / "calibration_fits.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass

    report_lines = [
        "# SeaCache speedup calibration",
        "",
        f"All {len(per_sample)} candidates ({len(thresholds)} thresholds x {NUM_SAMPLES} samples across {len(scan_inputs)} scan roots) passed completeness and count-closure checks.",
        "",
        "## Linear fits",
        "",
        f"- Threshold: `threshold = {threshold_fit['slope']:.8f} * speedup + {threshold_fit['intercept']:.8f}` (R²={threshold_fit['r_squared']:.6f}, RMSE={threshold_fit['rmse']:.6f}).",
        f"- Skipped DiT calls: `skip_calls = {branch_skip_fit['slope']:.8f} * speedup + {branch_skip_fit['intercept']:.8f}` out of 100 (R²={branch_skip_fit['r_squared']:.6f}, RMSE={branch_skip_fit['rmse']:.6f}).",
        f"- Equivalent 50-step count: `skip_steps = {step_skip_fit['slope']:.8f} * speedup + {step_skip_fit['intercept']:.8f}` (R²={step_skip_fit['r_squared']:.6f}).",
        "",
        "## Endpoint calibration",
        "",
        "| target speedup | fitted threshold | fitted skip calls / 100 | equivalent skipped steps / 50 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for speedup, threshold, skip_calls in zip(endpoints, mapped_thresholds, mapped_branch_skips):
        report_lines.append(f"| {speedup:.2f}x | {threshold:.6f} | {skip_calls:.3f} | {skip_calls / 2.0:.3f} |")
    report_lines.extend((
        "",
        "## Scan summary",
        "",
        "| threshold | speedup | inference s/video | skip calls / 100 | equivalent skip steps / 50 | T5 s | DiT s | VAE s | T5 TFLOPs/video | DiT TFLOPs/video | VAE TFLOPs/video |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for row in summary:
        report_lines.append(
            f"| {row['threshold']:.3f} | {row['speedup']:.4f}x | "
            f"{row['candidate_inference_seconds_sum'] / NUM_SAMPLES:.3f} | "
            f"{row['reuse_branch_calls_mean']:.2f} | {row['equivalent_reuse_steps_mean']:.2f} | "
            f"{row['candidate_t5_cuda_seconds_mean']:.3f} | {row['candidate_dit_cuda_seconds_mean']:.3f} | "
            f"{row['candidate_vae_decode_cuda_seconds_mean']:.3f} | "
            f"{row['candidate_t5_tflops_mean']:.3f} | {row['candidate_dit_tflops_mean']:.3f} | "
            f"{row['candidate_vae_decode_tflops_mean']:.3f} |"
        )
    report_lines.extend((
        "",
        "The primary skip unit is an actually omitted cond/uncond DiT forward call. Dividing by two gives the equivalent count on the 50-denoising-step scale; mixed CFG-branch decisions are therefore represented without rounding.",
        "",
    ))
    (analysis_dir / "calibration_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return {"fits": fits, "mapping": mapping, "quality": quality}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--additional-root",
        type=Path,
        action="append",
        default=[],
        help="Additional fixed-threshold scan root to merge into the fit (repeatable).",
    )
    args = parser.parse_args()
    result = analyze(args.output_root, args.additional_root)
    print(json.dumps({
        "status": "complete",
        "speedup_to_threshold": result["fits"]["speedup_to_threshold"],
        "speedup_to_skip_calls": result["fits"]["speedup_to_reuse_branch_calls_out_of_100"],
        "mapping": result["mapping"]["mapping"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
