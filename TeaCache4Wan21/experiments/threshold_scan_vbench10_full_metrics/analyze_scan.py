#!/usr/bin/env python3
"""Validate and aggregate the VBench10 full-metrics threshold scan."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


TARGETS = (1.8, 2.4, 3.0)
STAGE_FIELDS = ("text_encoding", "denoising_core", "vae_decode", "pipeline_other")
QUALITY_FIELDS = (
    "psnr_rgb_db",
    "ssim_rgb",
    "lpips_alex_v0_1_spatial",
)
TFLOP_DIVISOR = 1_000_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative; got {value!r}")
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty list")
    return {
        "count": len(values),
        "total": sum(values),
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def threshold_label(threshold: float) -> str:
    return f"threshold_{threshold:.4f}".replace(".", "p")


def load_flops_profile(root: Path) -> tuple[dict[str, Any], float, float, int]:
    profile = read_json(root / "profiling" / "calflops.json")
    if profile.get("input", {}).get("video_shape_fhw") != [81, 480, 832]:
        raise ValueError("Calflops profile shape does not match 81x480x832")
    forward = profile.get("per_model_forward")
    if not isinstance(forward, dict):
        raise ValueError("Calflops profile lacks per_model_forward")
    full_flops = finite_nonnegative(forward.get("estimated_full_flops"), "full FLOPs")
    always_on_flops = finite_nonnegative(
        forward.get("estimated_always_on_flops"), "always-on FLOPs"
    )
    block_count = int(profile.get("input", {}).get("transformer_blocks", 0))
    if block_count < 1 or always_on_flops > full_flops:
        raise ValueError("invalid Calflops profile")
    return profile, full_flops, always_on_flops, block_count


def estimate_trace_flops(
    calls: list[dict[str, Any]],
    *,
    full_flops: float,
    always_on_flops: float,
    block_count: int,
) -> float:
    transformer_flops = full_flops - always_on_flops
    total = 0.0
    for index, call in enumerate(calls):
        blocks = int(call.get("blocks_executed", -1))
        if not 0 <= blocks <= block_count:
            raise ValueError(f"invalid blocks_executed for call {index}: {blocks}")
        total += always_on_flops + transformer_flops * blocks / block_count
    return total


def validate_timing(
    path: Path,
    *,
    expected_implementation: str,
    full_flops: float,
    always_on_flops: float,
    block_count: int,
) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != 2 or payload.get("status") != "success":
        raise ValueError(f"invalid staged timing trace: {path}")
    if payload.get("implementation") != expected_implementation:
        raise ValueError(f"implementation mismatch: {path}")
    if payload.get("model_forward_call_count") != 100:
        raise ValueError(f"expected 100 DiT calls: {path}")
    calls = payload.get("calls")
    if not isinstance(calls, list) or len(calls) != 100:
        raise ValueError(f"missing per-call trace: {path}")
    full_calls = int(payload.get("full_compute_forward_calls", -1))
    reuse_calls = int(payload.get("reuse_forward_calls", -1))
    if full_calls + reuse_calls != 100:
        raise ValueError(f"full/reuse counts do not sum to 100: {path}")
    pipeline_seconds = finite_nonnegative(
        payload.get("pipeline_generate_wall_seconds"), f"pipeline latency {path}"
    )
    dit_cuda_seconds = finite_nonnegative(
        payload.get("model_forward_cuda_seconds"), f"DiT CUDA latency {path}"
    )
    stages = payload.get("stage_wall_seconds")
    if not isinstance(stages, dict):
        raise ValueError(f"missing stage_wall_seconds: {path}")
    stage_values = {
        field: finite_nonnegative(stages.get(field), f"{field} {path}")
        for field in STAGE_FIELDS
    }
    if not math.isclose(
        sum(stage_values.values()), pipeline_seconds, rel_tol=1e-8, abs_tol=1e-5
    ):
        raise ValueError(f"module timings do not reconcile to pipeline latency: {path}")
    stage_calls = payload.get("stage_calls")
    if not isinstance(stage_calls, dict):
        raise ValueError(f"missing stage call trace: {path}")
    if len(stage_calls.get("text_encoder", [])) != 2:
        raise ValueError(f"expected two T5 calls: {path}")
    if len(stage_calls.get("vae_decode", [])) != 1:
        raise ValueError(f"expected one VAE decode call: {path}")
    estimated_flops = estimate_trace_flops(
        calls,
        full_flops=full_flops,
        always_on_flops=always_on_flops,
        block_count=block_count,
    )
    return {
        "payload": payload,
        "pipeline_seconds": pipeline_seconds,
        "dit_cuda_seconds": dit_cuda_seconds,
        "stages": stage_values,
        "full_calls": full_calls,
        "reuse_calls": reuse_calls,
        "estimated_dit_flops": estimated_flops,
        "estimated_dit_tflops": estimated_flops / TFLOP_DIVISOR,
        "estimated_achieved_dit_tflops_per_second": (
            estimated_flops / dit_cuda_seconds / TFLOP_DIVISOR
        ),
    }


def load_condition_timings(
    condition_dir: Path,
    sample_ids: list[str],
    *,
    expected_implementation: str,
    full_flops: float,
    always_on_flops: float,
    block_count: int,
) -> dict[str, dict[str, Any]]:
    observed_videos = {path.stem for path in (condition_dir / "videos").glob("*.mp4")}
    if observed_videos != set(sample_ids):
        raise ValueError(f"video set mismatch: {condition_dir}")
    return {
        sample_id: validate_timing(
            condition_dir / "timings" / f"{sample_id}.json",
            expected_implementation=expected_implementation,
            full_flops=full_flops,
            always_on_flops=always_on_flops,
            block_count=block_count,
        )
        for sample_id in sample_ids
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_quality(condition_dir: Path, sample_ids: list[str]) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    metrics_dir = condition_dir / "metrics" / "video_metrics"
    summary = read_json(metrics_dir / "summary.json")
    if summary.get("video_count") != len(sample_ids):
        raise ValueError(f"metric video count mismatch: {metrics_dir}")
    if summary.get("frame_count_total") != len(sample_ids) * 81:
        raise ValueError(f"metric frame count mismatch: {metrics_dir}")
    if summary.get("selected_metrics") != ["psnr", "ssim", "lpips"]:
        raise ValueError(f"metric selection mismatch: {metrics_dir}")
    per_video_rows = read_csv_rows(metrics_dir / "per_video.csv")
    per_frame_rows = read_csv_rows(metrics_dir / "per_frame.csv")
    by_id = {row["video_id"]: row for row in per_video_rows}
    if set(by_id) != set(sample_ids) or len(per_frame_rows) != len(sample_ids) * 81:
        raise ValueError(f"per-video/per-frame metric completeness failure: {metrics_dir}")
    return summary, by_id


def summarize_timing_condition(timings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pipeline = [float(row["pipeline_seconds"]) for row in timings.values()]
    dit_cuda = [float(row["dit_cuda_seconds"]) for row in timings.values()]
    tflops = [float(row["estimated_dit_tflops"]) for row in timings.values()]
    total_flops = sum(float(row["estimated_dit_flops"]) for row in timings.values())
    stage_summary = {
        field: stats([float(row["stages"][field]) for row in timings.values()])
        for field in STAGE_FIELDS
    }
    full_calls = sum(int(row["full_calls"]) for row in timings.values())
    reuse_calls = sum(int(row["reuse_calls"]) for row in timings.values())
    return {
        "pipeline_generate_wall_seconds": stats(pipeline),
        "module_wall_seconds": stage_summary,
        "dit_cuda_seconds": stats(dit_cuda),
        "full_compute_forward_calls": full_calls,
        "reuse_forward_calls": reuse_calls,
        "reuse_fraction": reuse_calls / (full_calls + reuse_calls),
        "estimated_dit_tflops_per_video": stats(tflops),
        "estimated_dit_total_tflops": total_flops / TFLOP_DIVISOR,
        "estimated_achieved_dit_tflops_per_second_ratio_of_sums": (
            total_flops / sum(dit_cuda) / TFLOP_DIVISOR
        ),
    }


def quality_means(summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("quality summary lacks metrics")
    return {
        field: finite_nonnegative(metrics.get(field, {}).get("mean"), field)
        for field in QUALITY_FIELDS
    }


def select_target(
    rows: list[dict[str, Any]],
    target: float,
    *,
    speedup_field: str = "inference_speedup_ratio_of_sums",
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["threshold"]))
    nearest = min(
        ordered,
        key=lambda row: abs(float(row[speedup_field]) - target),
    )
    brackets: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for left, right in zip(ordered, ordered[1:]):
        left_speed = float(left[speedup_field])
        right_speed = float(right[speedup_field])
        if min(left_speed, right_speed) <= target <= max(left_speed, right_speed):
            brackets.append((abs(right_speed - left_speed), left, right))
    interpolation = None
    if brackets:
        _, left, right = min(brackets, key=lambda value: value[0])
        left_speed = float(left[speedup_field])
        right_speed = float(right[speedup_field])
        if math.isclose(left_speed, right_speed):
            estimate = (float(left["threshold"]) + float(right["threshold"])) / 2
        else:
            estimate = float(left["threshold"]) + (
                (target - left_speed)
                * (float(right["threshold"]) - float(left["threshold"]))
                / (right_speed - left_speed)
            )
        interpolation = {
            "threshold": estimate,
            "left_threshold": left["threshold"],
            "left_speedup": left_speed,
            "right_threshold": right["threshold"],
            "right_speedup": right_speed,
        }
    speeds = [float(row[speedup_field]) for row in ordered]
    return {
        "target_speedup": target,
        "target_bracketed_by_observed_grid": interpolation is not None,
        "observed_speedup_range": [min(speeds), max(speeds)],
        "nearest_observed": {
            "threshold": nearest["threshold"],
            "speedup": nearest[speedup_field],
            "absolute_error": abs(float(nearest[speedup_field]) - target),
            "relative_error": abs(float(nearest[speedup_field]) - target) / target,
        },
        "linear_interpolation": interpolation,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.result_root.expanduser().resolve(strict=True)
    protocol = read_json(root / "protocol.json")
    sample_ids = [str(value) for value in protocol.get("sample_ids", [])]
    thresholds = [float(value) for value in protocol.get("thresholds", [])]
    if len(sample_ids) != 10 or not thresholds:
        raise ValueError("protocol must contain 10 prompts and a non-empty threshold grid")
    profile, full_flops, always_on_flops, block_count = load_flops_profile(root)

    baseline_timings = load_condition_timings(
        root / "baseline",
        sample_ids,
        expected_implementation="wan21",
        full_flops=full_flops,
        always_on_flops=always_on_flops,
        block_count=block_count,
    )
    if any(row["reuse_calls"] != 0 for row in baseline_timings.values()):
        raise ValueError("baseline unexpectedly reused transformer residuals")
    baseline_summary = summarize_timing_condition(baseline_timings)
    baseline_total = float(baseline_summary["pipeline_generate_wall_seconds"]["total"])
    baseline_dit_cuda_total = float(baseline_summary["dit_cuda_seconds"]["total"])
    baseline_tflops_total = float(baseline_summary["estimated_dit_total_tflops"])

    condition_rows: list[dict[str, Any]] = []
    per_prompt_rows: list[dict[str, Any]] = []
    condition_summaries: list[dict[str, Any]] = []
    for threshold in thresholds:
        label = threshold_label(threshold)
        condition_dir = root / label
        condition = read_json(condition_dir / "condition.json")
        if not math.isclose(float(condition.get("threshold", -1)), threshold):
            raise ValueError(f"condition threshold mismatch: {condition_dir}")
        timings = load_condition_timings(
            condition_dir,
            sample_ids,
            expected_implementation="teacache",
            full_flops=full_flops,
            always_on_flops=always_on_flops,
            block_count=block_count,
        )
        metric_summary, per_video_metrics = load_quality(condition_dir, sample_ids)
        timing_summary = summarize_timing_condition(timings)
        quality = quality_means(metric_summary)
        candidate_total = float(timing_summary["pipeline_generate_wall_seconds"]["total"])
        normalized_candidate_total = sum(
            float(timings[sample_id]["stages"]["denoising_core"])
            + float(baseline_timings[sample_id]["stages"]["text_encoding"])
            + float(baseline_timings[sample_id]["stages"]["vae_decode"])
            + float(baseline_timings[sample_id]["stages"]["pipeline_other"])
            for sample_id in sample_ids
        )
        candidate_dit_cuda_total = float(timing_summary["dit_cuda_seconds"]["total"])
        candidate_tflops_total = float(timing_summary["estimated_dit_total_tflops"])
        prompt_speedups = [
            float(baseline_timings[sample_id]["pipeline_seconds"])
            / float(timings[sample_id]["pipeline_seconds"])
            for sample_id in sample_ids
        ]
        row = {
            "threshold": threshold,
            "prompt_count": len(sample_ids),
            "inference_speedup_ratio_of_sums": baseline_total / candidate_total,
            "fixed_module_normalized_speedup_ratio_of_sums": (
                baseline_total / normalized_candidate_total
            ),
            "mean_per_prompt_speedup": statistics.fmean(prompt_speedups),
            "min_per_prompt_speedup": min(prompt_speedups),
            "max_per_prompt_speedup": max(prompt_speedups),
            "pipeline_seconds_total": candidate_total,
            "fixed_module_normalized_pipeline_seconds_total": normalized_candidate_total,
            "pipeline_seconds_mean": timing_summary["pipeline_generate_wall_seconds"]["mean"],
            "text_encoding_seconds_mean": timing_summary["module_wall_seconds"]["text_encoding"]["mean"],
            "denoising_core_seconds_mean": timing_summary["module_wall_seconds"]["denoising_core"]["mean"],
            "vae_decode_seconds_mean": timing_summary["module_wall_seconds"]["vae_decode"]["mean"],
            "pipeline_other_seconds_mean": timing_summary["module_wall_seconds"]["pipeline_other"]["mean"],
            "dit_cuda_seconds_total": candidate_dit_cuda_total,
            "dit_cuda_speedup_ratio_of_sums": baseline_dit_cuda_total / candidate_dit_cuda_total,
            "full_compute_forward_calls": timing_summary["full_compute_forward_calls"],
            "reuse_forward_calls": timing_summary["reuse_forward_calls"],
            "reuse_fraction": timing_summary["reuse_fraction"],
            "estimated_dit_tflops_per_video_mean": timing_summary["estimated_dit_tflops_per_video"]["mean"],
            "estimated_achieved_dit_tflops_per_second": timing_summary[
                "estimated_achieved_dit_tflops_per_second_ratio_of_sums"
            ],
            "dit_tflops_reduction_fraction": 1.0 - candidate_tflops_total / baseline_tflops_total,
            "psnr_rgb_db": quality["psnr_rgb_db"],
            "ssim_rgb": quality["ssim_rgb"],
            "lpips_alex_v0_1_spatial": quality["lpips_alex_v0_1_spatial"],
        }
        condition_rows.append(row)
        condition_summaries.append(
            {
                "threshold": threshold,
                "label": label,
                "timing": timing_summary,
                "quality": quality,
                "comparison": {
                    "inference_speedup_ratio_of_sums": row["inference_speedup_ratio_of_sums"],
                    "fixed_module_normalized_speedup_ratio_of_sums": row[
                        "fixed_module_normalized_speedup_ratio_of_sums"
                    ],
                    "dit_cuda_speedup_ratio_of_sums": row["dit_cuda_speedup_ratio_of_sums"],
                    "dit_tflops_reduction_fraction": row["dit_tflops_reduction_fraction"],
                },
            }
        )
        for sample_id, prompt_speedup in zip(sample_ids, prompt_speedups):
            timing = timings[sample_id]
            metric = per_video_metrics[sample_id]
            normalized_prompt_seconds = (
                float(timing["stages"]["denoising_core"])
                + float(baseline_timings[sample_id]["stages"]["text_encoding"])
                + float(baseline_timings[sample_id]["stages"]["vae_decode"])
                + float(baseline_timings[sample_id]["stages"]["pipeline_other"])
            )
            per_prompt_rows.append(
                {
                    "threshold": threshold,
                    "sample_id": sample_id,
                    "baseline_pipeline_seconds": baseline_timings[sample_id]["pipeline_seconds"],
                    "candidate_pipeline_seconds": timing["pipeline_seconds"],
                    "inference_speedup": prompt_speedup,
                    "fixed_module_normalized_candidate_seconds": normalized_prompt_seconds,
                    "fixed_module_normalized_speedup": (
                        float(baseline_timings[sample_id]["pipeline_seconds"])
                        / normalized_prompt_seconds
                    ),
                    "text_encoding_seconds": timing["stages"]["text_encoding"],
                    "denoising_core_seconds": timing["stages"]["denoising_core"],
                    "vae_decode_seconds": timing["stages"]["vae_decode"],
                    "pipeline_other_seconds": timing["stages"]["pipeline_other"],
                    "dit_cuda_seconds": timing["dit_cuda_seconds"],
                    "full_compute_forward_calls": timing["full_calls"],
                    "reuse_forward_calls": timing["reuse_calls"],
                    "estimated_dit_tflops": timing["estimated_dit_tflops"],
                    "estimated_achieved_dit_tflops_per_second": timing[
                        "estimated_achieved_dit_tflops_per_second"
                    ],
                    "psnr_rgb_db": metric["psnr_rgb_db_mean"],
                    "ssim_rgb": metric["ssim_rgb_mean"],
                    "lpips_alex_v0_1_spatial": metric[
                        "lpips_alex_v0_1_spatial_mean"
                    ],
                }
            )

    monotonic_violations = [
        {
            "left_threshold": left["threshold"],
            "left_speedup": left["inference_speedup_ratio_of_sums"],
            "right_threshold": right["threshold"],
            "right_speedup": right["inference_speedup_ratio_of_sums"],
        }
        for left, right in zip(condition_rows, condition_rows[1:])
        if float(right["inference_speedup_ratio_of_sums"])
        < float(left["inference_speedup_ratio_of_sums"])
    ]
    targets = [select_target(condition_rows, target) for target in TARGETS]
    normalized_targets = [
        select_target(
            condition_rows,
            target,
            speedup_field="fixed_module_normalized_speedup_ratio_of_sums",
        )
        for target in TARGETS
    ]
    validation = {
        "status": "passed",
        "prompt_count": len(sample_ids),
        "threshold_count": len(thresholds),
        "condition_count_including_baseline": len(thresholds) + 1,
        "timing_trace_count": (len(thresholds) + 1) * len(sample_ids),
        "video_count": (len(thresholds) + 1) * len(sample_ids),
        "quality_pair_count": len(thresholds) * len(sample_ids),
        "quality_frame_count": len(thresholds) * len(sample_ids) * 81,
        "checks": [
            "exact prompt/video/timing sets",
            "100 DiT calls per video",
            "full+reuse call reconciliation",
            "module timings sum to pipeline latency",
            "two T5 calls and one VAE decode per video",
            "81 aligned frames per quality pair",
            "PSNR/SSIM/LPIPS all present",
            "actual block traces mapped to Calflops operation counts",
        ],
    }
    payload = {
        "schema_version": 1,
        "status": "complete",
        "protocol": protocol,
        "validation": validation,
        "calflops_profile": {
            "path": str(root / "profiling" / "calflops.json"),
            "scope": profile.get("scope"),
            "estimated_full_tflops_per_forward": full_flops / TFLOP_DIVISOR,
            "estimated_always_on_tflops_per_forward": always_on_flops / TFLOP_DIVISOR,
        },
        "baseline": baseline_summary,
        "conditions": condition_summaries,
        "targets": targets,
        "fixed_module_normalized_targets": normalized_targets,
        "monotonic_violations": monotonic_violations,
        "caveats": [
            "The 10 prompts are a fixed calibration subset, not all 200 VBench prompts.",
            "Each prompt/threshold has one seed and one timed run; prompt coverage is broader than repetition coverage.",
            "TFLOPs and TFLOP/s cover the DiT path only; T5 and VAE have measured latency but no FLOP count.",
            "Interpolated thresholds are descriptive because TeaCache cache decisions change in discrete steps.",
            "Fixed-module-normalized speedup replaces each candidate's T5/VAE/other time with the matched baseline time and is a sensitivity analysis, not the headline metric.",
        ],
    }

    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "summary.csv", condition_rows)
    write_csv(analysis_dir / "per_prompt.csv", per_prompt_rows)
    (analysis_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (analysis_dir / "data_quality.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = [
        "# TeaCache Wan2.1 VBench10 threshold scan",
        "",
        "Speedup uses the ratio of summed end-to-end inference time. Model loading, MP4 export, and metric evaluation are excluded.",
        "",
        "| threshold | raw speedup | normalized speedup | T5 s | denoise s | VAE s | other s | reuse | DiT TFLOPs/video | TFLOP/s | PSNR dB | SSIM | LPIPS |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in condition_rows:
        report.append(
            f"| {row['threshold']:.3f} | {row['inference_speedup_ratio_of_sums']:.4f}× | "
            f"{row['fixed_module_normalized_speedup_ratio_of_sums']:.4f}× | "
            f"{row['text_encoding_seconds_mean']:.2f} | {row['denoising_core_seconds_mean']:.2f} | "
            f"{row['vae_decode_seconds_mean']:.2f} | {row['pipeline_other_seconds_mean']:.2f} | "
            f"{row['reuse_fraction']:.3f} | {row['estimated_dit_tflops_per_video_mean']:.2f} | "
            f"{row['estimated_achieved_dit_tflops_per_second']:.2f} | {row['psnr_rgb_db']:.3f} | "
            f"{row['ssim_rgb']:.5f} | {row['lpips_alex_v0_1_spatial']:.5f} |"
        )
    report.extend(("", "## Target thresholds", ""))
    threshold_grid_label = (
        f"{min(protocol['thresholds']):.3g}–{max(protocol['thresholds']):.3g}"
    )
    for target in targets:
        nearest = target["nearest_observed"]
        interpolation = target["linear_interpolation"]
        if interpolation is None:
            detail = f"not bracketed by the observed {threshold_grid_label} grid"
        else:
            detail = f"linear interpolation {interpolation['threshold']:.5f}"
        report.append(
            f"- {target['target_speedup']:.1f}×: nearest observed threshold "
            f"{nearest['threshold']:.3f} gives {nearest['speedup']:.4f}× "
            f"(absolute error {nearest['absolute_error']:.4f}); {detail}."
        )
    report.extend(("", "## Fixed-module-normalized sensitivity", ""))
    for target in normalized_targets:
        nearest = target["nearest_observed"]
        interpolation = target["linear_interpolation"]
        if interpolation is None:
            detail = f"not bracketed by the observed {threshold_grid_label} grid"
        else:
            detail = f"linear interpolation {interpolation['threshold']:.5f}"
        report.append(
            f"- {target['target_speedup']:.1f}×: nearest observed threshold "
            f"{nearest['threshold']:.3f} gives normalized {nearest['speedup']:.4f}×; "
            f"{detail}."
        )
    report.extend(
        (
            "",
            "## Validation",
            "",
            f"All completeness checks passed for {validation['timing_trace_count']} timing traces, "
            f"{validation['video_count']} videos, {validation['quality_pair_count']} quality pairs, "
            f"and {validation['quality_frame_count']} paired frames.",
            "",
            "TFLOPs/TFLOP/s are DiT-only estimates. T5 and VAE are represented by measured stage latency.",
        )
    )
    if monotonic_violations:
        report.extend(("", f"Timing noise produced {len(monotonic_violations)} monotonic violation(s)."))
    (analysis_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "analysis": str(analysis_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
