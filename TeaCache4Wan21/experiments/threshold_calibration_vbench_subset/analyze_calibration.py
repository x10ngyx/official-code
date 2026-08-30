#!/usr/bin/env python3
"""Validate and aggregate TeaCache threshold calibration timings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


TARGETS = (1.8, 2.4, 3.0)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent
EXP_ROOT = Path("/all/yiran07-disk3/huteng_data/exp").resolve()
PERFORMANCE_SCRIPT = PROJECT_DIR / "experiments/vbench200_t2v/aggregate_performance.py"
VIDEO_METRICS_SCRIPT = REPOSITORY_DIR / "VideoMetrics/evaluate.py"
VBENCH_SCRIPT = REPOSITORY_DIR / "VbenchEvaluation/run_vbench200.sh"
sys.path.insert(0, str(REPOSITORY_DIR / "ComponentMetrics"))
from reporting import extract_component_latency  # noqa: E402

THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--targets", type=float, nargs="+", default=TARGETS)
    parser.add_argument("--calflops-profile", type=Path, required=True)
    parser.add_argument("--metric-device", default="cuda:0")
    parser.add_argument("--lpips-batch-size", type=int, default=4)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--vbench-python", type=Path, required=True)
    parser.add_argument("--vbench-cache", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_timings(condition_dir: Path, expected_ids: list[str]) -> dict[str, dict[str, Any]]:
    timings: dict[str, dict[str, Any]] = {}
    for sample_id in expected_ids:
        path = condition_dir / "timings" / f"{sample_id}.json"
        payload = load_json(path)
        if payload.get("status") != "success":
            raise ValueError(f"unsuccessful timing: {path}")
        if payload.get("model_forward_call_count") != 100:
            raise ValueError(f"expected 100 DiT calls: {path}")
        full = int(payload.get("full_compute_forward_calls", -1))
        reuse = int(payload.get("reuse_forward_calls", -1))
        if full + reuse != 100:
            raise ValueError(f"full/reuse calls do not sum to 100: {path}")
        latency = payload.get("pipeline_generate_wall_seconds")
        cuda_latency = payload.get("model_forward_cuda_seconds")
        if not isinstance(latency, (int, float)) or latency <= 0:
            raise ValueError(f"invalid pipeline latency: {path}")
        if not isinstance(cuda_latency, (int, float)) or cuda_latency <= 0:
            raise ValueError(f"invalid DiT CUDA latency: {path}")
        extract_component_latency(payload)
        timings[sample_id] = payload
    return timings


def require_external(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as error:
        raise ValueError(f"artifact must be below {EXP_ROOT}: {resolved}") from error
    return resolved


def run_logged(command: list[str], log: Path, *, env: dict[str, str] | None = None) -> None:
    if log.exists():
        raise FileExistsError(f"incomplete prior command log exists: {log}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as handle:
        handle.write("command=" + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )


def ensure_evaluations(
    *,
    root: Path,
    condition_dir: Path,
    label: str,
    expected_videos: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    destination = root / "analysis" / label
    performance_path = destination / "performance" / "summary.json"
    if not performance_path.is_file():
        run_logged(
            [
                sys.executable,
                str(PERFORMANCE_SCRIPT),
                "--baseline-dir",
                str(root / "baseline"),
                "--teacache-dir",
                str(condition_dir),
                "--calflops-profile",
                str(args.calflops_profile),
                "--output-dir",
                str(destination / "performance"),
                "--expected-videos",
                str(expected_videos),
            ],
            destination / "performance.log",
            env={**os.environ, **THREAD_ENV},
        )
    metrics_path = destination / "video_metrics" / "summary.json"
    if not metrics_path.is_file():
        run_logged(
            [
                sys.executable,
                str(VIDEO_METRICS_SCRIPT),
                "--reference-dir",
                str(root / "baseline" / "videos"),
                "--candidate-dir",
                str(condition_dir / "videos"),
                "--extension",
                ".mp4",
                "--expected-frames",
                "81",
                "--device",
                args.metric_device,
                "--lpips-batch-size",
                str(args.lpips_batch_size),
                "--model-cache",
                str(args.model_cache),
                "--output-dir",
                str(destination / "video_metrics"),
            ],
            destination / "video_metrics.log",
            env={**os.environ, **THREAD_ENV},
        )
    vbench_env = {**os.environ, **THREAD_ENV}
    vbench_env["PYTHON_BIN"] = str(args.vbench_python)
    vbench_env["VBENCH_CACHE_DIR"] = str(args.vbench_cache)
    baseline_vbench_path = (
        root / "analysis" / "vbench_baseline" / "vbench200_aggregate_scores.json"
    )
    if not baseline_vbench_path.is_file():
        run_logged(
            [
                "bash",
                str(VBENCH_SCRIPT),
                str(root / "baseline" / "videos"),
                str(root / "analysis" / "vbench_baseline"),
                "1",
                "--allow-missing",
            ],
            root / "analysis" / "vbench_baseline.log",
            env=vbench_env,
        )
    candidate_vbench_path = destination / "vbench" / "vbench200_aggregate_scores.json"
    if not candidate_vbench_path.is_file():
        run_logged(
            [
                "bash",
                str(VBENCH_SCRIPT),
                str(condition_dir / "videos"),
                str(destination / "vbench"),
                "1",
                "--allow-missing",
            ],
            destination / "vbench.log",
            env=vbench_env,
        )
    return tuple(
        load_json(path)
        for path in (
            performance_path,
            metrics_path,
            baseline_vbench_path,
            candidate_vbench_path,
        )
    )


def interpolate(rows: list[dict[str, Any]], target: float) -> dict[str, Any] | None:
    ordered = sorted(rows, key=lambda row: float(row["threshold"]))
    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for left, right in zip(ordered, ordered[1:]):
        left_speed = float(left["inference_speedup_ratio_of_sums"])
        right_speed = float(right["inference_speedup_ratio_of_sums"])
        if min(left_speed, right_speed) <= target <= max(left_speed, right_speed):
            candidates.append((abs(right_speed - left_speed), left, right))
    if not candidates:
        return None
    _, left, right = min(candidates, key=lambda item: item[0])
    left_speed = float(left["inference_speedup_ratio_of_sums"])
    right_speed = float(right["inference_speedup_ratio_of_sums"])
    if right_speed == left_speed:
        estimate = (float(left["threshold"]) + float(right["threshold"])) / 2
    else:
        estimate = float(left["threshold"]) + (
            (target - left_speed)
            * (float(right["threshold"]) - float(left["threshold"]))
            / (right_speed - left_speed)
        )
    return {
        "threshold": estimate,
        "left_threshold": left["threshold"],
        "left_speedup": left_speed,
        "right_threshold": right["threshold"],
        "right_speedup": right_speed,
    }


def main() -> None:
    args = parse_args()
    root = require_external(args.result_root)
    args.calflops_profile = require_external(args.calflops_profile)
    args.model_cache = args.model_cache.expanduser().resolve(strict=True)
    args.vbench_python = args.vbench_python.expanduser().resolve(strict=True)
    args.vbench_cache = args.vbench_cache.expanduser().resolve(strict=True)
    protocol = load_json(root / "calibration_protocol.json")
    sample_ids = protocol.get("sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ValueError("calibration protocol has no sample ids")
    sample_ids = [str(value) for value in sample_ids]
    baseline = load_timings(root / "baseline", sample_ids)
    baseline_pipeline = sum(
        float(baseline[sample_id]["pipeline_generate_wall_seconds"])
        for sample_id in sample_ids
    )
    baseline_cuda = sum(
        float(baseline[sample_id]["model_forward_cuda_seconds"])
        for sample_id in sample_ids
    )

    rows: list[dict[str, Any]] = []
    per_prompt: list[dict[str, Any]] = []
    for condition_path in sorted(root.glob("threshold_*/condition.json")):
        condition = load_json(condition_path)
        threshold = condition.get("threshold")
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError(f"invalid threshold condition: {condition_path}")
        timings = load_timings(condition_path.parent, sample_ids)
        candidate_pipeline = sum(
            float(timings[sample_id]["pipeline_generate_wall_seconds"])
            for sample_id in sample_ids
        )
        candidate_cuda = sum(
            float(timings[sample_id]["model_forward_cuda_seconds"])
            for sample_id in sample_ids
        )
        prompt_speedups = [
            float(baseline[sample_id]["pipeline_generate_wall_seconds"])
            / float(timings[sample_id]["pipeline_generate_wall_seconds"])
            for sample_id in sample_ids
        ]
        full_calls = sum(
            int(timings[sample_id]["full_compute_forward_calls"])
            for sample_id in sample_ids
        )
        reuse_calls = sum(
            int(timings[sample_id]["reuse_forward_calls"])
            for sample_id in sample_ids
        )
        row = {
            "condition": condition["label"],
            "threshold": float(threshold),
            "prompt_count": len(sample_ids),
            "baseline_pipeline_seconds": baseline_pipeline,
            "candidate_pipeline_seconds": candidate_pipeline,
            "inference_speedup_ratio_of_sums": baseline_pipeline / candidate_pipeline,
            "mean_per_prompt_speedup": statistics.fmean(prompt_speedups),
            "median_per_prompt_speedup": statistics.median(prompt_speedups),
            "min_per_prompt_speedup": min(prompt_speedups),
            "max_per_prompt_speedup": max(prompt_speedups),
            "baseline_dit_cuda_seconds": baseline_cuda,
            "candidate_dit_cuda_seconds": candidate_cuda,
            "dit_cuda_speedup_ratio_of_sums": baseline_cuda / candidate_cuda,
            "full_compute_forward_calls": full_calls,
            "reuse_forward_calls": reuse_calls,
            "reuse_fraction": reuse_calls / (full_calls + reuse_calls),
        }
        performance, metrics, baseline_vbench, candidate_vbench = ensure_evaluations(
            root=root,
            condition_dir=condition_path.parent,
            label=str(condition["label"]),
            expected_videos=len(sample_ids),
            args=args,
        )
        baseline_performance = performance["conditions"]["baseline"]
        candidate_performance = performance["conditions"]["teacache"]
        metric_values = metrics["metrics"]
        row.update(
            {
                "baseline_t5_cuda_seconds": baseline_performance["t5_cuda_seconds"]["total"],
                "candidate_t5_cuda_seconds": candidate_performance["t5_cuda_seconds"]["total"],
                "baseline_vae_decode_cuda_seconds": baseline_performance["vae_decode_cuda_seconds"]["total"],
                "candidate_vae_decode_cuda_seconds": candidate_performance["vae_decode_cuda_seconds"]["total"],
                "baseline_estimated_dit_total_tflops": baseline_performance["estimated_dit_total_tflops"],
                "candidate_estimated_dit_total_tflops": candidate_performance["estimated_dit_total_tflops"],
                "estimated_t5_tflops_per_video": candidate_performance["estimated_t5_tflops_per_video"],
                "estimated_vae_decode_tflops_per_video": candidate_performance["estimated_vae_decode_tflops_per_video"],
                "psnr_rgb_db": metric_values["psnr_rgb_db"]["mean"],
                "ssim_rgb": metric_values["ssim_rgb"]["mean"],
                "lpips_alex_v0_1_spatial": metric_values["lpips_alex_v0_1_spatial"]["mean"],
                "baseline_vbench_score": baseline_vbench["aggregate_scores"]["total_score"],
                "candidate_vbench_score": candidate_vbench["aggregate_scores"]["total_score"],
            }
        )
        rows.append(row)
        for sample_id, speedup in zip(sample_ids, prompt_speedups):
            per_prompt.append(
                {
                    "condition": condition["label"],
                    "threshold": float(threshold),
                    "sample_id": sample_id,
                    "baseline_pipeline_seconds": baseline[sample_id][
                        "pipeline_generate_wall_seconds"
                    ],
                    "candidate_pipeline_seconds": timings[sample_id][
                        "pipeline_generate_wall_seconds"
                    ],
                    "inference_speedup": speedup,
                    "full_compute_forward_calls": timings[sample_id][
                        "full_compute_forward_calls"
                    ],
                    "reuse_forward_calls": timings[sample_id]["reuse_forward_calls"],
                }
            )
    if not rows:
        raise ValueError("no threshold conditions found")

    rows.sort(key=lambda row: float(row["threshold"]))
    monotonic_violations = []
    for left, right in zip(rows, rows[1:]):
        if float(right["inference_speedup_ratio_of_sums"]) < float(
            left["inference_speedup_ratio_of_sums"]
        ):
            monotonic_violations.append(
                {
                    "left_threshold": left["threshold"],
                    "left_speedup": left["inference_speedup_ratio_of_sums"],
                    "right_threshold": right["threshold"],
                    "right_speedup": right["inference_speedup_ratio_of_sums"],
                }
            )

    targets = []
    for target in args.targets:
        nearest = min(
            rows,
            key=lambda row: abs(float(row["inference_speedup_ratio_of_sums"]) - target),
        )
        targets.append(
            {
                "target_speedup": target,
                "nearest_observed": {
                    "threshold": nearest["threshold"],
                    "speedup": nearest["inference_speedup_ratio_of_sums"],
                    "absolute_error": abs(
                        float(nearest["inference_speedup_ratio_of_sums"]) - target
                    ),
                },
                "linear_interpolation": interpolate(rows, target),
            }
        )

    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (analysis_dir / "per_prompt.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_prompt[0]))
        writer.writeheader()
        writer.writerows(per_prompt)
    summary = {
        "schema_version": 2,
        "status": "complete",
        "latency_metric": protocol["latency_metric"],
        "speedup_aggregation": protocol["speedup_aggregation"],
        "quality_metrics": [
            "psnr_rgb_db",
            "ssim_rgb",
            "lpips_alex_v0_1_spatial",
            "vbench_score",
        ],
        "flops_headline": "estimated DiT TFLOPs; T5 and VAE decode recorded separately",
        "sample_ids": sample_ids,
        "baseline": {
            "pipeline_seconds": baseline_pipeline,
            "dit_cuda_seconds": baseline_cuda,
        },
        "conditions": rows,
        "targets": targets,
        "monotonic_violations": monotonic_violations,
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report_lines = [
        "# TeaCache threshold calibration",
        "",
        "Inference speedup is the ratio of summed `pipeline_generate_wall_seconds`; "
        "model loading and MP4 export are excluded.",
        "",
        "| threshold | inference speedup | DiT TFLOPs | PSNR | SSIM | LPIPS | VBench |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['threshold']:.4f} | "
            f"{row['inference_speedup_ratio_of_sums']:.4f}× | "
            f"{row['candidate_estimated_dit_total_tflops']:.4f} | "
            f"{row['psnr_rgb_db']:.4f} | {row['ssim_rgb']:.4f} | "
            f"{row['lpips_alex_v0_1_spatial']:.4f} | "
            f"{row['candidate_vbench_score']:.4f} |"
        )
    report_lines.extend(("", "## Targets", ""))
    for target in targets:
        nearest = target["nearest_observed"]
        interpolation = target["linear_interpolation"]
        interpolation_text = (
            f"interpolated threshold {interpolation['threshold']:.5f}"
            if interpolation is not None
            else "target not bracketed"
        )
        report_lines.append(
            f"- {target['target_speedup']:.1f}×: nearest observed threshold "
            f"{nearest['threshold']:.4f} → {nearest['speedup']:.4f}×; "
            f"{interpolation_text}."
        )
    if monotonic_violations:
        report_lines.extend(
            ("", f"Warning: {len(monotonic_violations)} monotonic violations observed.")
        )
    (analysis_dir / "REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
