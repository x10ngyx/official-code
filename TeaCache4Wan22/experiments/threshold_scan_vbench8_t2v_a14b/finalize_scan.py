#!/usr/bin/env python3
"""Validate, evaluate, and aggregate the complete Vbench8 threshold scan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "scan_config.json"
PROMPTS_PATH = SCRIPT_DIR / "prompts.jsonl"

sys.path.insert(0, str(SCRIPT_DIR))
from plan_scan import load_prompts, validate_inputs  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def threshold_label(threshold: float) -> str:
    return f"threshold_{threshold:.3f}".replace(".", "p")


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_logged(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    if log_path.exists():
        raise FileExistsError(f"command log exists but output is incomplete: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as handle:
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


def ensure_link(link: Path, target: Path) -> None:
    target = target.resolve(strict=True)
    if link.is_symlink():
        if link.resolve(strict=True) != target:
            raise ValueError(f"staging link points to the wrong file: {link}")
        return
    if link.exists():
        raise FileExistsError(f"staging path is not a symlink: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)


def metric_video_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["video_id"]: row for row in csv.DictReader(handle)}


def performance_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def timing_payload(manifest_path: Path) -> dict[str, Any]:
    return read_json(manifest_path)["timing"]["payload"]


def nearest_targets(
    rows: list[dict[str, Any]], targets: list[float]
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["threshold"]))
    results = []
    for target in targets:
        nearest = min(
            ordered,
            key=lambda row: (
                round(
                    abs(float(row["latency_speedup_ratio_of_sums"]) - target), 12
                ),
                float(row["threshold"]),
            ),
        )
        brackets = []
        for left, right in zip(ordered, ordered[1:]):
            left_speed = float(left["latency_speedup_ratio_of_sums"])
            right_speed = float(right["latency_speedup_ratio_of_sums"])
            if left_speed == right_speed:
                continue
            if (left_speed - target) * (right_speed - target) <= 0:
                fraction = (target - left_speed) / (right_speed - left_speed)
                interpolated = float(left["threshold"]) + fraction * (
                    float(right["threshold"]) - float(left["threshold"])
                )
                brackets.append(
                    {
                        "lower_threshold": float(left["threshold"]),
                        "lower_speedup": left_speed,
                        "upper_threshold": float(right["threshold"]),
                        "upper_speedup": right_speed,
                        "linearly_interpolated_threshold": interpolated,
                    }
                )
        bracket = (
            min(
                brackets,
                key=lambda item: abs(item["upper_threshold"] - item["lower_threshold"]),
            )
            if brackets
            else None
        )
        observed_speed = float(nearest["latency_speedup_ratio_of_sums"])
        results.append(
            {
                "target_speedup": target,
                "nearest_measured_threshold": float(nearest["threshold"]),
                "nearest_measured_speedup": observed_speed,
                "absolute_speedup_error": abs(observed_speed - target),
                "relative_speedup_error": abs(observed_speed - target) / target,
                "quality_at_nearest_threshold": {
                    "psnr_rgb_db": float(nearest["psnr_rgb_db"]),
                    "ssim_rgb": float(nearest["ssim_rgb"]),
                    "lpips_alex_v0_1_spatial": float(
                        nearest["lpips_alex_v0_1_spatial"]
                    ),
                    "vbench_score": float(nearest["candidate_vbench_score"]),
                },
                "interpolation_diagnostic": bracket,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--calflops-profile", type=Path, required=True)
    parser.add_argument("--metric-device", default="cuda:0")
    parser.add_argument("--lpips-batch-size", type=int, default=2)
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=REPOSITORY_DIR.parents[1] / "models" / "torch-cache",
    )
    parser.add_argument("--vbench-python", required=True)
    parser.add_argument(
        "--vbench-cache",
        type=Path,
        default=REPOSITORY_DIR.parents[1] / "models" / "VBench",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve(strict=True)
    calflops_profile = args.calflops_profile.expanduser().resolve(strict=True)
    config, prompts = validate_inputs()
    if sha256(result_root / "scan_config.json") != sha256(CONFIG_PATH):
        raise ValueError("result scan_config.json differs from the canonical snapshot")
    if sha256(result_root / "prompts.jsonl") != sha256(PROMPTS_PATH):
        raise ValueError("result prompts.jsonl differs from the canonical snapshot")
    id_by_prompt = {row["prompt_en"]: row["sample_id"] for row in prompts}
    if len(id_by_prompt) != len(prompts):
        raise ValueError("selected prompt texts are not unique")

    aggregate_script = (
        PROJECT_DIR / "experiments" / "performance_t2v_a14b" / "aggregate_performance.py"
    )
    metrics_script = REPOSITORY_DIR / "VideoMetrics" / "evaluate.py"
    vbench_script = REPOSITORY_DIR / "VbenchEvaluation" / "run_vbench200.sh"
    analysis_root = result_root / "analysis"
    staging_root = analysis_root / "metric_staging"
    analysis_root.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    baseline_fingerprints: dict[str, tuple[float, float, str]] = {}

    for threshold in config["thresholds"]:
        label = threshold_label(float(threshold))
        threshold_root = analysis_root / label
        performance_root = threshold_root / "performance"
        metrics_root = threshold_root / "video_metrics"
        reference_vbench_root = analysis_root / "vbench_reference"
        candidate_vbench_root = threshold_root / "vbench_candidate"
        baseline_manifests = [
            result_root / "runs" / row["sample_id"] / "baseline.manifest.json"
            for row in prompts
        ]
        candidate_manifests = [
            result_root
            / "runs"
            / row["sample_id"]
            / f"{label}.manifest.json"
            for row in prompts
        ]
        for path in baseline_manifests + candidate_manifests:
            if not path.is_file():
                raise FileNotFoundError(path)

        performance_summary_path = performance_root / "summary.json"
        if not performance_summary_path.is_file():
            command = [sys.executable, str(aggregate_script)]
            for baseline, candidate in zip(baseline_manifests, candidate_manifests):
                command.extend(["--baseline-manifest", str(baseline)])
                command.extend(["--teacache-manifest", str(candidate)])
            command.extend(
                [
                    "--calflops-profile",
                    str(calflops_profile),
                    "--output-dir",
                    str(performance_root),
                ]
            )
            run_logged(command, threshold_root / "performance.log")

        reference_staging = staging_root / label / "reference"
        candidate_staging = staging_root / label / "candidate"
        for row in prompts:
            sample_id = row["sample_id"]
            ensure_link(
                reference_staging / f"{sample_id}.mp4",
                result_root / "runs" / sample_id / "baseline.mp4",
            )
            ensure_link(
                candidate_staging / f"{sample_id}.mp4",
                result_root / "runs" / sample_id / f"{label}.mp4",
            )

        metrics_summary_path = metrics_root / "summary.json"
        if not metrics_summary_path.is_file():
            environment = os.environ.copy()
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                environment[name] = "1"
            command = [
                sys.executable,
                str(metrics_script),
                "--reference-dir",
                str(reference_staging),
                "--candidate-dir",
                str(candidate_staging),
                "--extension",
                ".mp4",
                "--expected-frames",
                "45",
                "--device",
                args.metric_device,
                "--lpips-batch-size",
                str(args.lpips_batch_size),
                "--model-cache",
                str(args.model_cache),
                "--output-dir",
                str(metrics_root),
            ]
            run_logged(command, threshold_root / "video_metrics.log", environment)

        vbench_environment = os.environ.copy()
        vbench_environment["PYTHON_BIN"] = args.vbench_python
        vbench_environment["VBENCH_CACHE_DIR"] = str(
            args.vbench_cache.expanduser().resolve()
        )
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            vbench_environment[name] = "1"
        reference_vbench_path = (
            reference_vbench_root / "vbench200_aggregate_scores.json"
        )
        if not reference_vbench_path.is_file():
            run_logged(
                [
                    "bash",
                    str(vbench_script),
                    str(reference_staging),
                    str(reference_vbench_root),
                    "1",
                    "--allow-missing",
                ],
                analysis_root / "vbench_reference.log",
                vbench_environment,
            )
        candidate_vbench_path = (
            candidate_vbench_root / "vbench200_aggregate_scores.json"
        )
        if not candidate_vbench_path.is_file():
            run_logged(
                [
                    "bash",
                    str(vbench_script),
                    str(candidate_staging),
                    str(candidate_vbench_root),
                    "1",
                    "--allow-missing",
                ],
                threshold_root / "vbench_candidate.log",
                vbench_environment,
            )

        performance = read_json(performance_summary_path)
        metrics = read_json(metrics_summary_path)
        reference_vbench = read_json(reference_vbench_path)
        candidate_vbench = read_json(candidate_vbench_path)
        perf_rows = performance_rows(performance_root / "per_video.jsonl")
        metric_rows = metric_video_rows(metrics_root / "per_video.csv")
        if len(perf_rows) != 2 * len(prompts) or len(metric_rows) != len(prompts):
            raise ValueError(f"per-video row count mismatch for {label}")
        perf_by_key = {
            (row["condition"], id_by_prompt[row["prompt"]]): row for row in perf_rows
        }

        baseline_summary = performance["conditions"]["baseline"]
        candidate_summary = performance["conditions"]["teacache"]
        comparison = performance["comparison"]
        metric_values = metrics["metrics"]
        aggregate_row = {
            "threshold": float(threshold),
            "prompt_count": len(prompts),
            "baseline_pipeline_generate_wall_seconds_total": baseline_summary[
                "pipeline_generate_wall_seconds"
            ]["total"],
            "candidate_pipeline_generate_wall_seconds_total": candidate_summary[
                "pipeline_generate_wall_seconds"
            ]["total"],
            "baseline_pipeline_generate_wall_seconds_mean": baseline_summary[
                "pipeline_generate_wall_seconds"
            ]["mean"],
            "candidate_pipeline_generate_wall_seconds_mean": candidate_summary[
                "pipeline_generate_wall_seconds"
            ]["mean"],
            "baseline_t5_cuda_seconds_total": baseline_summary["t5_cuda_seconds"]["total"],
            "candidate_t5_cuda_seconds_total": candidate_summary["t5_cuda_seconds"]["total"],
            "baseline_dit_forward_cuda_seconds_total": baseline_summary["dit_forward_cuda_seconds"]["total"],
            "candidate_dit_forward_cuda_seconds_total": candidate_summary["dit_forward_cuda_seconds"]["total"],
            "baseline_vae_decode_cuda_seconds_total": baseline_summary["vae_decode_cuda_seconds"]["total"],
            "candidate_vae_decode_cuda_seconds_total": candidate_summary["vae_decode_cuda_seconds"]["total"],
            "latency_speedup_ratio_of_sums": comparison[
                "latency_speedup_ratio_of_sums"
            ],
            "baseline_estimated_dit_total_tflops": baseline_summary[
                "estimated_dit_total_tflops"
            ],
            "candidate_estimated_dit_total_tflops": candidate_summary[
                "estimated_dit_total_tflops"
            ],
            "baseline_estimated_t5_tflops_per_video": baseline_summary[
                "estimated_t5_tflops_per_video"
            ],
            "candidate_estimated_t5_tflops_per_video": candidate_summary[
                "estimated_t5_tflops_per_video"
            ],
            "baseline_estimated_vae_decode_tflops_per_video": baseline_summary[
                "estimated_vae_decode_tflops_per_video"
            ],
            "candidate_estimated_vae_decode_tflops_per_video": candidate_summary[
                "estimated_vae_decode_tflops_per_video"
            ],
            "dit_flops_speedup_ratio_of_sums": comparison[
                "dit_flops_speedup_ratio_of_sums"
            ],
            "candidate_full_compute_forward_calls": candidate_summary[
                "total_full_compute_forward_calls"
            ],
            "candidate_reuse_forward_calls": candidate_summary[
                "total_reuse_forward_calls"
            ],
            "candidate_reuse_steps_per_video_mean": candidate_summary[
                "total_reuse_forward_calls"
            ]
            / (2 * len(prompts)),
            "psnr_rgb_db": metric_values["psnr_rgb_db"]["mean"],
            "ssim_rgb": metric_values["ssim_rgb"]["mean"],
            "lpips_alex_v0_1_spatial": metric_values[
                "lpips_alex_v0_1_spatial"
            ]["mean"],
            "baseline_vbench_score": reference_vbench["aggregate_scores"][
                "total_score"
            ],
            "candidate_vbench_score": candidate_vbench["aggregate_scores"][
                "total_score"
            ],
            "performance_summary": str(performance_summary_path),
            "video_metrics_summary": str(metrics_summary_path),
            "baseline_vbench_summary": str(reference_vbench_path),
            "candidate_vbench_summary": str(candidate_vbench_path),
        }
        aggregate_rows.append(aggregate_row)

        for prompt_row in prompts:
            sample_id = prompt_row["sample_id"]
            baseline_row = perf_by_key[("baseline", sample_id)]
            candidate_row = perf_by_key[("teacache", sample_id)]
            metric_row = metric_rows[sample_id]
            baseline_manifest = result_root / "runs" / sample_id / "baseline.manifest.json"
            candidate_manifest = (
                result_root / "runs" / sample_id / f"{label}.manifest.json"
            )
            baseline_timing = timing_payload(baseline_manifest)
            candidate_timing = timing_payload(candidate_manifest)
            fingerprint = (
                float(baseline_row["pipeline_generate_wall_seconds"]),
                float(baseline_row["estimated_dit_tflops"]),
                sha256(baseline_manifest),
            )
            if sample_id in baseline_fingerprints and baseline_fingerprints[sample_id] != fingerprint:
                raise ValueError(f"baseline changed across thresholds: {sample_id}")
            baseline_fingerprints[sample_id] = fingerprint
            detailed_rows.append(
                {
                    "sample_id": sample_id,
                    "prompt": prompt_row["prompt_en"],
                    "dimensions": ";".join(prompt_row["dimension"]),
                    "threshold": float(threshold),
                    "baseline_pipeline_init_wall_seconds": baseline_timing[
                        "pipeline_init_wall_seconds"
                    ],
                    "candidate_pipeline_init_wall_seconds": candidate_timing[
                        "pipeline_init_wall_seconds"
                    ],
                    "baseline_pipeline_generate_wall_seconds": baseline_row[
                        "pipeline_generate_wall_seconds"
                    ],
                    "candidate_pipeline_generate_wall_seconds": candidate_row[
                        "pipeline_generate_wall_seconds"
                    ],
                    "per_video_latency_speedup": baseline_row[
                        "pipeline_generate_wall_seconds"
                    ]
                    / candidate_row["pipeline_generate_wall_seconds"],
                    "baseline_dit_forward_cuda_seconds": baseline_row[
                        "dit_forward_cuda_seconds"
                    ],
                    "candidate_dit_forward_cuda_seconds": candidate_row[
                        "dit_forward_cuda_seconds"
                    ],
                    "baseline_t5_cuda_seconds": baseline_row["t5_cuda_seconds"],
                    "candidate_t5_cuda_seconds": candidate_row["t5_cuda_seconds"],
                    "baseline_vae_decode_cuda_seconds": baseline_row[
                        "vae_decode_cuda_seconds"
                    ],
                    "candidate_vae_decode_cuda_seconds": candidate_row[
                        "vae_decode_cuda_seconds"
                    ],
                    "baseline_estimated_dit_tflops": baseline_row[
                        "estimated_dit_tflops"
                    ],
                    "candidate_estimated_dit_tflops": candidate_row[
                        "estimated_dit_tflops"
                    ],
                    "baseline_estimated_t5_tflops_per_video": baseline_row[
                        "estimated_t5_tflops_per_video"
                    ],
                    "candidate_estimated_t5_tflops_per_video": candidate_row[
                        "estimated_t5_tflops_per_video"
                    ],
                    "baseline_estimated_vae_decode_tflops_per_video": baseline_row[
                        "estimated_vae_decode_tflops_per_video"
                    ],
                    "candidate_estimated_vae_decode_tflops_per_video": candidate_row[
                        "estimated_vae_decode_tflops_per_video"
                    ],
                    "per_video_dit_flops_speedup": baseline_row[
                        "estimated_dit_tflops"
                    ]
                    / candidate_row["estimated_dit_tflops"],
                    "candidate_full_compute_forward_calls": candidate_row[
                        "full_compute_forward_calls"
                    ],
                    "candidate_reuse_forward_calls": candidate_row[
                        "reuse_forward_calls"
                    ],
                    "candidate_reuse_steps": candidate_row[
                        "reuse_forward_calls"
                    ]
                    / 2,
                    "psnr_rgb_db": metric_row["psnr_rgb_db_mean"],
                    "ssim_rgb": metric_row["ssim_rgb_mean"],
                    "lpips_alex_v0_1_spatial": metric_row[
                        "lpips_alex_v0_1_spatial_mean"
                    ],
                    "baseline_manifest": str(baseline_manifest),
                    "candidate_manifest": str(candidate_manifest),
                }
            )

    targets = nearest_targets(aggregate_rows, config["target_speedups"])
    monotonicity_violations = []
    for left, right in zip(aggregate_rows, aggregate_rows[1:]):
        if right["latency_speedup_ratio_of_sums"] < left[
            "latency_speedup_ratio_of_sums"
        ]:
            monotonicity_violations.append(
                {
                    "lower_threshold": left["threshold"],
                    "lower_speedup": left["latency_speedup_ratio_of_sums"],
                    "higher_threshold": right["threshold"],
                    "higher_speedup": right["latency_speedup_ratio_of_sums"],
                }
            )

    summary = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["name"],
        "prompt_count": len(prompts),
        "threshold_count": len(config["thresholds"]),
        "latency_definition": config["latency"],
        "flops_definition": config["flops"],
        "quality_definition": config["quality"],
        "calflops_profile": {
            "path": str(calflops_profile),
            "sha256": sha256(calflops_profile),
        },
        "thresholds": aggregate_rows,
        "target_thresholds": targets,
        "warnings": [
            "This is an eleven-prompt Vbench200 pilot with full dimension coverage, not a full Vbench200 calibration.",
            "TFLOPs is estimated DiT operation count; TFLOP/s is a separate achieved-throughput diagnostic.",
            "PSNR/SSIM/LPIPS measure same-seed fidelity to no-cache output, not reference-free video quality.",
        ],
    }
    validation = {
        "schema_version": 2,
        "status": "pass",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_count": len(prompts),
        "threshold_count": len(config["thresholds"]),
        "measured_generation_count": len(prompts) * (1 + len(config["thresholds"])),
        "per_video_summary_rows": len(detailed_rows),
        "expected_per_video_summary_rows": len(prompts) * len(config["thresholds"]),
        "monotonicity_violations": monotonicity_violations,
        "checks": [
            "canonical config and prompt snapshot hashes",
            "all run artifact hashes and timing traces",
            "50-step/100-call high-low CFG identity",
            "TeaCache decision versus measured block execution",
            "Calflops profile versus prepared source/checkpoint",
            "strict 45-frame RGB video alignment",
            "complete per-video latency, TFLOPs, PSNR, SSIM, and LPIPS rows",
            "T5/DiT/VAE component latency and TFLOPs",
            "official 16-dimension Vbench200 subset score for baseline and every threshold",
        ],
    }
    atomic_json(result_root / "scan_summary.json", summary)
    atomic_csv(result_root / "scan_summary.csv", aggregate_rows)
    atomic_csv(result_root / "per_video_summary.csv", detailed_rows)
    atomic_json(result_root / "target_thresholds.json", targets)
    atomic_json(result_root / "VALIDATION.json", validation)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
