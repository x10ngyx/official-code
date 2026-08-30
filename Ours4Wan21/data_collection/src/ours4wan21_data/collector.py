#!/usr/bin/env python3
"""Collect one Wan2.1 full-reference or random-threshold manifest shard."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .manifest import (
    NUM_CANDIDATES,
    NUM_SELECTED_PROMPTS,
    NUM_SHARDS,
    NUM_STEPS,
    PROTOCOL,
    SCHEMA_CANDIDATE_COMPLETE,
    SCHEMA_PLAN,
    SCHEMA_RUNNABLE,
    read_jsonl,
    validate_plan,
    validate_runnable,
)
from .metrics import (
    FullReferenceMetricEvaluator,
    load_metric_artifacts,
    metric_paths,
    resolve_metrics_model_cache,
)
from .performance import compare_matched, read_json, write_performance
from .paths import require_result_path
from .runtime import Wan21DataRuntime, create_pipeline
from .source_lock import file_sha256
from .timing import PipelineProfiler


THREAD_KEYS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    parser.add_argument("--wan21-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--flops-profile", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--ffprobe-bin", default=shutil.which("ffprobe") or "ffprobe")
    parser.add_argument("--metrics-device", default="auto")
    parser.add_argument("--lpips-batch-size", type=int, default=8)
    parser.add_argument("--metrics-model-cache", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu-validate", action="store_true")
    parser.add_argument("--verify-baselines", action="store_true")
    return parser.parse_args()


def thread_environment() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in THREAD_KEYS}


def require_thread_limits() -> None:
    invalid = {key: value for key, value in thread_environment().items() if value != "1"}
    if invalid:
        raise RuntimeError(f"all BLAS thread limits must equal one: {invalid}")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def load_manifest(path: Path, mode: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("manifest is empty")
    if rows[0].get("schema") == SCHEMA_PLAN:
        validate_plan(rows)
        if mode == "candidate":
            raise ValueError("candidate collection requires a calibrated runnable manifest")
    elif rows[0].get("schema") == SCHEMA_RUNNABLE:
        validate_runnable(rows)
    else:
        raise ValueError("unknown manifest schema")
    return rows


def unique_prompt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompts: dict[int, dict[str, Any]] = {}
    for row in rows:
        rank = int(row["prompt_rank"])
        prompts.setdefault(rank, row)
    if sorted(prompts) != list(range(NUM_SELECTED_PROMPTS)):
        raise ValueError("prompt ranks must be exactly 0..2999")
    return [prompts[index] for index in range(NUM_SELECTED_PROMPTS)]


def selected_rows(rows: list[dict[str, Any]], mode: str, shard: int) -> list[dict[str, Any]]:
    if not 0 <= shard < NUM_SHARDS:
        raise ValueError("shard index must be in [0,3]")
    if mode == "baseline":
        result = [row for row in unique_prompt_rows(rows) if int(row["prompt_rank"]) % NUM_SHARDS == shard]
        if len(result) != 750:
            raise ValueError(f"baseline shard must contain 750 prompts, got {len(result)}")
        return result
    result = [row for row in rows if int(row["shard_index"]) == shard]
    if len(result) != NUM_CANDIDATES // NUM_SHARDS:
        raise ValueError(f"candidate shard must contain 2250 rows, got {len(result)}")
    return result


def baseline_paths(parent: Path, sample_id: str) -> dict[str, Path]:
    root = parent / "shared_baselines" / sample_id
    return {
        "root": root,
        "video": root / "baseline.mp4",
        "timing": root / "timing.json",
        "performance": root / "performance.json",
        "trace": root / "trace.json",
        "latents": root / "latents",
        "ffprobe": root / "ffprobe.json",
        "complete": root / "BASELINE_COMPLETE.json",
    }


def candidate_paths(parent: Path, shard: int, trajectory_id: str) -> dict[str, Path]:
    root = parent / "shards" / f"shard_{shard:02d}" / "candidates" / trajectory_id
    return {
        "root": root,
        "video": root / "candidate.mp4",
        "timing": root / "timing.json",
        "performance": root / "performance.json",
        "trace": root / "trace.json",
        "latents": root / "latents",
        "ffprobe": root / "ffprobe.json",
        **metric_paths(root),
        "complete": parent / "completed" / f"{trajectory_id}.json",
    }


def latent_bundle_complete(root: Path) -> bool:
    if not root.is_dir():
        return False
    expected = {f"step_{index:03d}_input.pt" for index in range(NUM_STEPS)}
    observed = {path.name for path in root.iterdir() if path.is_file()}
    return observed == expected


def baseline_complete(paths: dict[str, Path], row: dict[str, Any] | None = None) -> bool:
    if not all(nonempty(paths[key]) for key in ("video", "timing", "performance", "trace", "ffprobe", "complete")) or not latent_bundle_complete(paths["latents"]):
        return False
    try:
        marker = read_json(paths["complete"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if marker.get("schema") != "ours4wan21_baseline_complete_v1":
        return False
    if row is not None and (
        marker.get("sample_id") != row.get("sample_id")
        or int(marker.get("prompt_rank", -1)) != int(row.get("prompt_rank", -2))
        or marker.get("protocol") != PROTOCOL
    ):
        return False
    return True


def candidate_complete(paths: dict[str, Path]) -> bool:
    required = (
        "video", "timing", "performance", "trace", "ffprobe", "metrics",
        "metrics_per_frame", "metrics_per_video", "metrics_summary", "complete",
    )
    if not all(nonempty(paths[key]) for key in required) or not latent_bundle_complete(paths["latents"]):
        return False
    try:
        marker = read_json(paths["complete"])
        load_metric_artifacts(paths["root"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    branch_rows = marker.get("branch_rows")
    expected_branch_order = [
        (call_index, call_index // 2, ("cond", "uncond")[call_index % 2])
        for call_index in range(2 * NUM_STEPS)
    ]
    observed_branch_order = [
        (
            int(branch.get("call_index", -1)),
            int(branch.get("step_index", -1)),
            branch.get("branch"),
        )
        for branch in branch_rows
    ] if isinstance(branch_rows, list) else []
    return (
        marker.get("schema") == SCHEMA_CANDIDATE_COMPLETE
        and marker.get("trajectory_id") == paths["root"].name
        and isinstance(marker.get("trajectory_row"), dict)
        and isinstance(marker.get("step_rows"), list)
        and len(marker["step_rows"]) == NUM_STEPS
        and observed_branch_order == expected_branch_order
    )


def run_ffprobe(binary: str, video: Path, output: Path) -> dict[str, Any]:
    command = [
        binary, "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate",
        "-of", "json", str(video),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"ffprobe failed: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError("ffprobe did not return exactly one video stream")
    stream = streams[0]
    if int(stream["width"]) != 832 or int(stream["height"]) != 480 or int(stream["nb_read_frames"]) != 81:
        raise ValueError(f"video geometry mismatch: {stream}")
    numerator, denominator = str(stream.get("r_frame_rate", "0/1")).split("/", 1)
    if float(numerator) / float(denominator) != 16.0:
        raise ValueError(f"video FPS mismatch: {stream}")
    atomic_json(output, payload)
    return payload


def encode_video(cache_video: Any, video: torch.Tensor, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_video(
        tensor=video[None], save_file=str(output), fps=16, nrow=1,
        normalize=True, value_range=(-1, 1),
    )
    if not nonempty(output):
        raise RuntimeError(f"video export failed: {output}")


def validate_flops_profile(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema") != "ours4wan21_calflops_profile_v2":
        raise ValueError("unexpected Calflops profile schema")
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Calflops profile protocol mismatch")
    per_forward = payload.get("per_model_forward")
    if not isinstance(per_forward, dict) or int(per_forward.get("transformer_blocks", 0)) <= 0:
        raise ValueError("Calflops profile lacks valid forward constants")
    return payload


def all_baselines_complete(rows: list[dict[str, Any]], parent: Path) -> tuple[bool, list[str]]:
    missing = [
        str(row["sample_id"])
        for row in unique_prompt_rows(rows)
        if not baseline_complete(baseline_paths(parent, str(row["sample_id"])), row)
    ]
    return not missing, missing


def cpu_validate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = selected_rows(rows, args.mode, args.shard_index)
    result = {
        "status": "ok",
        "mode": args.mode,
        "manifest_schema": rows[0]["schema"],
        "manifest_rows": len(rows),
        "selected_rows": len(selected),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "expected_latents": len(selected) * NUM_STEPS,
        "thread_environment": thread_environment(),
        "protocol": PROTOCOL,
    }
    if args.mode == "candidate":
        ready, missing = all_baselines_complete(rows, args.parent_root)
        result["all_baselines_complete"] = ready
        result["missing_baseline_count"] = len(missing)
    return result


def collect_one(
    args: argparse.Namespace,
    row: dict[str, Any],
    pipeline: Any,
    runtime: Wan21DataRuntime,
    init_seconds: float,
    cache_video: Any,
    profile_path: Path,
    metric_evaluator: FullReferenceMetricEvaluator | None,
) -> None:
    sample_id = str(row["sample_id"])
    if args.mode == "baseline":
        paths = baseline_paths(args.parent_root, sample_id)
        identity = f"baseline_{sample_id}"
        runtime_row = {
            **row,
            "trajectory_id": identity,
            "policy_family": "full_compute_reference",
            "threshold_path": None,
            "mean_threshold": None,
        }
        if args.resume and baseline_complete(paths, row):
            return
    else:
        paths = candidate_paths(args.parent_root, args.shard_index, str(row["trajectory_id"]))
        identity = str(row["trajectory_id"])
        runtime_row = row
        if args.resume and candidate_complete(paths):
            return
        baseline = baseline_paths(args.parent_root, sample_id)
        if not baseline_complete(baseline, row):
            raise RuntimeError(f"candidate requires complete baseline {sample_id}")

    if paths["root"].exists():
        raise FileExistsError(
            f"incomplete existing bundle requires manual archive before retry: {paths['root']}"
        )
    paths["root"].mkdir(parents=True)
    runtime.configure_sample(runtime_row)
    profiler = PipelineProfiler(
        pipeline,
        pipeline_init_wall_seconds=init_seconds,
        output_path=paths["timing"],
        implementation=("wan21_full_compute" if args.mode == "baseline" else "ours_random_threshold"),
    )
    profiler.install()
    video = pipeline.generate(
        str(row["prompt"]),
        size=(832, 480), frame_num=81, shift=5.0, sample_solver="unipc",
        sampling_steps=50, guide_scale=5.0, seed=42, offload_model=False,
    )
    encode_video(cache_video, video, paths["video"])
    del video
    trace = runtime.capture.save_artifacts(paths["trace"], paths["latents"])
    run_ffprobe(args.ffprobe_bin, paths["video"], paths["ffprobe"])
    performance = write_performance(paths["timing"], profile_path, paths["performance"])

    if args.mode == "baseline":
        marker = {
            "schema": "ours4wan21_baseline_complete_v1",
            "sample_id": sample_id,
            "trajectory_id": identity,
            "prompt_rank": int(row["prompt_rank"]),
            "split": row["split"],
            "video": str(paths["video"].resolve()),
            "trace": str(paths["trace"].resolve()),
            "timing": str(paths["timing"].resolve()),
            "performance": str(paths["performance"].resolve()),
            "pipeline_generate_wall_seconds": performance["pipeline_generate_wall_seconds"],
            "estimated_dit_tflops_per_video": performance["estimated_dit_tflops_per_video"],
            "dit_cuda_seconds": performance["dit_cuda_seconds"],
            "t5_cuda_seconds": performance["t5_cuda_seconds"],
            "vae_decode_cuda_seconds": performance["vae_decode_cuda_seconds"],
            "estimated_t5_tflops_per_video": performance[
                "estimated_t5_tflops_per_video"
            ],
            "estimated_vae_decode_tflops_per_video": performance[
                "estimated_vae_decode_tflops_per_video"
            ],
            "protocol": PROTOCOL,
            "thread_environment": thread_environment(),
        }
        atomic_json(paths["complete"], marker)
        return

    baseline = baseline_paths(args.parent_root, sample_id)
    if metric_evaluator is None:
        raise RuntimeError("candidate collection requires the full-reference metric evaluator")
    metric_result = metric_evaluator.evaluate(
        reference=baseline["video"],
        candidate=paths["video"],
        video_id=identity,
        candidate_root=paths["root"],
    )
    psnr = metric_result["metrics"]["psnr_rgb_db"]
    ssim = metric_result["metrics"]["ssim_rgb"]
    lpips = metric_result["metrics"]["lpips_alex_v0_1_spatial"]
    baseline_performance = read_json(baseline["performance"])
    comparison = compare_matched(baseline_performance, performance)
    summary_row = {
        **row,
        "baseline_video": str(baseline["video"].resolve()),
        "baseline_trace_json": str(baseline["trace"].resolve()),
        "baseline_latent_dir": str(baseline["latents"].resolve()),
        "candidate_video": str(paths["video"].resolve()),
        "trace_json": str(paths["trace"].resolve()),
        "latent_dir": str(paths["latents"].resolve()),
        "baseline_inference_seconds": baseline_performance["pipeline_generate_wall_seconds"],
        "candidate_inference_seconds": performance["pipeline_generate_wall_seconds"],
        "inference_latency_speedup": comparison["inference_latency_speedup"],
        "baseline_estimated_dit_tflops": baseline_performance["estimated_dit_tflops_per_video"],
        "candidate_estimated_dit_tflops": performance["estimated_dit_tflops_per_video"],
        "dit_flops_speedup": comparison["dit_flops_speedup"],
        "candidate_estimated_tflops_per_second": performance["estimated_achieved_tflops_per_second"],
        "baseline_dit_cuda_seconds": baseline_performance["dit_cuda_seconds"],
        "candidate_dit_cuda_seconds": performance["dit_cuda_seconds"],
        "baseline_t5_cuda_seconds": baseline_performance["t5_cuda_seconds"],
        "candidate_t5_cuda_seconds": performance["t5_cuda_seconds"],
        "baseline_vae_decode_cuda_seconds": baseline_performance[
            "vae_decode_cuda_seconds"
        ],
        "candidate_vae_decode_cuda_seconds": performance[
            "vae_decode_cuda_seconds"
        ],
        "baseline_estimated_t5_tflops_per_video": baseline_performance[
            "estimated_t5_tflops_per_video"
        ],
        "candidate_estimated_t5_tflops_per_video": performance[
            "estimated_t5_tflops_per_video"
        ],
        "baseline_estimated_vae_decode_tflops_per_video": baseline_performance[
            "estimated_vae_decode_tflops_per_video"
        ],
        "candidate_estimated_vae_decode_tflops_per_video": performance[
            "estimated_vae_decode_tflops_per_video"
        ],
        "mean_psnr": psnr["mean"],
        "std_psnr": psnr["std_population"],
        "min_psnr": psnr["min"],
        "max_psnr": psnr["max"],
        "mean_ssim": ssim["mean"],
        "std_ssim": ssim["std_population"],
        "min_ssim": ssim["min"],
        "max_ssim": ssim["max"],
        "mean_lpips": lpips["mean"],
        "std_lpips": lpips["std_population"],
        "min_lpips": lpips["min"],
        "max_lpips": lpips["max"],
        "metric_frames": metric_result["frames"],
        "psnr_frames": metric_result["frames"],
        "psnr_protocol": metric_result["protocol_id"],
        "video_metrics_protocol": metric_result["protocol_id"],
        "video_metrics_names": metric_result["metric_names"],
        "video_metrics_json": str(paths["metrics"].resolve()),
        "video_metrics_per_frame_csv": str(paths["metrics_per_frame"].resolve()),
        "video_metrics_per_video_csv": str(paths["metrics_per_video"].resolve()),
        "video_metrics_summary_json": str(paths["metrics_summary"].resolve()),
        "video_metrics_evaluation_seconds": metric_result["evaluation_elapsed_seconds"],
        "video_metrics_timing_scope": metric_result["timing_scope"],
        "action_count_unit": "cfg_branch_call",
        "actual_reuse": int(trace["reuse"]),
        "actual_recompute": int(trace["recompute"]),
        "actual_reuse_branch_calls": int(trace["reuse"]),
        "actual_recompute_branch_calls": int(trace["recompute"]),
        "cond_reuse_branch_calls": int(trace["per_branch"]["cond"]["reuse"]),
        "cond_recompute_branch_calls": int(trace["per_branch"]["cond"]["recompute"]),
        "uncond_reuse_branch_calls": int(trace["per_branch"]["uncond"]["reuse"]),
        "uncond_recompute_branch_calls": int(trace["per_branch"]["uncond"]["recompute"]),
        "actual_both_reuse_steps": sum(
            step["action"] == "reuse" for step in trace["step_records"]
        ),
        "actual_both_recompute_steps": sum(
            step["action"] == "recompute" for step in trace["step_records"]
        ),
        "actual_mixed_steps": sum(
            step["action"] == "mixed" for step in trace["step_records"]
        ),
        "timing_scope": "pipeline_generate_wall_seconds",
        "speedup_definition": comparison["speedup_definition"],
        "flops_speedup_definition": comparison["flops_speedup_definition"],
    }
    step_identity = {
        "release_index": int(row["release_index"]),
        "trajectory_id": identity,
        "sample_id": sample_id,
        "prompt_rank": int(row["prompt_rank"]),
        "split": row["split"],
        "candidate_index_for_prompt": int(row["candidate_index_for_prompt"]),
        "shard_index": int(row["shard_index"]),
        "target_speedup": float(row["target_speedup"]),
        "q": float(row["q"]),
        "mean_threshold": float(row["mean_threshold"]),
    }
    step_rows = [
        {
            **step_identity,
            **step,
            "baseline_latent_path": str(
                (baseline["latents"] / f"step_{int(step['step_index']):03d}_input.pt").resolve()
            ),
            "final_mean_psnr": psnr["mean"],
            "final_mean_ssim": ssim["mean"],
            "final_mean_lpips": lpips["mean"],
            "final_inference_latency_speedup": comparison["inference_latency_speedup"],
            "final_dit_flops_speedup": comparison["dit_flops_speedup"],
            "baseline_inference_seconds": baseline_performance["pipeline_generate_wall_seconds"],
            "candidate_inference_seconds": performance["pipeline_generate_wall_seconds"],
        }
        for step in trace["step_records"]
    ]
    if len(step_rows) != NUM_STEPS:
        raise RuntimeError("candidate trace did not produce 50 step rows")
    step_by_index = {int(step["step_index"]): step for step in step_rows}
    branch_rows = []
    for decision in trace["decisions"]:
        step_index = int(decision["step_index"])
        step = step_by_index[step_index]
        branch_rows.append({
            **step_identity,
            "step_index": step_index,
            "step_fraction": step["step_fraction"],
            "timestep": step["timestep"],
            "sigma": step["sigma"],
            "model_stage": step["model_stage"],
            **decision,
            "latent_path": step["latent_path"],
            "baseline_latent_path": step["baseline_latent_path"],
            "latent_shape": step["latent_shape"],
            "latent_dtype": step["latent_dtype"],
            "final_mean_psnr": psnr["mean"],
            "final_mean_ssim": ssim["mean"],
            "final_mean_lpips": lpips["mean"],
            "final_inference_latency_speedup": comparison["inference_latency_speedup"],
            "final_dit_flops_speedup": comparison["dit_flops_speedup"],
            "baseline_inference_seconds": baseline_performance["pipeline_generate_wall_seconds"],
            "candidate_inference_seconds": performance["pipeline_generate_wall_seconds"],
        })
    if len(branch_rows) != 2 * NUM_STEPS:
        raise RuntimeError("candidate trace did not produce 100 CFG branch rows")
    completion = {
        "schema": SCHEMA_CANDIDATE_COMPLETE,
        "release_index": int(row["release_index"]),
        "trajectory_id": identity,
        "trajectory_row": summary_row,
        "step_rows": step_rows,
        "branch_rows": branch_rows,
        "thread_environment": thread_environment(),
    }
    atomic_json(paths["complete"], completion)


def write_shard_summary(args: argparse.Namespace, selected: list[dict[str, Any]]) -> None:
    shard_root = args.parent_root / "shards" / f"shard_{args.shard_index:02d}"
    if args.mode == "baseline":
        completed = sum(
            baseline_complete(baseline_paths(args.parent_root, str(row["sample_id"])), row)
            for row in selected
        )
    else:
        completed = sum(
            candidate_complete(candidate_paths(args.parent_root, args.shard_index, str(row["trajectory_id"])))
            for row in selected
        )
    atomic_json(
        shard_root / f"{args.mode}_shard_summary.json",
        {
            "status": "complete" if completed == len(selected) else "partial",
            "mode": args.mode,
            "shard_index": args.shard_index,
            "expected": len(selected),
            "completed": completed,
            "thread_environment": thread_environment(),
        },
    )


def main() -> None:
    args = parse_args()
    if args.num_shards != NUM_SHARDS:
        raise ValueError("v1 collection is fixed to four shards")
    if args.lpips_batch_size < 1:
        raise ValueError("--lpips-batch-size must be at least one")
    args.manifest = args.manifest.expanduser().resolve(strict=True)
    args.parent_root = require_result_path(args.parent_root)
    args.wan21_root = args.wan21_root.expanduser().resolve(strict=True)
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve(strict=True)
    args.flops_profile = args.flops_profile.expanduser().resolve(strict=True)
    if shutil.which(args.ffprobe_bin) is None:
        raise FileNotFoundError("ffprobe is missing")
    rows = load_manifest(args.manifest, args.mode)
    selected = selected_rows(rows, args.mode, args.shard_index)
    validate_flops_profile(args.flops_profile)
    if args.cpu_validate:
        print(json.dumps(cpu_validate(args, rows), ensure_ascii=False, indent=2))
        return
    if args.verify_baselines:
        ready, missing = all_baselines_complete(rows, args.parent_root)
        payload = {"status": "ok" if ready else "incomplete", "missing_count": len(missing), "missing": missing}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not ready:
            raise SystemExit(2)
        return
    require_thread_limits()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU collection requires CUDA")
    if args.mode == "candidate":
        ready, missing = all_baselines_complete(rows, args.parent_root)
        if not ready:
            raise RuntimeError(f"candidate phase is blocked by {len(missing)} incomplete baselines")
    args.parent_root.mkdir(parents=True, exist_ok=True)
    (args.parent_root / "completed").mkdir(exist_ok=True)
    shard_root = args.parent_root / "shards" / f"shard_{args.shard_index:02d}"
    shard_root.mkdir(parents=True, exist_ok=True)
    metrics_model_info: dict[str, Any] | None = None
    if args.mode == "candidate":
        cache_root, alexnet_weight, alexnet_sha = resolve_metrics_model_cache(args.metrics_model_cache)
        args.metrics_model_cache = cache_root
        metrics_model_info = {
            "selected_metrics": ["psnr", "ssim", "lpips"],
            "protocol_id": "rgb_full_reference_v1",
            "device": args.metrics_device,
            "lpips_batch_size": args.lpips_batch_size,
            "torch_home": str(cache_root),
            "alexnet_weight": str(alexnet_weight),
            "alexnet_weight_sha256": alexnet_sha,
            "timing_scope": "evaluation_only_excluded_from_inference_speedup",
        }
    config = {
        "schema": "ours4wan21_collection_shard_config_v1",
        "mode": args.mode,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "wan21_root": str(args.wan21_root),
        "checkpoint_dir": str(args.checkpoint_dir),
        "flops_profile": str(args.flops_profile),
        "flops_profile_sha256": file_sha256(args.flops_profile),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_count": len(selected),
        "protocol": PROTOCOL,
        "full_reference_metrics": metrics_model_info,
        "thread_environment": thread_environment(),
    }
    config_path = shard_root / f"{args.mode}_config.json"
    if config_path.exists():
        if not args.resume or read_json(config_path) != config:
            raise ValueError(f"existing shard config differs: {config_path}")
    else:
        atomic_json(config_path, config)
    pipeline, init_seconds, cache_video = create_pipeline(args.wan21_root, args.checkpoint_dir)
    runtime = Wan21DataRuntime(pipeline, args.mode)
    metric_evaluator = (
        FullReferenceMetricEvaluator(
            device=args.metrics_device,
            lpips_batch_size=args.lpips_batch_size,
            model_cache=args.metrics_model_cache,
        )
        if args.mode == "candidate"
        else None
    )
    failure_dir = shard_root / "failures"
    failure_dir.mkdir(exist_ok=True)
    try:
        for row in selected:
            try:
                collect_one(
                    args, row, pipeline, runtime, init_seconds, cache_video,
                    args.flops_profile, metric_evaluator,
                )
            except Exception as exc:
                atomic_json(
                    failure_dir / (
                        f"baseline_{row['sample_id']}.json"
                        if args.mode == "baseline"
                        else f"{row['trajectory_id']}.json"
                    ),
                    {"row": row, "error": repr(exc)},
                )
                raise
    finally:
        write_shard_summary(args, selected)


if __name__ == "__main__":
    main()
