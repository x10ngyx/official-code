#!/usr/bin/env python3
"""Atomically publish the longest contiguous completed candidate prefix."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from .manifest import (
    NUM_CANDIDATES,
    NUM_STEPS,
    SCHEMA_CANDIDATE_COMPLETE,
    read_jsonl,
    validate_runnable,
)
from .metrics import load_metric_artifacts, metric_paths
from .paths import require_result_path


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


TRAJECTORY_FIELDS = (
    "release_index", "trajectory_id", "sample_id", "prompt_rank", "split",
    "candidate_index_for_prompt", "shard_index", "target_speedup", "q",
    "mean_threshold", "threshold_min", "threshold_max", "action_count_unit",
    "actual_reuse", "actual_recompute", "actual_reuse_branch_calls",
    "actual_recompute_branch_calls", "cond_reuse_branch_calls",
    "cond_recompute_branch_calls", "uncond_reuse_branch_calls",
    "uncond_recompute_branch_calls", "actual_both_reuse_steps",
    "actual_both_recompute_steps", "actual_mixed_steps",
    "baseline_inference_seconds", "candidate_inference_seconds",
    "inference_latency_speedup", "baseline_estimated_dit_tflops",
    "candidate_estimated_dit_tflops", "dit_flops_speedup",
    "candidate_estimated_tflops_per_second",
    "baseline_t5_cuda_seconds", "candidate_t5_cuda_seconds",
    "baseline_dit_cuda_seconds", "candidate_dit_cuda_seconds",
    "baseline_vae_decode_cuda_seconds", "candidate_vae_decode_cuda_seconds",
    "baseline_estimated_t5_tflops_per_video",
    "candidate_estimated_t5_tflops_per_video",
    "baseline_estimated_vae_decode_tflops_per_video",
    "candidate_estimated_vae_decode_tflops_per_video",
    "mean_psnr", "std_psnr", "min_psnr", "max_psnr",
    "mean_ssim", "std_ssim", "min_ssim", "max_ssim",
    "mean_lpips", "std_lpips", "min_lpips", "max_lpips",
    "metric_frames", "psnr_frames", "psnr_protocol", "video_metrics_protocol",
    "video_metrics_names", "video_metrics_json", "video_metrics_per_frame_csv",
    "video_metrics_per_video_csv", "video_metrics_summary_json",
    "video_metrics_evaluation_seconds", "video_metrics_timing_scope",
    "prompt", "baseline_video",
    "baseline_trace_json", "baseline_latent_dir", "candidate_video", "trace_json",
    "latent_dir", "timing_scope",
    "speedup_definition", "flops_speedup_definition", "calibration_file",
    "calibration_sha256", "calibration_fit_source", "calibration_fit_source_sha256",
)
STEP_FIELDS = (
    "release_index", "trajectory_id", "sample_id", "prompt_rank", "split",
    "candidate_index_for_prompt", "shard_index", "target_speedup", "q",
    "mean_threshold", "step_index", "step_fraction", "timestep", "sigma",
    "model_stage", "requested_threshold", "action", "reason", "branches",
    "cond_action", "uncond_action", "filtered_relative_l1",
    "cond_filtered_relative_l1", "uncond_filtered_relative_l1",
    "accumulated_distance_before", "accumulated_distance_with_current",
    "accumulated_distance_after", "cond_accumulated_distance_before",
    "cond_accumulated_distance_with_current", "cond_accumulated_distance_after",
    "uncond_accumulated_distance_before", "uncond_accumulated_distance_with_current",
    "uncond_accumulated_distance_after", "distance_reference", "distance_feature",
    "distance_metric", "native_forced_recompute", "branch_decisions",
    "latent_path", "baseline_latent_path", "latent_shape", "latent_dtype", "latent_mean",
    "latent_std", "latent_min", "latent_max", "final_mean_psnr",
    "final_mean_ssim", "final_mean_lpips",
    "final_inference_latency_speedup", "final_dit_flops_speedup",
    "baseline_inference_seconds", "candidate_inference_seconds",
)
BRANCH_FIELDS = (
    "release_index", "trajectory_id", "sample_id", "prompt_rank", "split",
    "candidate_index_for_prompt", "shard_index", "target_speedup", "q",
    "mean_threshold", "call_index", "step_index", "branch", "step_fraction",
    "timestep", "sigma", "model_stage", "requested_threshold", "action", "reason",
    "filtered_relative_l1", "accumulated_distance_before",
    "accumulated_distance_with_current", "accumulated_distance_after",
    "distance_reference", "distance_feature", "distance_metric", "stored_feature",
    "native_forced_recompute", "execution", "latent_path", "baseline_latent_path",
    "latent_shape", "latent_dtype", "final_mean_psnr", "final_mean_ssim",
    "final_mean_lpips", "final_inference_latency_speedup", "final_dit_flops_speedup",
    "baseline_inference_seconds", "candidate_inference_seconds",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_completion(parent: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    paths = candidate_paths(parent, int(row["shard_index"]), str(row["trajectory_id"]))
    path = paths["complete"]
    if not path.is_file() or path.stat().st_size == 0:
        return None
    required = (
        "video", "timing", "performance", "trace", "ffprobe", "metrics",
        "metrics_per_frame", "metrics_per_video", "metrics_summary",
    )
    if any(not paths[key].is_file() or paths[key].stat().st_size == 0 for key in required):
        return None
    expected_latents = {f"step_{index:03d}_input.pt" for index in range(NUM_STEPS)}
    if (
        not paths["latents"].is_dir()
        or {path.name for path in paths["latents"].iterdir() if path.is_file()}
        != expected_latents
    ):
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA_CANDIDATE_COMPLETE:
        raise ValueError(f"invalid completion schema: {path}")
    if int(payload.get("release_index", -1)) != int(row["release_index"]):
        raise ValueError(f"completion release index mismatch: {path}")
    if payload.get("trajectory_id") != row["trajectory_id"]:
        raise ValueError(f"completion trajectory mismatch: {path}")
    trajectory = payload.get("trajectory_row")
    steps = payload.get("step_rows")
    branches = payload.get("branch_rows")
    if not isinstance(trajectory, dict) or not isinstance(steps, list) or len(steps) != NUM_STEPS:
        raise ValueError(f"completion tables are malformed: {path}")
    if [int(step.get("step_index", -1)) for step in steps] != list(range(NUM_STEPS)):
        raise ValueError(f"completion step order is malformed: {path}")
    if not isinstance(branches, list) or len(branches) != 2 * NUM_STEPS:
        raise ValueError(f"completion branch table is malformed: {path}")
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
        for branch in branches
    ]
    if observed_branch_order != expected_branch_order:
        raise ValueError(f"completion branch order is malformed: {path}")
    load_metric_artifacts(paths["root"])
    return payload


def completed_prefix(rows: list[dict[str, Any]], parent: Path) -> int:
    count = 0
    for row in rows:
        if load_completion(parent, row) is None:
            break
        count += 1
    return count


def numeric_summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "mean": sum(finite) / len(finite),
        "max": max(finite),
    }


def write_snapshot(
    rows: list[dict[str, Any]], parent: Path, destination: Path, manifest: Path
) -> dict[str, Any]:
    destination.mkdir(parents=True)
    tables = destination / "tables"
    tables.mkdir()
    trajectory_jsonl = tables / "trajectory_summary.jsonl"
    trajectory_csv = tables / "trajectory_summary.csv"
    step_jsonl = tables / "step_transitions.jsonl"
    step_csv = tables / "step_transitions.csv"
    branch_jsonl = tables / "branch_transitions.jsonl"
    branch_csv = tables / "branch_transitions.csv"
    trajectory_rows: list[dict[str, Any]] = []
    with (
        trajectory_jsonl.open("x", encoding="utf-8") as trajectory_json_handle,
        trajectory_csv.open("x", encoding="utf-8", newline="") as trajectory_csv_handle,
        step_jsonl.open("x", encoding="utf-8") as step_json_handle,
        step_csv.open("x", encoding="utf-8", newline="") as step_csv_handle,
        branch_jsonl.open("x", encoding="utf-8") as branch_json_handle,
        branch_csv.open("x", encoding="utf-8", newline="") as branch_csv_handle,
    ):
        trajectory_writer = csv.DictWriter(trajectory_csv_handle, fieldnames=TRAJECTORY_FIELDS, extrasaction="ignore")
        step_writer = csv.DictWriter(step_csv_handle, fieldnames=STEP_FIELDS, extrasaction="ignore")
        branch_writer = csv.DictWriter(
            branch_csv_handle, fieldnames=BRANCH_FIELDS, extrasaction="ignore"
        )
        trajectory_writer.writeheader()
        step_writer.writeheader()
        branch_writer.writeheader()
        for row in rows:
            completion = load_completion(parent, row)
            if completion is None:
                raise RuntimeError("published prefix changed while snapshot was being written")
            trajectory = completion["trajectory_row"]
            trajectory_rows.append(trajectory)
            trajectory_json_handle.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
            trajectory_writer.writerow({key: json_text(trajectory.get(key)) for key in TRAJECTORY_FIELDS})
            for step in completion["step_rows"]:
                step_json_handle.write(json.dumps(step, ensure_ascii=False) + "\n")
                step_writer.writerow({key: json_text(step.get(key)) for key in STEP_FIELDS})
            for branch in completion["branch_rows"]:
                branch_json_handle.write(json.dumps(branch, ensure_ascii=False) + "\n")
                branch_writer.writerow({key: json_text(branch.get(key)) for key in BRANCH_FIELDS})
    summary = {
        "schema": "ours4wan21_published_snapshot_v3",
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "archive_root": str(parent),
        "published_candidate_count": len(rows),
        "published_step_count": len(rows) * NUM_STEPS,
        "published_branch_transition_count": len(rows) * 2 * NUM_STEPS,
        "complete": len(rows) == NUM_CANDIDATES,
        "distributions": {
            key: numeric_summary(item.get(key) for item in trajectory_rows)
            for key in (
                "target_speedup", "q", "mean_threshold", "inference_latency_speedup",
                "baseline_inference_seconds", "candidate_inference_seconds",
                "baseline_estimated_dit_tflops", "candidate_estimated_dit_tflops",
                "dit_flops_speedup", "baseline_t5_cuda_seconds",
                "candidate_t5_cuda_seconds", "baseline_dit_cuda_seconds",
                "candidate_dit_cuda_seconds", "baseline_vae_decode_cuda_seconds",
                "candidate_vae_decode_cuda_seconds", "mean_psnr", "mean_ssim", "mean_lpips",
                "video_metrics_evaluation_seconds", "actual_reuse", "actual_recompute",
            )
        },
        "tables": {
            path.name: {"path": f"tables/{path.name}", "sha256": sha256(path)}
            for path in (
                trajectory_jsonl, trajectory_csv, step_jsonl, step_csv,
                branch_jsonl, branch_csv,
            )
        },
    }
    atomic_json(destination / "SUMMARY.json", summary)
    return summary


def publish(manifest: Path, parent: Path, *, require_complete: bool = False) -> dict[str, Any]:
    manifest = manifest.expanduser().resolve(strict=True)
    parent = require_result_path(parent)
    rows = read_jsonl(manifest)
    validate_runnable(rows)
    published_root = parent / "published"
    published_root.mkdir(parents=True, exist_ok=True)
    lock_path = published_root / ".publish.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        prefix = completed_prefix(rows, parent)
        if require_complete and prefix != NUM_CANDIDATES:
            raise RuntimeError(f"complete publication requires 9000 candidates; prefix={prefix}")
        snapshots = published_root / "snapshots"
        snapshots.mkdir(exist_ok=True)
        destination = snapshots / f"prefix_{prefix:09d}"
        if destination.exists():
            summary_path = destination / "SUMMARY.json"
            if not summary_path.is_file():
                raise RuntimeError(f"existing snapshot is incomplete: {destination}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("schema") != "ours4wan21_published_snapshot_v3":
                raise RuntimeError("existing prefix snapshot uses an obsolete schema")
            if summary.get("manifest_sha256") != sha256(manifest):
                raise RuntimeError("existing prefix snapshot belongs to a different manifest")
            if int(summary.get("published_candidate_count", -1)) != prefix:
                raise RuntimeError("existing prefix snapshot count is inconsistent")
            for table in summary.get("tables", {}).values():
                table_path = destination / str(table["path"])
                if not table_path.is_file() or sha256(table_path) != table.get("sha256"):
                    raise RuntimeError(f"existing prefix snapshot table is corrupt: {table_path}")
        else:
            temporary = snapshots / f".prefix_{prefix:09d}.tmp.{os.getpid()}"
            if temporary.exists():
                shutil.rmtree(temporary)
            try:
                summary = write_snapshot(rows[:prefix], parent, temporary, manifest)
                os.replace(temporary, destination)
            except BaseException:
                if temporary.exists():
                    shutil.rmtree(temporary)
                raise
        current = {
            "schema": "ours4wan21_current_publication_v3",
            "snapshot": str(destination.resolve()),
            "published_candidate_count": prefix,
            "published_step_count": prefix * NUM_STEPS,
            "published_branch_transition_count": prefix * 2 * NUM_STEPS,
            "complete": prefix == NUM_CANDIDATES,
            "summary_sha256": sha256(destination / "SUMMARY.json"),
        }
        atomic_json(published_root / "CURRENT.json", current)
        return {**current, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = publish(args.manifest, args.parent_root, require_complete=args.require_complete)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
