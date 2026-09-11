#!/usr/bin/env python3
"""Prepare or run the four-GPU, ten-sample SeaCache speedup calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PROJECT = SCRIPT_DIR.parents[1]
OFFICIAL_CODE = DATA_PROJECT.parents[1]
sys.path.insert(0, str(DATA_PROJECT / "src"))

from ours4wan21_data.collector import baseline_complete, baseline_paths  # noqa: E402
from ours4wan21_data.manifest import NUM_STEPS, PROTOCOL  # noqa: E402
from ours4wan21_data.performance import (  # noqa: E402
    compare_matched,
    read_json,
    write_performance,
)
from ours4wan21_data.runtime import Wan21DataRuntime, create_pipeline  # noqa: E402
from ours4wan21_data.timing import PipelineProfiler  # noqa: E402


SCHEMA = "ours4wan21_seacache_speedup_calibration_v1"
COMPLETE_SCHEMA = "ours4wan21_seacache_speedup_calibration_complete_v1"
DEFAULT_THRESHOLDS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.21, 0.24)
NUM_SAMPLES = 10
NUM_SHARDS = 4
THREAD_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"existing immutable file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def condition_label(threshold: float) -> str:
    return f"threshold_{threshold:.4f}".replace(".", "p")


def result_paths(root: Path, row: dict[str, Any]) -> dict[str, Path]:
    condition = condition_label(float(row["fixed_threshold"]))
    candidate_root = root / "shards" / f"shard_{int(row['shard_index']):02d}" / condition / str(row["sample_id"])
    return {
        "root": candidate_root,
        "timing": candidate_root / "timing.json",
        "performance": candidate_root / "performance.json",
        "trace": candidate_root / "trace.json",
        "complete": candidate_root / "CALIBRATION_COMPLETE.json",
    }


def completion_valid(paths: dict[str, Path], row: dict[str, Any]) -> bool:
    try:
        payload = read_json(paths["complete"])
        timing = read_json(paths["timing"])
        performance = read_json(paths["performance"])
        trace = read_json(paths["trace"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return (
        payload.get("schema") == COMPLETE_SCHEMA
        and payload.get("trajectory_id") == row.get("trajectory_id")
        and timing.get("status") == "success"
        and int(timing.get("model_forward_call_count", -1)) == 100
        and int(trace.get("reuse", -1)) + int(trace.get("recompute", -1)) == 100
        and performance.get("schema") == "ours4wan21_per_video_performance_v2"
    )


def validate_thread_limits() -> None:
    invalid = {key: os.environ.get(key) for key in THREAD_KEYS if os.environ.get(key) != "1"}
    if invalid:
        raise RuntimeError(f"all BLAS thread limits must equal one: {invalid}")


def select_samples(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in source_rows:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            continue
        seen.add(sample_id)
        selected.append(row)
        if len(selected) == NUM_SAMPLES:
            break
    if len(selected) != NUM_SAMPLES:
        raise ValueError(f"source manifest has fewer than {NUM_SAMPLES} unique samples")
    return selected


def build_rows(samples: list[dict[str, Any]], thresholds: tuple[float, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold_index, threshold in enumerate(thresholds):
        for sample_index, sample in enumerate(samples):
            release_index = threshold_index * NUM_SAMPLES + sample_index
            rows.append({
                **sample,
                "schema": SCHEMA,
                "release_index": release_index,
                "trajectory_id": f"{sample['sample_id']}__cal_{condition_label(threshold)}",
                "prompt_rank": sample_index,
                "candidate_index_for_prompt": threshold_index,
                "shard_index": release_index % NUM_SHARDS,
                "num_shards": NUM_SHARDS,
                "target_speedup": None,
                "q": None,
                "fixed_threshold": threshold,
                "mean_threshold": threshold,
                "threshold_min": threshold,
                "threshold_max": threshold,
                "threshold_path": [threshold] * NUM_STEPS,
                "threshold_grid": list(thresholds),
                "forced_recompute_steps": [0, NUM_STEPS - 1],
                "policy_family": "fixed_seacache_threshold",
                "calibration_status": "calibration_scan",
                "protocol": PROTOCOL,
            })
    validate_rows(rows, thresholds)
    return rows


def validate_rows(rows: list[dict[str, Any]], thresholds: tuple[float, ...]) -> None:
    expected = NUM_SAMPLES * len(thresholds)
    if len(rows) != expected or [int(row["release_index"]) for row in rows] != list(range(expected)):
        raise ValueError("calibration manifest release indices are incomplete")
    if len({str(row["trajectory_id"]) for row in rows}) != expected:
        raise ValueError("calibration trajectory IDs are not unique")
    sample_ids = {str(row["sample_id"]) for row in rows}
    if len(sample_ids) != NUM_SAMPLES:
        raise ValueError("calibration manifest must contain exactly ten samples")
    for threshold in thresholds:
        group = [row for row in rows if float(row["fixed_threshold"]) == threshold]
        if len(group) != NUM_SAMPLES:
            raise ValueError(f"threshold {threshold} does not contain ten samples")
    if max(sum(int(row["shard_index"]) == shard for row in rows) for shard in range(NUM_SHARDS)) - min(
        sum(int(row["shard_index"]) == shard for row in rows) for shard in range(NUM_SHARDS)
    ) > 1:
        raise ValueError("calibration manifest is not balanced across four shards")
    for row in rows:
        threshold = float(row["fixed_threshold"])
        if row.get("schema") != SCHEMA or row.get("protocol") != PROTOCOL:
            raise ValueError("calibration row schema/protocol mismatch")
        if threshold not in thresholds or row.get("threshold_path") != [threshold] * NUM_STEPS:
            raise ValueError("calibration threshold path mismatch")
        if int(row["shard_index"]) != int(row["release_index"]) % NUM_SHARDS:
            raise ValueError("calibration shard assignment mismatch")


def prepare(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    baseline_root = args.baseline_root.resolve(strict=True)
    source_manifest = args.source_manifest.resolve(strict=True)
    flops_profile = args.flops_profile.resolve(strict=True)
    thresholds = tuple(float(value) for value in args.thresholds)
    if len(set(thresholds)) != len(thresholds) or list(thresholds) != sorted(thresholds):
        raise ValueError("thresholds must be unique and sorted")
    if any(not math.isfinite(value) or value <= 0 for value in thresholds):
        raise ValueError("thresholds must be finite and positive")
    samples = select_samples(load_jsonl(source_manifest))
    rows = build_rows(samples, thresholds)
    for sample in samples:
        paths = baseline_paths(baseline_root, str(sample["sample_id"]))
        if not baseline_complete(paths, sample):
            raise RuntimeError(f"matched baseline is incomplete: {sample['sample_id']}")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)
    baseline_link = output_root / "shared_baselines"
    expected_target = baseline_root / "shared_baselines"
    if baseline_link.is_symlink():
        if baseline_link.resolve() != expected_target.resolve():
            raise ValueError(f"existing baseline symlink points elsewhere: {baseline_link}")
    elif baseline_link.exists():
        raise ValueError(f"baseline index exists and is not a symlink: {baseline_link}")
    else:
        baseline_link.symlink_to(expected_target)

    protocol = {
        "schema": SCHEMA,
        "purpose": "linear calibration of speedup-to-threshold and speedup-to-skip-count",
        "protocol": PROTOCOL,
        "sample_count": NUM_SAMPLES,
        "thresholds": list(thresholds),
        "target_speedup_domain": [1.5, 3.5],
        "sample_ids": [str(row["sample_id"]) for row in samples],
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "baseline_run_root": str(baseline_root),
        "baseline_mode": "reuse_matched_complete_bundles_via_symlink",
        "flops_profile": str(flops_profile),
        "flops_profile_sha256": sha256(flops_profile),
        "worker_shards": NUM_SHARDS,
        "warmup": "one unmeasured full 50-step threshold-0.10 generation per GPU after model load",
        "latency_headline": "sum matched baseline pipeline_generate_wall_seconds / sum candidate pipeline_generate_wall_seconds",
        "skip_primary": "reused CFG-branch DiT forward calls out of 100 per video",
        "skip_equivalent_steps": "reuse_branch_calls / 2, reported on a 50-step scale",
        "quality_metrics": "not_run; calibration scope is speed, threshold, and executed skip count",
        "source_sha256": {
            "run_calibration.py": sha256(Path(__file__)),
            "analyze_calibration.py": sha256(SCRIPT_DIR / "analyze_calibration.py"),
        },
    }
    rendered_protocol = json.dumps(protocol, indent=2, ensure_ascii=False) + "\n"
    rendered_manifest = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    write_once(output_root / "protocol.json", rendered_protocol)
    write_once(output_root / "manifests" / "calibration_manifest.jsonl", rendered_manifest)
    csv_path = output_root / "manifests" / "calibration_manifest.csv"
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "release_index", "trajectory_id", "sample_id", "prompt_rank",
                "fixed_threshold", "shard_index", "split", "part", "content_group",
                "length_group", "motion_group", "topic_tag", "prompt",
            ))
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in rows)
    readme = (
        "# SeaCache speedup calibration result\n\n"
        "Ten-sample, four-GPU fixed-threshold scan for the locked Wan2.1 protocol. "
        "See protocol.json, manifests/, shards/, analysis/, and logs/. "
        "shared_baselines is a symlink to the matched source run.\n"
    )
    write_once(output_root / "README.md", readme)
    print(json.dumps({"status": "prepared", "rows": len(rows), "output_root": str(output_root)}))


def warmup(pipeline: Any, runtime: Wan21DataRuntime, row: dict[str, Any]) -> None:
    warmup_row = {
        **row,
        "trajectory_id": f"warmup_shard_{row['shard_index']}",
        "fixed_threshold": 0.10,
        "mean_threshold": 0.10,
        "threshold_path": [0.10] * NUM_STEPS,
    }
    runtime.configure_sample(warmup_row)
    video = pipeline.generate(
        str(row["prompt"]), size=(832, 480), frame_num=81, shift=5.0,
        sample_solver="unipc", sampling_steps=50, guide_scale=5.0, seed=42,
        offload_model=False,
    )
    del video
    runtime.capture.latents.clear()
    runtime.capture.step_metadata.clear()
    runtime.capture.trace_payload = None
    torch.cuda.empty_cache()


def collect_one(args: argparse.Namespace, row: dict[str, Any], pipeline: Any, runtime: Wan21DataRuntime, init_seconds: float) -> None:
    paths = result_paths(args.output_root, row)
    if args.resume and completion_valid(paths, row):
        print(json.dumps({"status": "skipped", "trajectory_id": row["trajectory_id"]}), flush=True)
        return
    if paths["root"].exists():
        raise FileExistsError(f"incomplete result exists and is preserved for inspection: {paths['root']}")
    paths["root"].mkdir(parents=True)
    runtime.configure_sample(row)
    profiler = PipelineProfiler(
        pipeline,
        pipeline_init_wall_seconds=init_seconds,
        output_path=paths["timing"],
        implementation="seacache_fixed_threshold_calibration",
    )
    profiler.install()
    video = pipeline.generate(
        str(row["prompt"]), size=(832, 480), frame_num=81, shift=5.0,
        sample_solver="unipc", sampling_steps=50, guide_scale=5.0, seed=42,
        offload_model=False,
    )
    del video
    trace = runtime.capture.trace_payload
    if not isinstance(trace, dict):
        raise RuntimeError("candidate did not produce a SeaCache trace")
    trace = {**trace, "trajectory_id": row["trajectory_id"], "manifest_record": row}
    atomic_json(paths["trace"], trace)
    runtime.capture.latents.clear()
    runtime.capture.step_metadata.clear()
    runtime.capture.trace_payload = None
    torch.cuda.empty_cache()
    performance = write_performance(paths["timing"], args.flops_profile, paths["performance"])
    baseline = baseline_paths(args.output_root, str(row["sample_id"]))
    baseline_performance = read_json(baseline["performance"])
    comparison = compare_matched(baseline_performance, performance)
    decisions = trace.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 100:
        raise RuntimeError("SeaCache trace must contain 100 ordered CFG-branch decisions")
    both_reuse = both_recompute = mixed = 0
    for step in range(NUM_STEPS):
        pair = decisions[2 * step:2 * step + 2]
        actions = [decision.get("action") for decision in pair]
        if actions == ["reuse", "reuse"]:
            both_reuse += 1
        elif actions == ["recompute", "recompute"]:
            both_recompute += 1
        else:
            mixed += 1
    reuse = int(trace["reuse"])
    recompute = int(trace["recompute"])
    if reuse + recompute != 100 or both_reuse + both_recompute + mixed != NUM_STEPS:
        raise RuntimeError("SeaCache action counts do not close")
    completion = {
        "schema": COMPLETE_SCHEMA,
        "trajectory_id": row["trajectory_id"],
        "sample_id": row["sample_id"],
        "prompt_rank": row["prompt_rank"],
        "threshold": row["fixed_threshold"],
        "shard_index": row["shard_index"],
        "baseline_performance": str(baseline["performance"].resolve()),
        "timing": str(paths["timing"].resolve()),
        "performance": str(paths["performance"].resolve()),
        "trace": str(paths["trace"].resolve()),
        "baseline_inference_seconds": baseline_performance["pipeline_generate_wall_seconds"],
        "candidate_inference_seconds": performance["pipeline_generate_wall_seconds"],
        **comparison,
        "reuse_branch_calls": reuse,
        "recompute_branch_calls": recompute,
        "equivalent_reuse_steps": reuse / 2.0,
        "both_reuse_steps": both_reuse,
        "both_recompute_steps": both_recompute,
        "mixed_steps": mixed,
        "protocol": PROTOCOL,
    }
    atomic_json(paths["complete"], completion)
    print(json.dumps({
        "status": "complete", "trajectory_id": row["trajectory_id"],
        "threshold": row["fixed_threshold"],
        "speedup": comparison["inference_latency_speedup"],
        "reuse_branch_calls": reuse,
    }), flush=True)


def worker(args: argparse.Namespace) -> None:
    validate_thread_limits()
    args.output_root = args.output_root.resolve(strict=True)
    args.flops_profile = args.flops_profile.resolve(strict=True)
    rows = load_jsonl(args.output_root / "manifests" / "calibration_manifest.jsonl")
    protocol = read_json(args.output_root / "protocol.json")
    thresholds = tuple(float(value) for value in protocol["thresholds"])
    validate_rows(rows, thresholds)
    selected = [row for row in rows if int(row["shard_index"]) == args.shard_index]
    pipeline, init_seconds, _ = create_pipeline(args.wan21_root, args.checkpoint_dir)
    runtime = Wan21DataRuntime(pipeline, "candidate")
    if not args.skip_warmup:
        warmup(pipeline, runtime, selected[0])
    failure_dir = args.output_root / "failures"
    failure_dir.mkdir(exist_ok=True)
    for row in selected:
        try:
            collect_one(args, row, pipeline, runtime, init_seconds)
        except Exception as exc:
            atomic_json(failure_dir / f"{row['trajectory_id']}.json", {"row": row, "error": repr(exc)})
            raise
    atomic_json(args.output_root / "shards" / f"shard_{args.shard_index:02d}" / "SHARD_COMPLETE.json", {
        "status": "complete",
        "shard_index": args.shard_index,
        "expected": len(selected),
        "completed": sum(completion_valid(result_paths(args.output_root, row), row) for row in selected),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--baseline-root", type=Path, required=True)
    prepare_parser.add_argument("--source-manifest", type=Path, required=True)
    prepare_parser.add_argument("--flops-profile", type=Path, required=True)
    prepare_parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--output-root", type=Path, required=True)
    worker_parser.add_argument("--shard-index", type=int, choices=range(NUM_SHARDS), required=True)
    worker_parser.add_argument("--wan21-root", type=Path, required=True)
    worker_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    worker_parser.add_argument("--flops-profile", type=Path, required=True)
    worker_parser.add_argument("--resume", action="store_true")
    worker_parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
