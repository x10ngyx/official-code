#!/usr/bin/env python3
"""Generate one VBench200 shard with one persistent WanT2V pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent
PROMPTS_PATH = REPOSITORY_DIR / "Vbench200" / "prompts.jsonl"
VALIDATOR = PROJECT_DIR / "scripts" / "validate_prepared_tree.py"
EXP_ROOT = Path("/all/yiran07-disk3/huteng_data/exp").resolve()
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}
for _name, _value in THREAD_ENV.items():
    os.environ[_name] = _value
sys.path.insert(0, str(REPOSITORY_DIR / "ComponentMetrics"))
from reporting import extract_component_latency  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as exc:
        raise ValueError(f"output must be below {EXP_ROOT}: {resolved}") from exc
    return resolved


def load_prompts(limit: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 200:
        raise ValueError(f"expected 200 Vbench200 prompts, found {len(rows)}")
    if limit is not None:
        if not 1 <= limit <= 200:
            raise ValueError("--limit must be in [1,200]")
        rows = rows[:limit]
    return rows


def write_locked(path: Path, payload: dict[str, Any], resume: bool) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if not resume:
            raise FileExistsError(f"configuration exists; use --resume: {path}")
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"resume configuration mismatch: {path}")
        return
    path.write_text(rendered, encoding="utf-8")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_timing(
    path: Path, implementation: str, *, require_persistent: bool = False
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    seconds = payload.get("pipeline_generate_wall_seconds")
    if payload.get("status") != "success" or payload.get("implementation") != implementation or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"invalid timing trace: {path}")
    extract_component_latency(payload)
    lifecycle = payload.get("pipeline_lifecycle")
    if require_persistent and (
        not isinstance(lifecycle, dict)
        or lifecycle.get("persistent_pipeline") is not True
        or lifecycle.get("pipeline_init_accounted_in_this_sample") is not False
    ):
        raise ValueError(f"timing trace lacks persistent-pipeline provenance: {path}")
    return payload


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("baseline", "seacache"), required=True)
    parser.add_argument("--wan22-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--use-ret-steps", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--physical-gpu",
        help="Physical GPU identifier exposed as the worker's sole CUDA device.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    if args.condition == "seacache":
        if args.threshold is None or not math.isfinite(args.threshold) or args.threshold <= 0:
            raise ValueError("SeaCache requires a finite --threshold > 0")
    elif args.threshold is not None or args.use_ret_steps:
        raise ValueError("SeaCache options cannot be used for baseline")
    args.wan22_root = args.wan22_root.expanduser().resolve(strict=True)
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve(strict=not args.dry_run)
    args.output_dir = external(args.output_dir)
    subprocess.run(
        [sys.executable, str(VALIDATOR), "--source", str(args.wan22_root), "--mode", "prepared"],
        check=True, env={**os.environ, **THREAD_ENV},
    )
    prepared_manifest = args.wan22_root / ".seacache4wan22_prepared.json"
    if not prepared_manifest.is_file():
        raise FileNotFoundError(f"missing prepared-source manifest: {prepared_manifest}")
    prompts = load_prompts(args.limit)
    jobs = [(index, row) for index, row in enumerate(prompts) if index % args.num_shards == args.shard_index]
    videos, timings = args.output_dir / "videos", args.output_dir / "timings"
    traces, logs = args.output_dir / "traces", args.output_dir / "logs"
    worker_status = args.output_dir / "worker_status" / f"worker_{args.shard_index:03d}.json"
    if not args.dry_run:
        for path in (videos, timings, logs, worker_status.parent):
            path.mkdir(parents=True, exist_ok=True)
        if args.condition == "seacache":
            traces.mkdir(parents=True, exist_ok=True)
        readme = args.output_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "# SeaCache4Wan22 Vbench200 generation artifacts\n\n"
                "ID-named videos, inference-only timing traces, SeaCache decision "
                "traces when enabled, per-sample logs, and persistent-worker status. "
                "Each GPU loads WanT2V once and processes its shard sequentially.\n",
                encoding="utf-8",
            )
    config = {
        "schema": "seacache4wan22_vbench200_generation_v2", "condition": args.condition,
        "threshold": args.threshold, "use_ret_steps": args.use_ret_steps,
        "wan22_root": str(args.wan22_root),
        "prepared_manifest": str(prepared_manifest), "prepared_manifest_sha256": sha256(prepared_manifest),
        "checkpoint_dir": str(args.checkpoint_dir), "prompt_manifest": str(PROMPTS_PATH),
        "prompt_manifest_sha256": sha256(PROMPTS_PATH), "selected_prompt_count": len(prompts),
        "protocol": {
            "task": "t2v-A14B", "size": "832*480", "frame_num": 45, "fps": 16,
            "sample_steps": 50, "sample_solver": "dpm++", "sample_shift": 12.0,
            "guide_scale_low_high": [3.0, 4.0], "boundary": 0.875, "seed": 42,
            "dtype": "bfloat16", "offload_model": True, "t5_cpu": False,
        },
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "physical_gpu": args.physical_gpu,
        "runner": {
            "type": "persistent_batch_worker",
            "persistent_pipeline": True,
            "pipeline_initializations_per_worker": 1 if jobs else 0,
            "sample_batch_size": 1,
            "assigned_sample_count": len(jobs),
            "profiler_lifecycle": "fresh_install_and_restore_per_sample",
        },
        "thread_env": THREAD_ENV,
    }
    if not args.dry_run:
        write_locked(args.output_dir / f"generation_config.shard_{args.shard_index:03d}.json", config, args.resume)
    manifest = args.output_dir / f"generation_manifest.shard_{args.shard_index:03d}.jsonl"
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run",
            "runner": config["runner"],
            "assigned_sample_ids": [str(row["sample_id"]) for _, row in jobs],
        }, ensure_ascii=False, indent=2))
        return

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices or "," in visible_devices:
        raise ValueError(
            "persistent worker requires exactly one CUDA_VISIBLE_DEVICES entry"
        )
    if args.physical_gpu is not None and visible_devices != args.physical_gpu:
        raise ValueError(
            f"--physical-gpu={args.physical_gpu} does not match "
            f"CUDA_VISIBLE_DEVICES={visible_devices}"
        )
    physical_gpu = args.physical_gpu or visible_devices

    sys.path.insert(0, str(args.wan22_root))
    import torch
    import wan
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
    from wan.inference_timing import _PipelineProfiler
    from wan.seacache import SeaCacheConfig
    from wan.utils.utils import save_video

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )
    torch.cuda.set_device(0)
    completed = skipped = 0
    runtime = {
        "schema_version": 1,
        "status": "initializing" if jobs else "complete",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "worker_index": args.shard_index,
        "worker_count": args.num_shards,
        "worker_pid": os.getpid(),
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": visible_devices,
        "persistent_pipeline": True,
        "pipeline_initialization_count": 0,
        "assigned_sample_ids": [str(row["sample_id"]) for _, row in jobs],
        "completed_sample_count": 0,
        "skipped_sample_count": 0,
    }
    atomic_json(worker_status, runtime)
    if not jobs:
        runtime["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(worker_status, runtime)
        print(json.dumps({"status": "complete", "assigned": 0, "completed": 0, "skipped": 0}, indent=2))
        return

    init_started = time.perf_counter()
    pipeline = wan.WanT2V(
        config=WAN_CONFIGS["t2v-A14B"],
        checkpoint_dir=str(args.checkpoint_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
    )
    torch.cuda.synchronize()
    pipeline_init_wall_seconds = time.perf_counter() - init_started
    runtime.update({
        "status": "running",
        "cuda_device_name": torch.cuda.get_device_name(0),
        "pipeline_initialization_count": 1,
        "pipeline_init_wall_seconds_once": pipeline_init_wall_seconds,
    })
    atomic_json(worker_status, runtime)
    logging.info(
        "Persistent WanT2V pipeline initialized once in %.3f seconds on physical GPU %s",
        pipeline_init_wall_seconds,
        physical_gpu,
    )

    manifest_ids: set[str] = set()
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                manifest_ids.add(str(json.loads(line)["sample_id"]))

    try:
        for ordinal, sample in jobs:
            sample_id = str(sample["sample_id"])
            video, timing = videos / f"{sample_id}.mp4", timings / f"{sample_id}.json"
            trace = traces / f"{sample_id}.json"
            required = [video, timing] + ([trace] if args.condition == "seacache" else [])
            expected_impl = "seacache" if args.condition == "seacache" else "wan22"
            if any(path.exists() for path in required):
                if args.resume and all(path.is_file() and path.stat().st_size > 0 for path in required):
                    read_timing(timing, expected_impl, require_persistent=True)
                    if args.condition == "seacache":
                        trace_payload = json.loads(trace.read_text(encoding="utf-8"))
                        if trace_payload.get("schema") != "seacache4wan22_trace_v1" or len(trace_payload.get("decisions", [])) != 50:
                            raise ValueError(f"invalid SeaCache trace: {trace}")
                    skipped += 1
                    runtime["skipped_sample_count"] = skipped
                    atomic_json(worker_status, runtime)
                    continue
                raise FileExistsError(f"incomplete or existing output for {sample_id}")

            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log = logs / f"{sample_id}.{stamp}.log"
            runtime["current_sample_id"] = sample_id
            atomic_json(worker_status, runtime)
            logging.info(
                "Generating sample=%s condition=%s prompt=%r",
                sample_id,
                args.condition,
                sample["prompt_en"],
            )
            started = time.perf_counter()
            profiler = _PipelineProfiler(
                pipeline,
                init_wall_seconds=0.0,
                output_path=timing,
                implementation=expected_impl,
            )
            profiler.install()
            cache_config = (
                SeaCacheConfig(
                    threshold=float(args.threshold),
                    trace_path=str(trace),
                    use_ret_steps=args.use_ret_steps,
                )
                if args.condition == "seacache"
                else None
            )
            generated = None
            try:
                generated = pipeline.generate(
                    str(sample["prompt_en"]),
                    size=SIZE_CONFIGS["832*480"],
                    frame_num=45,
                    shift=12.0,
                    sample_solver="dpm++",
                    sampling_steps=50,
                    guide_scale=(3.0, 4.0),
                    seed=42,
                    offload_model=True,
                    seacache_config=cache_config,
                )
                inference_finished = time.perf_counter()
                export_started = time.perf_counter()
                save_video(
                    tensor=generated[None],
                    save_file=str(video),
                    fps=16,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                torch.cuda.synchronize()
                export_finished = time.perf_counter()
            finally:
                if generated is not None:
                    del generated
                gc.collect()
                torch.cuda.empty_cache()

            if not video.is_file() or video.stat().st_size == 0:
                raise RuntimeError(f"missing generated video: {video}")

            timing_payload = json.loads(timing.read_text(encoding="utf-8"))
            timing_payload["pipeline_lifecycle"] = {
                "runner": "persistent_batch_worker",
                "persistent_pipeline": True,
                "worker_index": args.shard_index,
                "physical_gpu": physical_gpu,
                "pipeline_init_wall_seconds_once": pipeline_init_wall_seconds,
                "pipeline_init_accounted_in_this_sample": False,
                "profiler_freshly_installed_for_sample": True,
            }
            atomic_json(timing, timing_payload)
            timing_payload = read_timing(timing, expected_impl, require_persistent=True)
            if args.condition == "seacache":
                trace_payload = json.loads(trace.read_text(encoding="utf-8"))
                if trace_payload.get("schema") != "seacache4wan22_trace_v1" or len(trace_payload.get("decisions", [])) != 50:
                    raise ValueError(f"invalid SeaCache trace: {trace}")

            sample_payload = {
                "status": "success",
                "sample_id": sample_id,
                "condition": args.condition,
                "prompt_en": sample["prompt_en"],
                "seed": 42,
                "persistent_pipeline": True,
                "shared_pipeline_init_wall_seconds_once": pipeline_init_wall_seconds,
                "pipeline_generate_wall_seconds": timing_payload["pipeline_generate_wall_seconds"],
                "generation_call_wall_seconds_observed_by_worker": inference_finished - started,
                "video_export_wall_seconds": export_finished - export_started,
                "sample_wall_seconds_including_export": export_finished - started,
                "video": str(video),
                "timing": str(timing),
                "trace": str(trace) if args.condition == "seacache" else None,
            }
            log.write_text(json.dumps(sample_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if sample_id not in manifest_ids:
                append_jsonl(manifest, {
                    "job_ordinal": ordinal,
                    **sample_payload,
                    "log": str(log),
                    "process_wall_seconds": export_finished - started,
                })
                manifest_ids.add(sample_id)
            completed += 1
            runtime.update({
                "completed_sample_count": completed,
                "last_completed_sample_id": sample_id,
                "last_completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            })
            runtime.pop("current_sample_id", None)
            atomic_json(worker_status, runtime)
            logging.info("Completed sample=%s", sample_id)
    except BaseException as exc:
        runtime.update({
            "status": "failed",
            "error": repr(exc),
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        })
        atomic_json(worker_status, runtime)
        raise

    runtime.update({
        "status": "complete",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    runtime.pop("current_sample_id", None)
    atomic_json(worker_status, runtime)
    print(json.dumps({"status": "complete", "assigned": len(jobs), "completed": completed, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
