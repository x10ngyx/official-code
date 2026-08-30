#!/usr/bin/env python3
"""Generate one statically sharded SeaCache4Wan22 Vbench200 condition."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shlex
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


def read_timing(path: Path, implementation: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    seconds = payload.get("pipeline_generate_wall_seconds")
    if payload.get("status") != "success" or payload.get("implementation") != implementation or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"invalid timing trace: {path}")
    extract_component_latency(payload)
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
    if not args.dry_run:
        for path in (videos, timings, logs):
            path.mkdir(parents=True, exist_ok=True)
        if args.condition == "seacache":
            traces.mkdir(parents=True, exist_ok=True)
        readme = args.output_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "# SeaCache4Wan22 Vbench200 generation artifacts\n\n"
                "ID-named videos, inference-only timing traces, SeaCache decision "
                "traces when enabled, and process logs.\n", encoding="utf-8",
            )
    config = {
        "schema": "seacache4wan22_vbench200_generation_v1", "condition": args.condition,
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
        "shard_index": args.shard_index, "num_shards": args.num_shards, "thread_env": THREAD_ENV,
    }
    if not args.dry_run:
        write_locked(args.output_dir / f"generation_config.shard_{args.shard_index:03d}.json", config, args.resume)
    manifest = args.output_dir / f"generation_manifest.shard_{args.shard_index:03d}.jsonl"
    env = {**os.environ, **THREAD_ENV}
    env["PYTHONPATH"] = str(args.wan22_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = skipped = 0
    for ordinal, sample in jobs:
        sample_id = str(sample["sample_id"])
        video, timing = videos / f"{sample_id}.mp4", timings / f"{sample_id}.json"
        trace = traces / f"{sample_id}.json"
        command = [
            sys.executable, str(args.wan22_root / "generate.py"), "--task", "t2v-A14B",
            "--size", "832*480", "--frame_num", "45", "--ckpt_dir", str(args.checkpoint_dir),
            "--offload_model", "true", "--sample_solver", "dpm++", "--sample_steps", "50",
            "--sample_shift", "12", "--base_seed", "42", "--convert_model_dtype",
            "--timing_trace", str(timing), "--prompt", str(sample["prompt_en"]),
            "--save_file", str(video), "--timestep_cache", "none",
        ]
        if args.condition == "seacache":
            command[-1] = "seacache"
            command.extend(["--seacache_threshold", str(args.threshold), "--seacache_trace", str(trace)])
            if args.use_ret_steps:
                command.append("--seacache_use_ret_steps")
        if args.dry_run:
            print(shlex.join(command))
            continue
        required = [video, timing] + ([trace] if args.condition == "seacache" else [])
        expected_impl = "seacache" if args.condition == "seacache" else "wan22"
        if any(path.exists() for path in required):
            if args.resume and all(path.is_file() and path.stat().st_size > 0 for path in required):
                read_timing(timing, expected_impl)
                skipped += 1
                continue
            raise FileExistsError(f"incomplete or existing output for {sample_id}")
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log = logs / f"{sample_id}.{stamp}.log"
        started = time.monotonic()
        with log.open("x", encoding="utf-8") as handle:
            handle.write(f"command: {shlex.join(command)}\n")
            handle.flush()
            result = subprocess.run(command, cwd=args.wan22_root, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError(f"missing generated video: {video}")
        timing_payload = read_timing(timing, expected_impl)
        if args.condition == "seacache":
            trace_payload = json.loads(trace.read_text(encoding="utf-8"))
            if trace_payload.get("schema") != "seacache4wan22_trace_v1" or len(trace_payload.get("decisions", [])) != 50:
                raise ValueError(f"invalid SeaCache trace: {trace}")
        append_jsonl(manifest, {
            "job_ordinal": ordinal, "sample_id": sample_id, "prompt_en": sample["prompt_en"], "seed": 42,
            "video": str(video), "timing": str(timing), "trace": str(trace) if args.condition == "seacache" else None,
            "log": str(log), "pipeline_generate_wall_seconds": timing_payload["pipeline_generate_wall_seconds"],
            "process_wall_seconds": time.monotonic() - started,
        })
        completed += 1
    print(json.dumps({"status": "dry_run" if args.dry_run else "complete", "assigned": len(jobs), "completed": completed, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
