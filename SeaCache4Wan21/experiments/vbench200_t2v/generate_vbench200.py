#!/usr/bin/env python3
"""Generate one statically sharded SeaCache4Wan21 Vbench200 condition."""

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
ENTRYPOINT = PROJECT_DIR / "generate.py"
VALIDATOR = PROJECT_DIR / "validate_reproduction.py"
EXP_ROOT = Path("/all/yiran07-disk3/huteng_data/exp").resolve()
EXPECTED_WAN21_COMMIT = "65386b2e03c490796eede31b0325a6a595cc684e"
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
sys.path.insert(0, str(REPOSITORY_DIR / "ComponentMetrics"))
from reporting import extract_component_latency  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as exc:
        raise ValueError(f"output must be below {EXP_ROOT}: {resolved}") from exc
    return resolved


def load_prompts(limit: int | None) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 200:
        raise ValueError(f"expected 200 Vbench200 prompts, found {len(rows)}")
    if limit is not None:
        if not 1 <= limit <= 200:
            raise ValueError("--limit must be in [1, 200]")
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


def read_success(path: Path, implementation: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "success" or payload.get("implementation") != implementation:
        raise ValueError(f"invalid timing implementation in {path}")
    seconds = payload.get("pipeline_generate_wall_seconds")
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"invalid pipeline timing in {path}")
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
    parser.add_argument("--wan21-root", type=Path, required=True)
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

    args.wan21_root = args.wan21_root.expanduser().resolve(strict=True)
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve(strict=not args.dry_run)
    args.output_dir = require_external(args.output_dir)
    subprocess.run(
        [sys.executable, str(VALIDATOR), "--wan21-root", str(args.wan21_root)],
        check=True,
        env={**os.environ, **THREAD_ENV},
    )
    prompts = load_prompts(args.limit)
    jobs = [(index, row) for index, row in enumerate(prompts) if index % args.num_shards == args.shard_index]

    videos = args.output_dir / "videos"
    timings = args.output_dir / "timings"
    traces = args.output_dir / "traces"
    logs = args.output_dir / "logs"
    if not args.dry_run:
        for path in (videos, timings, logs):
            path.mkdir(parents=True, exist_ok=True)
        if args.condition == "seacache":
            traces.mkdir(parents=True, exist_ok=True)
        readme = args.output_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "# SeaCache4Wan21 Vbench200 generation artifacts\n\n"
                "`videos/` contains ID-named MP4 files, `timings/` contains "
                "inference-only wall time, `traces/` contains SeaCache decisions "
                "when enabled, and `logs/` contains process output.\n",
                encoding="utf-8",
            )

    config = {
        "schema": "seacache4wan21_vbench200_generation_v1",
        "condition": args.condition,
        "threshold": args.threshold,
        "use_ret_steps": args.use_ret_steps,
        "wan21_root": str(args.wan21_root),
        "wan21_commit": EXPECTED_WAN21_COMMIT,
        "wan21_generate_sha256": sha256(args.wan21_root / "generate.py"),
        "entrypoint": str(ENTRYPOINT),
        "entrypoint_sha256": sha256(ENTRYPOINT),
        "checkpoint_dir": str(args.checkpoint_dir),
        "prompt_manifest": str(PROMPTS_PATH),
        "prompt_manifest_sha256": sha256(PROMPTS_PATH),
        "selected_prompt_count": len(prompts),
        "protocol": {
            "task": "t2v-1.3B", "size": "832*480", "frame_num": 81,
            "fps": 16, "sample_steps": 50, "sample_solver": "unipc",
            "sample_shift": 5.0, "guide_scale": 5.0, "seed": 42,
            "dtype": "bfloat16", "offload_model": False, "t5_cpu": False,
        },
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "thread_env": THREAD_ENV,
    }
    if not args.dry_run:
        write_locked(
            args.output_dir / f"generation_config.shard_{args.shard_index:03d}.json",
            config,
            args.resume,
        )

    manifest = args.output_dir / f"generation_manifest.shard_{args.shard_index:03d}.jsonl"
    env = {**os.environ, **THREAD_ENV}
    env["PYTHONPATH"] = str(args.wan21_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = skipped = 0
    for ordinal, sample in jobs:
        sample_id = str(sample["sample_id"])
        video = videos / f"{sample_id}.mp4"
        timing = timings / f"{sample_id}.json"
        trace = traces / f"{sample_id}.json"
        command = [
            sys.executable, str(ENTRYPOINT), "--wan21_root", str(args.wan21_root),
            "--task", "t2v-1.3B", "--size", "832*480", "--frame_num", "81",
            "--ckpt_dir", str(args.checkpoint_dir), "--prompt", str(sample["prompt_en"]),
            "--base_seed", "42", "--offload_model", "False",
            "--sample_solver", "unipc", "--sample_steps", "50",
            "--sample_shift", "5", "--sample_guide_scale", "5",
            "--timing_json", str(timing), "--save_file", str(video),
        ]
        if args.condition == "seacache":
            command.extend([
                "--enable_seacache", "--seacache_thresh", str(args.threshold),
                "--seacache_trace", str(trace),
            ])
            if args.use_ret_steps:
                command.append("--use_ret_steps")
        if args.dry_run:
            print(shlex.join(command))
            continue
        expected_impl = "seacache" if args.condition == "seacache" else "wan21"
        required = [video, timing] + ([trace] if args.condition == "seacache" else [])
        if any(path.exists() for path in required):
            if args.resume and all(path.is_file() and path.stat().st_size > 0 for path in required):
                read_success(timing, expected_impl)
                skipped += 1
                continue
            raise FileExistsError(f"incomplete or existing output for {sample_id}")

        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log = logs / f"{sample_id}.{stamp}.log"
        started = time.monotonic()
        with log.open("x", encoding="utf-8") as handle:
            handle.write(f"command: {shlex.join(command)}\n")
            handle.flush()
            result = subprocess.run(
                command, cwd=args.wan21_root, env=env, stdout=handle,
                stderr=subprocess.STDOUT, text=True,
            )
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError(f"missing generated video: {video}")
        timing_payload = read_success(timing, expected_impl)
        if args.condition == "seacache":
            trace_payload = json.loads(trace.read_text(encoding="utf-8"))
            if trace_payload.get("schema") != "seacache4wan21_trace_v3" or len(trace_payload.get("decisions", [])) != 100:
                raise ValueError(f"invalid SeaCache trace: {trace}")
        append_jsonl(manifest, {
            "job_ordinal": ordinal, "sample_id": sample_id,
            "prompt_en": sample["prompt_en"], "seed": 42,
            "video": str(video), "timing": str(timing),
            "trace": str(trace) if args.condition == "seacache" else None,
            "log": str(log),
            "pipeline_generate_wall_seconds": timing_payload["pipeline_generate_wall_seconds"],
            "process_wall_seconds": time.monotonic() - started,
        })
        completed += 1
    print(json.dumps({"status": "dry_run" if args.dry_run else "complete", "assigned": len(jobs), "completed": completed, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
