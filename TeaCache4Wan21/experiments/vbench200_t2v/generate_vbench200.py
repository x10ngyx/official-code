#!/usr/bin/env python3
"""Generate Vbench200 T2V videos with locked Wan2.1 or official TeaCache."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent
PROMPTS_PATH = REPOSITORY_DIR / "Vbench200" / "prompts.jsonl"
ENTRYPOINT = PROJECT_DIR / "generate.py"
VALIDATOR = PROJECT_DIR / "validate_reproduction.py"
EXP_ROOT = Path("/mnt/hdd/xiongyuxiang/tmp/exp").resolve()
EXPECTED_WAN21_COMMIT = "65386b2e03c490796eede31b0325a6a595cc684e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("wan21", "teacache"), required=True)
    parser.add_argument("--task", choices=("t2v-1.3B", "t2v-14B"), required=True)
    parser.add_argument("--wan21-root", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacache-thresh", type=float)
    parser.add_argument("--use-ret-steps", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--size")
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sample-shift", type=float, default=5.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--sample-solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--no-offload-model", action="store_false", dest="offload_model")
    parser.add_argument("--t5-on-gpu", action="store_false", dest="t5_cpu")
    parser.set_defaults(offload_model=True, t5_cpu=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_external_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as exc:
        raise ValueError(f"output must be below {EXP_ROOT}: {resolved}") from exc
    return resolved


def load_prompts(
    limit: int | None,
    sample_ids: list[str] | None,
) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 200:
        raise ValueError(f"expected 200 Vbench200 prompts, found {len(rows)}")
    if limit is not None:
        if limit < 1 or limit > len(rows):
            raise ValueError(f"--limit must be between 1 and {len(rows)}")
        rows = rows[:limit]
    elif sample_ids is not None:
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("--sample-ids must not contain duplicates")
        by_id = {str(row["sample_id"]): row for row in rows}
        unknown = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if unknown:
            raise ValueError(f"unknown Vbench200 sample ids: {unknown}")
        rows = [by_id[sample_id] for sample_id in sample_ids]
    return rows


def validate_args(args: argparse.Namespace) -> None:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must satisfy 0 <= index < num_shards")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    if args.size is None:
        args.size = "832*480" if args.task == "t2v-1.3B" else "1280*720"
    if args.implementation == "teacache":
        if args.teacache_thresh is None or args.teacache_thresh < 0:
            raise ValueError("TeaCache generation requires --teacache-thresh >= 0")
        checkpoint_marker = "1.3B" if args.task == "t2v-1.3B" else "14B"
        if checkpoint_marker not in str(args.ckpt_dir):
            raise ValueError(
                "official TeaCache selects T2V coefficients from the checkpoint-path marker "
                f"{checkpoint_marker!r}"
            )
    elif args.teacache_thresh is not None or args.use_ret_steps:
        raise ValueError("TeaCache-only options cannot be used with --implementation wan21")
    if not args.wan21_root.is_dir():
        raise FileNotFoundError(args.wan21_root)
    if not args.dry_run and not args.ckpt_dir.is_dir():
        raise FileNotFoundError(args.ckpt_dir)

    subprocess.run(
        [sys.executable, str(VALIDATOR), "--wan21-root", str(args.wan21_root)],
        check=True,
    )


def output_name(sample_id: str, seed_index: int, seed_count: int) -> str:
    if seed_count == 1:
        return f"{sample_id}.mp4"
    return f"{sample_id}-{seed_index}.mp4"


def build_command(
    args: argparse.Namespace,
    sample: dict[str, object],
    seed: int,
    output: Path,
    timing_output: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ENTRYPOINT),
        "--wan21_root",
        str(args.wan21_root),
        "--task",
        args.task,
        "--size",
        args.size,
        "--frame_num",
        str(args.frame_num),
        "--ckpt_dir",
        str(args.ckpt_dir.resolve()),
        "--prompt",
        str(sample["prompt_en"]),
        "--base_seed",
        str(seed),
        "--offload_model",
        str(args.offload_model),
        "--sample_solver",
        args.sample_solver,
        "--sample_steps",
        str(args.sample_steps),
        "--sample_shift",
        str(args.sample_shift),
        "--sample_guide_scale",
        str(args.guide_scale),
        "--save_file",
        str(output),
    ]
    if timing_output is not None:
        command.extend(("--timing_json", str(timing_output)))
    if args.t5_cpu:
        command.append("--t5_cpu")
    if args.implementation == "teacache":
        command.extend(
            (
                "--enable_teacache",
                "--teacache_thresh",
                str(args.teacache_thresh),
            )
        )
        if args.use_ret_steps:
            command.append("--use_ret_steps")
    return command


def write_config(path: Path, config: dict[str, object], resume: bool) -> None:
    payload = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if not resume:
            raise FileExistsError(f"configuration already exists; use --resume: {path}")
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"resume configuration mismatch: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def ensure_result_readme(output_dir: Path) -> None:
    path = output_dir / "README.md"
    if path.exists():
        return
    path.write_text(
        "# TeaCache4Wan21 generation result\n\n"
        "This directory is an external experiment artifact produced by "
        "`experiments/vbench200_t2v/generate_vbench200.py`. "
        "Videos are under `videos/`, model-internal performance traces under "
        "`timings/`, process logs under `logs/`, and shard-level configuration/"
        "manifests at the directory root. `pipeline_generate_wall_seconds` is the "
        "inference latency: it excludes model loading and MP4 export.\n",
        encoding="utf-8",
    )


def append_manifest(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    args.wan21_root = args.wan21_root.expanduser().resolve()
    args.ckpt_dir = args.ckpt_dir.expanduser().resolve()
    args.output_dir = require_external_output(args.output_dir)
    validate_args(args)
    prompts = load_prompts(args.limit, args.sample_ids)

    jobs: list[tuple[int, dict[str, object], int, int]] = []
    for sample in prompts:
        for seed_index, seed in enumerate(args.seeds):
            ordinal = len(jobs)
            jobs.append((ordinal, sample, seed_index, seed))
    shard_jobs = [job for job in jobs if job[0] % args.num_shards == args.shard_index]

    videos_dir = args.output_dir / "videos"
    logs_dir = args.output_dir / "logs"
    timings_dir = args.output_dir / "timings"
    if not args.dry_run:
        videos_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        timings_dir.mkdir(parents=True, exist_ok=True)
        ensure_result_readme(args.output_dir)

    original_source = args.wan21_root / "generate.py"
    config = {
        "schema_version": 2,
        "implementation": args.implementation,
        "task": args.task,
        "wan21_root": str(args.wan21_root),
        "wan21_commit": EXPECTED_WAN21_COMMIT,
        "entrypoint": str(ENTRYPOINT),
        "entrypoint_sha256": sha256(ENTRYPOINT),
        "original_wan21_entrypoint": str(original_source),
        "original_wan21_entrypoint_sha256": sha256(original_source),
        "checkpoint_dir": str(args.ckpt_dir),
        "prompt_manifest": str(PROMPTS_PATH),
        "prompt_manifest_sha256": sha256(PROMPTS_PATH),
        "prompt_count": len(prompts),
        "selected_sample_ids": [str(sample["sample_id"]) for sample in prompts],
        "seeds": args.seeds,
        "size": args.size,
        "frame_num": args.frame_num,
        "output_fps": 16,
        "parameter_dtype": "bfloat16",
        "sample_steps": args.sample_steps,
        "sample_shift": args.sample_shift,
        "guide_scale": args.guide_scale,
        "sample_solver": args.sample_solver,
        "offload_model": args.offload_model,
        "t5_cpu": args.t5_cpu,
        "teacache_thresh": args.teacache_thresh,
        "use_ret_steps": args.use_ret_steps,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "job_count_total": len(jobs),
        "job_count_shard": len(shard_jobs),
        "performance_trace": {
            "enabled": True,
            "inference_latency_field": "pipeline_generate_wall_seconds",
            "inference_latency_excludes": ["model_loading", "mp4_export"],
        },
    }
    config_path = args.output_dir / f"generation_config.shard_{args.shard_index:03d}.json"
    if not args.dry_run:
        write_config(config_path, config, args.resume)

    manifest_path = args.output_dir / f"generation_manifest.shard_{args.shard_index:03d}.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.wan21_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    completed = 0
    skipped = 0
    for ordinal, sample, seed_index, seed in shard_jobs:
        filename = output_name(str(sample["sample_id"]), seed_index, len(args.seeds))
        output = videos_dir / filename
        timing_output = timings_dir / f"{output.stem}.json"
        command = build_command(args, sample, seed, output, timing_output)
        if args.dry_run:
            print(shlex.join(command))
            continue
        if output.exists():
            if args.resume and output.stat().st_size > 0:
                if not timing_output.is_file() or timing_output.stat().st_size == 0:
                    raise RuntimeError(
                        "resume found a video without its required inference timing "
                        f"trace: {timing_output}"
                    )
                previous_timing = json.loads(
                    timing_output.read_text(encoding="utf-8")
                )
                expected_implementation = (
                    "teacache" if args.implementation == "teacache" else "wan21"
                )
                if (
                    not isinstance(previous_timing, dict)
                    or previous_timing.get("status") != "success"
                    or previous_timing.get("implementation")
                    != expected_implementation
                ):
                    raise RuntimeError(
                        "resume found an invalid inference timing trace: "
                        f"{timing_output}"
                    )
                skipped += 1
                continue
            raise FileExistsError(f"refusing to overwrite existing output: {output}")

        started = dt.datetime.now(dt.timezone.utc)
        attempt = started.strftime("%Y%m%dT%H%M%SZ")
        log_path = logs_dir / f"{output.stem}.{attempt}.log"
        start_monotonic = time.monotonic()
        with log_path.open("x", encoding="utf-8") as log_handle:
            log_handle.write(f"command: {shlex.join(command)}\n")
            log_handle.flush()
            result = subprocess.run(
                command,
                cwd=args.wan21_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        finished = dt.datetime.now(dt.timezone.utc)
        timing = None
        if timing_output.is_file() and timing_output.stat().st_size > 0:
            timing = json.loads(timing_output.read_text(encoding="utf-8"))
        record = {
            "job_ordinal": ordinal,
            "sample_id": sample["sample_id"],
            "prompt_en": sample["prompt_en"],
            "seed_index": seed_index,
            "seed": seed,
            "output": str(output),
            "output_bytes": output.stat().st_size if output.exists() else None,
            "log": str(log_path),
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "wall_seconds": time.monotonic() - start_monotonic,
            "process_wall_scope": "includes model loading, inference, and MP4 export",
            "timing": str(timing_output),
            "pipeline_generate_wall_seconds": (
                timing.get("pipeline_generate_wall_seconds")
                if isinstance(timing, dict)
                else None
            ),
            "model_forward_cuda_seconds": (
                timing.get("model_forward_cuda_seconds")
                if isinstance(timing, dict)
                else None
            ),
            "full_compute_forward_calls": (
                timing.get("full_compute_forward_calls")
                if isinstance(timing, dict)
                else None
            ),
            "reuse_forward_calls": (
                timing.get("reuse_forward_calls")
                if isinstance(timing, dict)
                else None
            ),
            "returncode": result.returncode,
        }
        append_manifest(manifest_path, record)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"generation returned success without a non-empty video: {output}")
        if not isinstance(timing, dict) or timing.get("status") != "success":
            raise RuntimeError(
                "generation returned success without a successful inference timing "
                f"trace: {timing_output}"
            )
        completed += 1

    summary = {
        "status": "dry_run" if args.dry_run else "complete",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "jobs_assigned": len(shard_jobs),
        "completed": completed,
        "skipped": skipped,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
