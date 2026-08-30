#!/usr/bin/env python3
"""Run the complete SeaCache4Wan21 Vbench200 benchmark."""

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
WORKSPACE_ROOT = REPOSITORY_DIR.parents[1]
EXP_ROOT = Path("/all/yiran07-disk3/huteng_data/exp").resolve()
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "models" / "Wan2.1-T2V-1.3B"
DEFAULT_VBENCH_CACHE = WORKSPACE_ROOT / "models" / "VBench"
DEFAULT_METRICS_CACHE = WORKSPACE_ROOT / "models" / "torch-cache"
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}


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


def check_python(path: Path, modules: list[str], label: str) -> None:
    result = subprocess.run(
        [str(path), "-c", ";".join(f"import {name}" for name in modules)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ, **THREAD_ENV},
    )
    if result.returncode:
        raise RuntimeError(f"{label} Python lacks {modules}: {path}\n{result.stderr}")


def write_locked(path: Path, payload: dict[str, Any], resume: bool) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if not resume:
            raise FileExistsError(f"configuration exists; use --resume: {path}")
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"resume configuration mismatch: {path}")
        return
    path.write_text(rendered, encoding="utf-8")


def status_write(root: Path, payload: dict[str, Any]) -> None:
    (root / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_logged(command: list[str], log: Path, *, env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    if dry_run:
        print(shlex.join(command))
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        log = log.with_name(f"{log.stem}.{stamp}{log.suffix}")
    with log.open("x", encoding="utf-8") as handle:
        handle.write(f"command: {shlex.join(command)}\n")
        handle.flush()
        subprocess.run(command, env=env or {**os.environ, **THREAD_ENV}, stdout=handle, stderr=subprocess.STDOUT, text=True, check=True)


def generation_command(args: argparse.Namespace, condition: str, shard: int) -> list[str]:
    command = [
        str(args.generation_python), str(SCRIPT_DIR / "generate_vbench200.py"),
        "--condition", condition, "--wan21-root", str(args.wan21_root),
        "--checkpoint-dir", str(args.checkpoint_dir),
        "--output-dir", str(args.output_dir / condition),
        "--shard-index", str(shard), "--num-shards", str(len(args.gpu_ids)),
    ]
    if condition == "seacache":
        command.extend(["--threshold", str(args.threshold)])
        if args.use_ret_steps:
            command.append("--use-ret-steps")
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.resume:
        command.append("--resume")
    return command


def run_generation(args: argparse.Namespace, condition: str, status: dict[str, Any]) -> None:
    commands = [generation_command(args, condition, shard) for shard in range(len(args.gpu_ids))]
    if args.dry_run:
        for gpu, command in zip(args.gpu_ids, commands):
            print(f"CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}")
        return
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    processes = []
    started = time.monotonic()
    for shard, (gpu, command) in enumerate(zip(args.gpu_ids, commands)):
        log = args.output_dir / "orchestration_logs" / f"generate_{condition}_shard_{shard}.{stamp}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("x", encoding="utf-8")
        handle.write(f"CUDA_VISIBLE_DEVICES={gpu}\ncommand: {shlex.join(command)}\n")
        handle.flush()
        env = {**os.environ, **THREAD_ENV, "CUDA_VISIBLE_DEVICES": gpu}
        processes.append((subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True), handle, log))
    failures = []
    for process, handle, log in processes:
        code = process.wait()
        handle.close()
        if code:
            failures.append({"returncode": code, "log": str(log)})
    status["phases"][f"generation_{condition}"] = {
        "status": "failed" if failures else "complete",
        "orchestration_wall_seconds": time.monotonic() - started, "failures": failures,
    }
    status_write(args.output_dir, status)
    if failures:
        raise RuntimeError(f"{condition} generation failed: {failures}")


def link_result(root: Path) -> None:
    link = PROJECT_DIR / "experiment_results" / root.name
    if link.is_symlink() and link.resolve() == root:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"result link already exists: {link}")
    link.symlink_to(root, target_is_directory=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--wan21-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gpu-ids", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--generation-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--calflops-source", type=Path)
    parser.add_argument("--video-metrics-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--vbench-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--video-metrics-cache-dir", type=Path, default=DEFAULT_METRICS_CACHE)
    parser.add_argument("--vbench-cache-dir", type=Path, default=DEFAULT_VBENCH_CACHE)
    parser.add_argument("--use-ret-steps", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        raise ValueError("--threshold must be finite and positive")
    if not args.gpu_ids or len(set(args.gpu_ids)) != len(args.gpu_ids) or any(not gpu.isdigit() for gpu in args.gpu_ids):
        raise ValueError("--gpu-ids must be distinct non-negative integers")
    if args.limit is not None and (not 1 <= args.limit <= 200 or not args.skip_evaluation):
        raise ValueError("--limit must be in [1,200] and requires --skip-evaluation")
    args.output_dir = external(args.output_dir)
    args.wan21_root = args.wan21_root.expanduser().resolve(strict=True)
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve(strict=not args.dry_run)
    for name in ("generation_python", "video_metrics_python", "vbench_python"):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    args.video_metrics_cache_dir = args.video_metrics_cache_dir.expanduser().resolve(strict=not args.dry_run)
    args.vbench_cache_dir = args.vbench_cache_dir.expanduser().resolve(strict=not args.dry_run)
    if args.calflops_source is not None:
        args.calflops_source = args.calflops_source.expanduser().resolve(strict=True)
    if not args.dry_run:
        check_python(args.generation_python, ["torch"], "generation")
        if args.calflops_source is None:
            check_python(args.generation_python, ["calflops"], "Calflops")
        if not args.skip_evaluation:
            check_python(args.video_metrics_python, ["torch", "lpips"], "VideoMetrics")
            check_python(args.vbench_python, ["torch", "vbench"], "VBench")

    config = {
        "schema": "seacache4wan21_vbench200_run_v1", "method": "SeaCache4Wan21",
        "threshold": args.threshold, "use_ret_steps": args.use_ret_steps,
        "protocol": {
            "dataset": "Vbench200", "prompt_count": args.limit or 200,
            "model": "Wan2.1-T2V-1.3B", "video": {"width": 832, "height": 480, "frames": 81, "fps": 16},
            "sampling": {"steps": 50, "solver": "unipc", "shift": 5.0, "cfg": 5.0, "seed": 42},
            "precision": "DiT bfloat16", "model_cpu_offload": False, "t5_cpu": False,
        },
        "gpu_ids": args.gpu_ids,
        "paths": {
            "wan21_root": str(args.wan21_root), "checkpoint_dir": str(args.checkpoint_dir),
            "generation_python": str(args.generation_python), "video_metrics_python": str(args.video_metrics_python),
            "vbench_python": str(args.vbench_python), "output_dir": str(args.output_dir),
            "calflops_source": str(args.calflops_source) if args.calflops_source else None,
        },
        "scripts": {path.name: sha256(path) for path in sorted(SCRIPT_DIR.glob("*.py"))},
        "shared_profile_script": {
            "path": str(SCRIPT_DIR.parent / "performance_t2v_1_3b" / "profile_calflops.py"),
            "sha256": sha256(SCRIPT_DIR.parent / "performance_t2v_1_3b" / "profile_calflops.py"),
        },
        "evaluation_enabled": not args.skip_evaluation, "thread_env": THREAD_ENV,
    }
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
    else:
        if args.output_dir.exists() and not args.resume:
            raise FileExistsError(f"output exists; use --resume: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_locked(args.output_dir / "run_config.json", config, args.resume)
        if not (args.output_dir / "README.md").exists():
            (args.output_dir / "README.md").write_text(
                "# SeaCache4Wan21 Vbench200 result\n\nBaseline/candidate generation, trace-weighted performance, repository-standard quality metrics, and final reports.\n",
                encoding="utf-8",
            )
        link_result(args.output_dir)
    status = {"status": "running", "phases": {}}
    if not args.dry_run:
        status_write(args.output_dir, status)

    run_generation(args, "baseline", status)
    run_generation(args, "seacache", status)
    profile = args.output_dir / "performance" / "calflops_profile.json"
    if not (args.resume and profile.is_file()):
        profile_command = [
            str(args.generation_python), str(SCRIPT_DIR / "profile_calflops.py"),
            "--wan21-root", str(args.wan21_root), "--checkpoint-dir", str(args.checkpoint_dir), "--output", str(profile),
        ]
        if args.calflops_source is not None:
            profile_command.extend(["--calflops-source", str(args.calflops_source)])
        run_logged(profile_command, args.output_dir / "orchestration_logs" / "calflops_profile.log",
            env={**os.environ, **THREAD_ENV, "CUDA_VISIBLE_DEVICES": args.gpu_ids[0]}, dry_run=args.dry_run)
    performance = args.output_dir / "performance" / "summary.json"
    if not (args.resume and performance.is_file()):
        run_logged([
            str(args.generation_python), str(SCRIPT_DIR / "aggregate_performance.py"),
            "--baseline-dir", str(args.output_dir / "baseline"), "--seacache-dir", str(args.output_dir / "seacache"),
            "--calflops-profile", str(profile), "--output-dir", str(args.output_dir / "performance"),
            "--expected-videos", str(args.limit or 200),
        ], args.output_dir / "orchestration_logs" / "aggregate_performance.log", dry_run=args.dry_run)
    if not args.dry_run:
        status["phases"]["performance"] = {"status": "complete", "summary": str(performance)}
        status_write(args.output_dir, status)

    if not args.skip_evaluation:
        evaluation_summary = args.output_dir / "evaluation" / "video_metrics" / "summary.json"
        if not (args.resume and evaluation_summary.is_file() and (args.output_dir / "evaluation" / "vbench_candidate" / "vbench200_aggregate_scores.json").is_file()):
            command = [
                str(args.generation_python), str(SCRIPT_DIR / "evaluate_results.py"),
                "--reference-videos", str(args.output_dir / "baseline" / "videos"),
                "--candidate-videos", str(args.output_dir / "seacache" / "videos"),
                "--output-dir", str(args.output_dir / "evaluation"), "--expected-frames", "81",
                "--gpu-id", args.gpu_ids[0], "--video-metrics-python", str(args.video_metrics_python),
                "--vbench-python", str(args.vbench_python), "--video-metrics-cache-dir", str(args.video_metrics_cache_dir),
                "--vbench-cache-dir", str(args.vbench_cache_dir),
            ]
            if args.resume:
                command.append("--resume")
            run_logged(command, args.output_dir / "orchestration_logs" / "evaluation.log", dry_run=args.dry_run)
        report = args.output_dir / "benchmark_report.json"
        if not (args.resume and report.is_file()):
            run_logged([
                str(args.generation_python), str(SCRIPT_DIR / "build_final_report.py"), "--result-dir", str(args.output_dir)
            ], args.output_dir / "orchestration_logs" / "final_report.log", dry_run=args.dry_run)
        if not args.dry_run:
            status["phases"]["evaluation_and_report"] = {"status": "complete", "report": str(report)}
    if not args.dry_run:
        status["status"] = "complete"
        status_write(args.output_dir, status)


if __name__ == "__main__":
    main()
