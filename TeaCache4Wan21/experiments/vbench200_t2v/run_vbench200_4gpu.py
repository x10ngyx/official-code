#!/usr/bin/env python3
"""Run the locked four-GPU Wan2.1 baseline/TeaCache Vbench200 benchmark."""

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
WORKSPACE_ROOT = PROJECT_DIR.parents[2]
EXP_ROOT = Path("/all/yiran07-disk3/huteng_data/exp").resolve()
DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "models" / "Wan2.1-T2V-1.3B"
DEFAULT_VBENCH_CACHE = WORKSPACE_ROOT / "models" / "VBench"
DEFAULT_VIDEO_METRICS_CACHE = WORKSPACE_ROOT / "models" / "torch-cache"
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


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


def threshold_slug(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def check_python(path: Path, modules: list[str], label: str) -> None:
    expression = "; ".join(f"import {module}" for module in modules)
    result = subprocess.run(
        [str(path), "-c", expression],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **THREAD_ENV},
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} Python is missing required modules {modules}: {path}\n"
            f"{result.stderr.strip()}"
        )


def write_locked_json(path: Path, payload: dict[str, Any], resume: bool) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if not resume:
            raise FileExistsError(f"configuration exists; use --resume: {path}")
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"resume configuration mismatch: {path}")
        return
    path.write_text(rendered, encoding="utf-8")


def update_status(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def generation_command(
    args: argparse.Namespace,
    *,
    implementation: str,
    output_dir: Path,
    shard_index: int,
) -> list[str]:
    command = [
        str(args.generation_python),
        str(SCRIPT_DIR / "generate_vbench200.py"),
        "--implementation",
        implementation,
        "--task",
        "t2v-1.3B",
        "--wan21-root",
        str(args.wan21_root),
        "--ckpt-dir",
        str(args.checkpoint_dir),
        "--output-dir",
        str(output_dir),
        "--seeds",
        "42",
        "--size",
        "832*480",
        "--frame-num",
        "81",
        "--sample-steps",
        "50",
        "--sample-shift",
        "5",
        "--guide-scale",
        "5",
        "--sample-solver",
        "unipc",
        "--shard-index",
        str(shard_index),
        "--num-shards",
        "4",
    ]
    if implementation == "teacache":
        command.extend(("--teacache-thresh", str(args.teacache_thresh)))
        if args.use_ret_steps:
            command.append("--use-ret-steps")
    if args.resume:
        command.append("--resume")
    return command


def run_generation_condition(
    args: argparse.Namespace,
    *,
    implementation: str,
    output_dir: Path,
    status: dict[str, Any],
) -> None:
    commands = [
        generation_command(
            args,
            implementation=implementation,
            output_dir=output_dir,
            shard_index=index,
        )
        for index in range(4)
    ]
    if args.dry_run:
        for gpu_id, command in zip(args.gpu_ids, commands):
            print(f"CUDA_VISIBLE_DEVICES={gpu_id} {shlex.join(command)}")
        return

    started = time.monotonic()
    attempt = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    processes: list[tuple[subprocess.Popen[str], Any, Path]] = []
    for shard_index, (gpu_id, command) in enumerate(zip(args.gpu_ids, commands)):
        log_path = (
            args.output_dir
            / "orchestration_logs"
            / f"generation_{implementation}_shard_{shard_index}.{attempt}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("x", encoding="utf-8")
        handle.write(f"CUDA_VISIBLE_DEVICES={gpu_id}\n")
        handle.write(f"command: {shlex.join(command)}\n")
        handle.flush()
        env = os.environ.copy()
        env.update(THREAD_ENV)
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        process = subprocess.Popen(
            command,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((process, handle, log_path))

    failures: list[dict[str, Any]] = []
    for shard_index, (process, handle, log_path) in enumerate(processes):
        returncode = process.wait()
        handle.close()
        if returncode:
            failures.append(
                {
                    "shard_index": shard_index,
                    "returncode": returncode,
                    "log": str(log_path),
                }
            )
    elapsed = time.monotonic() - started
    status["phases"][f"generation_{implementation}"] = {
        "status": "failed" if failures else "complete",
        "orchestration_wall_seconds_including_model_load_and_mp4_export": elapsed,
        "failures": failures,
    }
    update_status(args.output_dir / "status.json", status)
    if failures:
        raise RuntimeError(f"{implementation} generation failed: {failures}")


def run_logged(
    command: list[str],
    *,
    gpu_id: str | None,
    log_path: Path,
    dry_run: bool,
) -> None:
    prefix = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id is not None else ""
    if dry_run:
        print(prefix + shlex.join(command))
        return
    env = os.environ.copy()
    env.update(THREAD_ENV)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as handle:
        handle.write(prefix + f"command: {shlex.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def run_performance(args: argparse.Namespace, status: dict[str, Any]) -> None:
    profile = args.output_dir / "performance" / "calflops_profile.json"
    summary = args.output_dir / "performance" / "summary.json"
    attempt = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not (args.resume and profile.is_file()):
        command = [
            str(args.generation_python),
            str(SCRIPT_DIR / "profile_calflops.py"),
            "--wan21-root",
            str(args.wan21_root),
            "--checkpoint-dir",
            str(args.checkpoint_dir),
            "--output",
            str(profile),
            "--width",
            "832",
            "--height",
            "480",
            "--frame-num",
            "81",
        ]
        if args.calflops_source is not None:
            command.extend(("--calflops-source", str(args.calflops_source)))
        run_logged(
            command,
            gpu_id=args.gpu_ids[0],
            log_path=(
                args.output_dir
                / "orchestration_logs"
                / f"calflops_profile.{attempt}.log"
            ),
            dry_run=args.dry_run,
        )
    elif not args.dry_run:
        print(f"resume: keeping {profile}")

    if not (args.resume and summary.is_file()):
        command = [
            str(args.generation_python),
            str(SCRIPT_DIR / "aggregate_performance.py"),
            "--baseline-dir",
            str(args.output_dir / "baseline"),
            "--teacache-dir",
            str(args.output_dir / "teacache"),
            "--calflops-profile",
            str(profile),
            "--output-dir",
            str(args.output_dir / "performance"),
            "--expected-videos",
            "200",
        ]
        run_logged(
            command,
            gpu_id=None,
            log_path=(
                args.output_dir
                / "orchestration_logs"
                / f"performance_aggregate.{attempt}.log"
            ),
            dry_run=args.dry_run,
        )
    elif not args.dry_run:
        print(f"resume: keeping {summary}")

    if not args.dry_run:
        status["phases"]["performance"] = {
            "status": "complete",
            "summary": str(summary),
            "calflops_profile": str(profile),
        }
        update_status(args.output_dir / "status.json", status)


def run_evaluation(args: argparse.Namespace, status: dict[str, Any]) -> None:
    command = [
        str(args.generation_python),
        str(SCRIPT_DIR / "evaluate_results_4gpu.py"),
        "--reference-videos",
        str(args.output_dir / "baseline" / "videos"),
        "--candidate-videos",
        str(args.output_dir / "teacache" / "videos"),
        "--output-dir",
        str(args.output_dir / "evaluation"),
        "--video-metrics-python",
        str(args.video_metrics_python),
        "--video-metrics-cache-dir",
        str(args.video_metrics_cache_dir),
        "--vbench-python",
        str(args.vbench_python),
        "--vbench-cache-dir",
        str(args.vbench_cache_dir),
        "--gpu-ids",
        *args.gpu_ids,
    ]
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    attempt = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_logged(
        command,
        gpu_id=None,
        log_path=(
            args.output_dir
            / "orchestration_logs"
            / f"evaluation.{attempt}.log"
        ),
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        status["phases"]["evaluation"] = {
            "status": "complete",
            "video_metrics": str(
                args.output_dir / "evaluation" / "video_metrics" / "summary.json"
            ),
            "vbench_reference": str(
                args.output_dir
                / "evaluation"
                / "vbench_reference"
                / "vbench200_aggregate_scores.json"
            ),
            "vbench_candidate": str(
                args.output_dir
                / "evaluation"
                / "vbench_candidate"
                / "vbench200_aggregate_scores.json"
            ),
        }
        update_status(args.output_dir / "status.json", status)


def run_final_report(args: argparse.Namespace, status: dict[str, Any]) -> None:
    report = args.output_dir / "benchmark_report.json"
    if args.resume and report.is_file():
        print(f"resume: keeping {report}")
    else:
        attempt = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_logged(
            [
                str(args.generation_python),
                str(SCRIPT_DIR / "build_final_report.py"),
                "--result-dir",
                str(args.output_dir),
            ],
            gpu_id=None,
            log_path=(
                args.output_dir
                / "orchestration_logs"
                / f"final_report.{attempt}.log"
            ),
            dry_run=args.dry_run,
        )
    if not args.dry_run:
        status["phases"]["final_report"] = {
            "status": "complete",
            "json": str(report),
            "markdown": str(args.output_dir / "benchmark_report.md"),
        }
        update_status(args.output_dir / "status.json", status)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--teacache-thresh",
        "--threshold",
        type=float,
        required=True,
        dest="teacache_thresh",
    )
    parser.add_argument("--use-ret-steps", action="store_true")
    parser.add_argument("--gpu-ids", nargs=4, default=["0", "1", "2", "3"])
    parser.add_argument("--generation-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--video-metrics-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--vbench-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--video-metrics-cache-dir", type=Path, default=DEFAULT_VIDEO_METRICS_CACHE
    )
    parser.add_argument("--vbench-cache-dir", type=Path, default=DEFAULT_VBENCH_CACHE)
    parser.add_argument("--wan21-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calflops-source", type=Path)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.teacache_thresh) or args.teacache_thresh < 0:
        raise ValueError("--teacache-thresh must be finite and non-negative")
    if len(set(args.gpu_ids)) != 4 or any(
        not gpu_id.isdigit() for gpu_id in args.gpu_ids
    ):
        raise ValueError("--gpu-ids must contain four distinct non-negative GPU IDs")
    args.output_dir = require_external(args.output_dir)
    args.generation_python = args.generation_python.expanduser().resolve(strict=True)
    args.video_metrics_python = args.video_metrics_python.expanduser().resolve(strict=True)
    args.vbench_python = args.vbench_python.expanduser().resolve(strict=True)
    args.video_metrics_cache_dir = (
        args.video_metrics_cache_dir.expanduser().resolve(strict=True)
    )
    args.vbench_cache_dir = args.vbench_cache_dir.expanduser().resolve(strict=True)
    args.wan21_root = args.wan21_root.expanduser().resolve(strict=True)
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve(strict=True)
    if args.calflops_source is not None:
        args.calflops_source = args.calflops_source.expanduser().resolve(strict=True)
    if "1.3B" not in str(args.checkpoint_dir):
        raise ValueError("the locked benchmark requires Wan2.1-T2V-1.3B")
    if not args.dry_run:
        check_python(args.generation_python, ["torch"], "generation")
        if args.calflops_source is None:
            check_python(args.generation_python, ["calflops"], "Calflops")
        if not args.skip_evaluation:
            check_python(
                args.video_metrics_python,
                ["torch", "lpips"],
                "VideoMetrics",
            )
            check_python(args.vbench_python, ["torch", "vbench"], "VBench")

    config = {
        "schema_version": 1,
        "benchmark": "TeaCache4Wan21 Vbench200 four-GPU comparison",
        "threshold": args.teacache_thresh,
        "threshold_slug": threshold_slug(args.teacache_thresh),
        "use_ret_steps": args.use_ret_steps,
        "gpu_ids": args.gpu_ids,
        "protocol": {
            "dataset": "Vbench200",
            "prompt_count": 200,
            "model": "Wan2.1-T2V-1.3B",
            "video": {"width": 832, "height": 480, "frames": 81, "fps": 16},
            "sampling": {
                "steps": 50,
                "solver": "unipc",
                "shift": 5.0,
                "cfg": 5.0,
                "seed": 42,
            },
            "precision": "DiT bfloat16",
            "memory": {"model_cpu_offload": False, "t5_cpu_offload": False},
        },
        "paths": {
            "wan21_root": str(args.wan21_root),
            "checkpoint_dir": str(args.checkpoint_dir),
            "generation_python": str(args.generation_python),
            "video_metrics_python": str(args.video_metrics_python),
            "vbench_python": str(args.vbench_python),
            "video_metrics_cache_dir": str(args.video_metrics_cache_dir),
            "vbench_cache_dir": str(args.vbench_cache_dir),
            "output_dir": str(args.output_dir),
            "calflops_source": (
                str(args.calflops_source) if args.calflops_source else None
            ),
        },
        "scripts": {
            path.name: sha256(path)
            for path in (
                SCRIPT_DIR / "run_vbench200_4gpu.py",
                SCRIPT_DIR / "generate_vbench200.py",
                SCRIPT_DIR / "profile_calflops.py",
                SCRIPT_DIR / "aggregate_performance.py",
                SCRIPT_DIR / "evaluate_results_4gpu.py",
                SCRIPT_DIR / "build_final_report.py",
            )
        },
        "evaluation_enabled": not args.skip_evaluation,
    }

    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
    else:
        if args.output_dir.exists() and not args.resume:
            raise FileExistsError(
                f"output exists; choose a new directory or pass --resume: {args.output_dir}"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "orchestration_logs").mkdir(exist_ok=True)
        write_locked_json(args.output_dir / "run_config.json", config, args.resume)
        readme = args.output_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "# TeaCache4Wan21 Vbench200 four-GPU result\n\n"
                "`baseline/` and `teacache/` contain generated videos and pure "
                "inference timing traces. `performance/` contains Calflops TFLOPs "
                "and latency aggregation. `evaluation/` contains repository "
                "PSNR/SSIM/LPIPS and Vbench200 subset scores. Process orchestration "
                "times include loading/saving and are never reported as inference "
                "latency.\n",
                encoding="utf-8",
            )
        link = PROJECT_DIR / "experiment_results" / args.output_dir.name
        if link.is_symlink():
            if link.resolve() != args.output_dir:
                raise FileExistsError(f"result link points elsewhere: {link}")
        elif link.exists():
            raise FileExistsError(f"result link path already exists: {link}")
        else:
            link.symlink_to(args.output_dir)

    status: dict[str, Any] = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "running",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phases": {},
    }
    if not args.dry_run:
        update_status(args.output_dir / "status.json", status)

    run_generation_condition(
        args,
        implementation="wan21",
        output_dir=args.output_dir / "baseline",
        status=status,
    )
    run_generation_condition(
        args,
        implementation="teacache",
        output_dir=args.output_dir / "teacache",
        status=status,
    )
    run_performance(args, status)
    if not args.skip_evaluation:
        run_evaluation(args, status)
        run_final_report(args, status)

    status["status"] = "dry_run" if args.dry_run else "complete"
    status["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    if not args.dry_run:
        update_status(args.output_dir / "status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
