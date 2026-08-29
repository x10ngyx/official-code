#!/usr/bin/env python3
"""Run repository VideoMetrics and four-GPU VBench evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent
VIDEO_METRICS_DIR = REPOSITORY_DIR / "VideoMetrics"
VBENCH_DIR = REPOSITORY_DIR / "VbenchEvaluation"
EXP_ROOT = Path("/mnt/hdd/xiongyuxiang/tmp/exp").resolve()


def available_log_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return path.with_name(f"{path.stem}.{stamp}{path.suffix}")


def require_external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as exc:
        raise ValueError(f"path must be below {EXP_ROOT}: {resolved}") from exc
    return resolved


def check_python(path: Path, modules: list[str], label: str) -> None:
    resolved = path.expanduser().resolve(strict=True)
    expression = "; ".join(f"import {module}" for module in modules)
    result = subprocess.run(
        [str(resolved), "-c", expression],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} Python is missing required modules {modules}: {resolved}\n"
            f"{result.stderr.strip()}"
        )


def run_logged(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
    if dry_run:
        print(shlex.join(command))
        return
    log_path = available_log_path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as handle:
        handle.write(f"command: {shlex.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def vbench_environment(cache_dir: Path, gpu_id: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "VBENCH_CACHE_DIR": str(cache_dir),
            "TORCH_HOME": str(cache_dir / "torch"),
            "HF_HOME": str(cache_dir / "huggingface"),
            "XDG_CACHE_HOME": str(cache_dir / "xdg"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    return env


def run_video_metrics(args: argparse.Namespace) -> None:
    summary = args.output_dir / "video_metrics" / "summary.json"
    if args.resume and summary.is_file():
        print(f"resume: keeping {summary}")
        return
    command = [
        str(args.video_metrics_python),
        str(VIDEO_METRICS_DIR / "evaluate.py"),
        "--reference-dir",
        str(args.reference_videos),
        "--candidate-dir",
        str(args.candidate_videos),
        "--extension",
        ".mp4",
        "--expected-frames",
        "81",
        "--device",
        "cuda:0",
        "--model-cache",
        str(args.video_metrics_cache_dir),
        "--output-dir",
        str(args.output_dir / "video_metrics"),
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu_ids[0],
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    run_logged(
        command,
        env=env,
        log_path=args.output_dir / "logs" / "video_metrics.log",
        dry_run=args.dry_run,
    )


def load_dimensions() -> list[str]:
    payload = json.loads((VBENCH_DIR / "dimensions.json").read_text(encoding="utf-8"))
    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, list) or len(dimensions) != 16:
        raise ValueError("repository VBench dimension lock must contain 16 dimensions")
    return [str(item) for item in dimensions]


def stage_videos(
    *,
    args: argparse.Namespace,
    videos_dir: Path,
    work_dir: Path,
    label: str,
) -> None:
    command = [
        str(args.vbench_python),
        str(VBENCH_DIR / "prepare_videos.py"),
        "--videos-dir",
        str(videos_dir),
        "--staging-dir",
        str(work_dir / "staged_videos"),
        "--manifest",
        str(work_dir / "staging_manifest.json"),
        "--expected-seeds",
        "1",
    ]
    run_logged(
        command,
        env=vbench_environment(args.vbench_cache_dir),
        log_path=args.output_dir / "logs" / f"vbench_{label}_stage.log",
        dry_run=args.dry_run,
    )


def run_vbench_condition(
    *,
    args: argparse.Namespace,
    videos_dir: Path,
    work_dir: Path,
    label: str,
) -> None:
    aggregate = work_dir / "vbench200_aggregate_scores.json"
    if args.resume and aggregate.is_file():
        print(f"resume: keeping {aggregate}")
        return
    stage_videos(args=args, videos_dir=videos_dir, work_dir=work_dir, label=label)
    dimensions = load_dimensions()
    groups = [dimensions[index::4] for index in range(4)]
    processes: list[tuple[subprocess.Popen[str], Any, Path, list[str]]] = []
    for shard_index, (gpu_id, group) in enumerate(zip(args.gpu_ids, groups)):
        shard_output = work_dir / "scores" / f"shard_{shard_index}"
        marker = shard_output / "COMPLETE.json"
        if args.resume and marker.is_file():
            print(f"resume: keeping {marker}")
            continue
        command = [
            str(args.vbench_python),
            str(VBENCH_DIR / "evaluate_vbench.py"),
            "--videos-dir",
            str(work_dir / "staged_videos"),
            "--output-dir",
            str(shard_output),
            "--dimensions",
            *group,
            "--name-prefix",
            f"vbench200_{label}_shard_{shard_index}",
            "--device",
            "cuda:0",
            "--load-ckpt-from-local",
        ]
        if args.dry_run:
            print(f"CUDA_VISIBLE_DEVICES={gpu_id} {shlex.join(command)}")
            continue
        shard_output.mkdir(parents=True, exist_ok=True)
        log_path = available_log_path(
            args.output_dir / "logs" / f"vbench_{label}_shard_{shard_index}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("x", encoding="utf-8")
        handle.write(f"command: {shlex.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            env=vbench_environment(args.vbench_cache_dir, gpu_id),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((process, handle, marker, group))

    failures: list[int] = []
    for process, handle, marker, group in processes:
        returncode = process.wait()
        handle.close()
        if returncode:
            failures.append(returncode)
        else:
            marker.write_text(
                json.dumps({"status": "complete", "dimensions": group}, indent=2)
                + "\n",
                encoding="utf-8",
            )
    if failures:
        raise RuntimeError(f"VBench {label} worker failures: {failures}")

    command = [
        str(args.vbench_python),
        str(VBENCH_DIR / "aggregate_vbench_scores.py"),
        "--score-dir",
        str(work_dir / "scores"),
        "--output",
        str(aggregate),
        "--label",
        f"Vbench200 {label}",
    ]
    run_logged(
        command,
        env=vbench_environment(args.vbench_cache_dir),
        log_path=args.output_dir / "logs" / f"vbench_{label}_aggregate.log",
        dry_run=args.dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-videos", type=Path, required=True)
    parser.add_argument("--candidate-videos", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-metrics-python", type=Path, required=True)
    parser.add_argument("--video-metrics-cache-dir", type=Path, required=True)
    parser.add_argument("--vbench-python", type=Path, required=True)
    parser.add_argument("--vbench-cache-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", nargs=4, default=["0", "1", "2", "3"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reference_videos = require_external(args.reference_videos)
    args.candidate_videos = require_external(args.candidate_videos)
    args.output_dir = require_external(args.output_dir)
    args.video_metrics_python = args.video_metrics_python.expanduser().resolve()
    args.video_metrics_cache_dir = (
        args.video_metrics_cache_dir.expanduser().resolve(strict=True)
    )
    args.vbench_python = args.vbench_python.expanduser().resolve()
    args.vbench_cache_dir = args.vbench_cache_dir.expanduser().resolve(strict=True)
    if len(set(args.gpu_ids)) != 4 or any(
        not gpu_id.isdigit() for gpu_id in args.gpu_ids
    ):
        raise ValueError("--gpu-ids must contain four distinct non-negative GPU IDs")
    if not args.dry_run:
        if not args.reference_videos.is_dir() or not args.candidate_videos.is_dir():
            raise NotADirectoryError("reference and candidate video directories must exist")
        reference_names = sorted(path.name for path in args.reference_videos.glob("*.mp4"))
        candidate_names = sorted(path.name for path in args.candidate_videos.glob("*.mp4"))
        if reference_names != candidate_names or len(reference_names) != 200:
            raise ValueError(
                "evaluation requires exactly 200 matched baseline/TeaCache MP4 files"
            )
        check_python(args.video_metrics_python, ["torch", "lpips"], "VideoMetrics")
        check_python(args.vbench_python, ["torch", "vbench"], "VBench")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        readme = args.output_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "# TeaCache4Wan21 Vbench200 evaluation\n\n"
                "`video_metrics/` contains the repository PSNR/SSIM/LPIPS result. "
                "`vbench_reference/` and `vbench_candidate/` contain official "
                "16-dimension Vbench200 subset scores. VBench dimensions run in "
                "four static GPU shards.\n",
                encoding="utf-8",
            )

    run_video_metrics(args)
    run_vbench_condition(
        args=args,
        videos_dir=args.reference_videos,
        work_dir=args.output_dir / "vbench_reference",
        label="reference",
    )
    run_vbench_condition(
        args=args,
        videos_dir=args.candidate_videos,
        work_dir=args.output_dir / "vbench_candidate",
        label="candidate",
    )
    print(
        json.dumps(
            {
                "status": "dry_run" if args.dry_run else "complete",
                "video_metrics": str(args.output_dir / "video_metrics" / "summary.json"),
                "vbench_reference": str(
                    args.output_dir
                    / "vbench_reference"
                    / "vbench200_aggregate_scores.json"
                ),
                "vbench_candidate": str(
                    args.output_dir
                    / "vbench_candidate"
                    / "vbench200_aggregate_scores.json"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
