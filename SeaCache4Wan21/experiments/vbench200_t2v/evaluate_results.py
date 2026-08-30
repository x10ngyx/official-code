#!/usr/bin/env python3
"""Run the repository-standard VideoMetrics and Vbench200 evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = SCRIPT_DIR.parents[2]
VIDEO_METRICS = REPOSITORY_DIR / "VideoMetrics"
VBENCH = REPOSITORY_DIR / "VbenchEvaluation"
EXP_ROOT = Path("/all/yiran07-disk3/huteng_data/exp").resolve()
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}


def external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as exc:
        raise ValueError(f"output must be below {EXP_ROOT}: {resolved}") from exc
    return resolved


def run(command: list[str], env: dict[str, str], log: Path, dry_run: bool) -> None:
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
        subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-videos", type=Path, required=True)
    parser.add_argument("--candidate-videos", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--video-metrics-python", type=Path, required=True)
    parser.add_argument("--vbench-python", type=Path, required=True)
    parser.add_argument("--video-metrics-cache-dir", type=Path, required=True)
    parser.add_argument("--vbench-cache-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.reference_videos = args.reference_videos.expanduser().resolve(strict=not args.dry_run)
    args.candidate_videos = args.candidate_videos.expanduser().resolve(strict=not args.dry_run)
    args.output_dir = external(args.output_dir)
    args.video_metrics_python = args.video_metrics_python.expanduser().resolve(strict=True)
    args.vbench_python = args.vbench_python.expanduser().resolve(strict=True)
    args.video_metrics_cache_dir = args.video_metrics_cache_dir.expanduser().resolve(strict=True)
    args.vbench_cache_dir = args.vbench_cache_dir.expanduser().resolve(strict=True)
    if args.expected_frames not in {45, 81}:
        raise ValueError("expected frames must be 45 or 81")
    if not args.gpu_id.isdigit():
        raise ValueError("--gpu-id must be a non-negative integer")
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    base_env = {**os.environ, **THREAD_ENV, "CUDA_VISIBLE_DEVICES": args.gpu_id}
    metrics_summary = args.output_dir / "video_metrics" / "summary.json"
    if not (args.resume and metrics_summary.is_file()):
        metrics_env = dict(base_env)
        metrics_env["PATH"] = str(args.video_metrics_python.parent) + os.pathsep + metrics_env.get("PATH", "")
        metrics_env["TORCH_HOME"] = str(args.video_metrics_cache_dir)
        run([
            "bash", str(VIDEO_METRICS / "run_evaluation.sh"),
            "--reference-dir", str(args.reference_videos),
            "--candidate-dir", str(args.candidate_videos), "--extension", ".mp4",
            "--expected-frames", str(args.expected_frames), "--device", "cuda:0",
            "--model-cache", str(args.video_metrics_cache_dir),
            "--output-dir", str(args.output_dir / "video_metrics"),
        ], metrics_env, args.output_dir / "logs" / "video_metrics.log", args.dry_run)

    vbench_env = {
        **base_env, "PYTHON_BIN": str(args.vbench_python),
        "VBENCH_CACHE_DIR": str(args.vbench_cache_dir),
        "TORCH_HOME": str(args.vbench_cache_dir / "torch"),
        "HF_HOME": str(args.vbench_cache_dir / "huggingface"),
        "XDG_CACHE_HOME": str(args.vbench_cache_dir / "xdg"),
    }
    for label, videos in (("reference", args.reference_videos), ("candidate", args.candidate_videos)):
        target = args.output_dir / f"vbench_{label}"
        aggregate = target / "vbench200_aggregate_scores.json"
        if args.resume and aggregate.is_file():
            continue
        run(
            ["bash", str(VBENCH / "run_vbench200.sh"), str(videos), str(target), "1"],
            vbench_env, args.output_dir / "logs" / f"vbench_{label}.log", args.dry_run,
        )
    if not args.dry_run:
        (args.output_dir / "README.md").write_text(
            "# Vbench200 evaluation artifacts\n\n"
            "`video_metrics/` is produced by the repository VideoMetrics fixed "
            "protocol. `vbench_reference/` and `vbench_candidate/` are produced "
            "by VbenchEvaluation and are Vbench200 subset scores.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
