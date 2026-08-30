#!/usr/bin/env python3
"""Run a reproducible four-GPU TeaCache threshold calibration."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = PROJECT_DIR.parents[2]
GENERATOR = PROJECT_DIR / "experiments" / "vbench200_t2v" / "generate_vbench200.py"
EXP_ROOT = Path("/all/yiran07-disk3/huteng_data/exp").resolve()
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "models" / "Wan2.1-T2V-1.3B"
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
DEFAULT_SAMPLE_IDS = [
    "vbench200_001",
    "vbench200_026",
    "vbench200_051",
    "vbench200_056",
    "vbench200_076",
    "vbench200_085",
    "vbench200_101",
    "vbench200_126",
    "vbench200_151",
    "vbench200_176",
    "vbench200_177",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="*", default=[])
    parser.add_argument("--sample-ids", nargs="+", default=DEFAULT_SAMPLE_IDS)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wan21-root", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def require_external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as exc:
        raise ValueError(f"output must be below {EXP_ROOT}: {resolved}") from exc
    return resolved


def threshold_label(threshold: float) -> str:
    return f"threshold_{threshold:.4f}".replace(".", "p")


def build_gpu_mapping(sample_ids: list[str], gpus: list[str]) -> dict[str, str]:
    if not gpus:
        raise ValueError("GPU IDs must be non-empty")
    return {
        sample_id: gpus[index % len(gpus)]
        for index, sample_id in enumerate(sample_ids)
    }


def write_once_or_match(path: Path, payload: dict[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"existing protocol differs: {path}")
        return
    path.write_text(rendered, encoding="utf-8")


def run_condition(
    args: argparse.Namespace,
    *,
    output_root: Path,
    label: str,
    threshold: float | None,
    num_shards: int,
) -> None:
    condition_dir = output_root / label
    if condition_dir.exists() and not args.resume:
        raise FileExistsError(f"condition exists; use --resume: {condition_dir}")
    condition_dir.mkdir(parents=True, exist_ok=True)
    condition = {
        "schema_version": 1,
        "label": label,
        "implementation": "wan21" if threshold is None else "teacache",
        "threshold": threshold,
        "use_ret_steps": False,
    }
    write_once_or_match(condition_dir / "condition.json", condition)

    processes: list[tuple[int, subprocess.Popen[str], object, Path]] = []
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for shard_index in range(num_shards):
        command = [
            sys.executable,
            str(GENERATOR),
            "--implementation",
            "wan21" if threshold is None else "teacache",
            "--task",
            "t2v-1.3B",
            "--wan21-root",
            str(args.wan21_root.resolve()),
            "--ckpt-dir",
            str(args.ckpt_dir.resolve()),
            "--output-dir",
            str(condition_dir),
            "--seeds",
            str(args.seed),
            "--size",
            "832*480",
            "--frame-num",
            "81",
            "--sample-steps",
            "50",
            "--sample-shift",
            "5.0",
            "--guide-scale",
            "5.0",
            "--sample-solver",
            "unipc",
            "--sample-ids",
            *args.sample_ids,
            "--shard-index",
            str(shard_index),
            "--num-shards",
            str(num_shards),
        ]
        if threshold is not None:
            command.extend(("--teacache-thresh", str(threshold)))
        if args.resume:
            command.append("--resume")
        env = os.environ.copy()
        env.update(THREAD_ENV)
        env["CUDA_VISIBLE_DEVICES"] = args.gpus[shard_index]
        log_path = condition_dir / f"controller.shard_{shard_index:03d}.{timestamp}.log"
        log_handle = log_path.open("x", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=SCRIPT_DIR,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((shard_index, process, log_handle, log_path))

    failures: list[str] = []
    for shard_index, process, log_handle, log_path in processes:
        returncode = process.wait()
        log_handle.close()
        if returncode != 0:
            failures.append(f"shard {shard_index}: returncode={returncode}, log={log_path}")
    if failures:
        raise RuntimeError(f"condition {label} failed: {'; '.join(failures)}")


def main() -> None:
    args = parse_args()
    args.output_root = require_external(args.output_root)
    if args.seed != 42:
        raise ValueError("the fixed Wan2.1 protocol requires seed 42")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        raise ValueError("GPU IDs must be non-empty and unique")
    if len(set(args.sample_ids)) != len(args.sample_ids):
        raise ValueError("sample ids must be unique")
    if len(set(args.thresholds)) != len(args.thresholds):
        raise ValueError("thresholds must be unique")
    if any(value <= 0 for value in args.thresholds):
        raise ValueError("calibration thresholds must be positive")
    if not args.wan21_root.is_dir() or not args.ckpt_dir.is_dir():
        raise FileNotFoundError("Wan2.1 source or checkpoint directory is missing")

    args.output_root.mkdir(parents=True, exist_ok=True)
    link = PROJECT_DIR / "experiment_results" / args.output_root.name
    if link.is_symlink():
        if link.resolve(strict=True) != args.output_root:
            raise ValueError(f"result symlink points elsewhere: {link}")
    elif link.exists():
        raise FileExistsError(f"result index exists and is not a symlink: {link}")
    else:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(args.output_root, target_is_directory=True)
    protocol = {
        "schema_version": 1,
        "task": "t2v-1.3B",
        "size": "832*480",
        "frame_num": 81,
        "sample_steps": 50,
        "sample_solver": "unipc",
        "sample_shift": 5.0,
        "guide_scale": 5.0,
        "seed": args.seed,
        "sample_ids": args.sample_ids,
        "gpu_mapping": build_gpu_mapping(args.sample_ids, args.gpus),
        "use_ret_steps": False,
        "latency_metric": "pipeline_generate_wall_seconds",
        "speedup_aggregation": "sum(baseline latency) / sum(candidate latency)",
    }
    write_once_or_match(args.output_root / "calibration_protocol.json", protocol)
    readme = args.output_root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# TeaCache Wan2.1 threshold calibration result\n\n"
            "This external result contains a fixed VBench200 prompt subset, per-condition "
            "videos/logs/timing traces, and aggregate inference-latency analysis. "
            "`pipeline_generate_wall_seconds` excludes model loading and MP4 export.\n",
            encoding="utf-8",
        )

    num_shards = len(args.gpus)
    if not args.skip_baseline:
        run_condition(
            args,
            output_root=args.output_root,
            label="baseline",
            threshold=None,
            num_shards=num_shards,
        )
    for threshold in args.thresholds:
        run_condition(
            args,
            output_root=args.output_root,
            label=threshold_label(threshold),
            threshold=threshold,
            num_shards=num_shards,
        )


if __name__ == "__main__":
    main()
