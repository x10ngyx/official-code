#!/usr/bin/env python3
"""Run the VBench10 no-offload low-speed TeaCache threshold extension."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
RUNNER = PROJECT_DIR / "experiments" / "threshold_scan_vbench10_full_metrics" / "run_scan.py"
DEFAULT_PYTHON = Path(
    "/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python"
)
DEFAULT_OUTPUT = Path(
    "/mnt/hdd/xiongyuxiang/tmp/exp/teacache_wan21_vbench10_low_speed_no_offload_scan"
)
DEFAULT_ANCHOR = Path(
    "/mnt/hdd/xiongyuxiang/tmp/exp/teacache_wan21_threshold_0p15_no_offload_vbench10_4gpu"
)
DEFAULT_THRESHOLDS = [0.06, 0.07, 0.075, 0.08, 0.085, 0.09, 0.10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--anchor-root", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpus", nargs=4, default=["0", "1", "2", "3"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_anchor_link(output_root: Path, name: str, target: Path) -> None:
    if not target.exists():
        raise FileNotFoundError(target)
    link = output_root / name
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise ValueError(f"existing link points elsewhere: {link}")
        return
    if link.exists():
        raise FileExistsError(f"expected a symlink or absent path: {link}")
    os.symlink(target.resolve(), link, target_is_directory=True)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    anchor_root = args.anchor_root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    if args.thresholds != sorted(args.thresholds) or len(set(args.thresholds)) != len(
        args.thresholds
    ):
        raise ValueError("thresholds must be unique and sorted")
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_anchor_link(output_root, "baseline", anchor_root / "baseline")
    ensure_anchor_link(output_root, "profiling", anchor_root / "profiling")

    command = [
        str(python),
        str(RUNNER),
        "--output-root",
        str(output_root),
        "--thresholds",
        *(str(value) for value in args.thresholds),
        "--gpus",
        *args.gpus,
        "--phases",
        "profile",
        "generate",
        "metrics",
        "analyze",
        "--allow-low-thresholds",
        "--resume",
    ]
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


if __name__ == "__main__":
    main()
