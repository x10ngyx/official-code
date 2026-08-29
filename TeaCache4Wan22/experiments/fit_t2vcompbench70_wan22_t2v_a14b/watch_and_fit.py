#!/usr/bin/env python3
"""Wait for all prompt traces, then run the deterministic quartic fitter."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def is_complete(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "complete" and len(payload.get("records", [])) == 100


def main() -> None:
    args = parse_args()
    fitter = Path(__file__).resolve().parent / "fit_polynomials.py"
    while True:
        failures = sorted((args.result_root / "failures").glob("*.json"))
        if failures:
            raise RuntimeError(f"Calibration failure files detected: {[path.name for path in failures]}")
        complete = sum(is_complete(path) for path in (args.result_root / "samples").glob("*.json"))
        print(f"complete_samples={complete}/70", flush=True)
        if complete == 70:
            break
        if complete > 70:
            raise RuntimeError(f"Found {complete} completed samples; expected exactly 70")
        time.sleep(args.poll_seconds)

    command = [
        sys.executable,
        str(fitter),
        "--result-root",
        str(args.result_root),
        "--manifest",
        str(args.manifest),
    ]
    print("running_fitter=" + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=os.environ.copy())
    print("fit_complete", flush=True)


if __name__ == "__main__":
    main()
