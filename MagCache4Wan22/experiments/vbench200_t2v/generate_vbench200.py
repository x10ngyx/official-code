#!/usr/bin/env python3
"""One persistent model-loading process for a round-robin prompt shard."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime"))
from common import check_environment, external_output
from batch import worker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    check_environment()
    worker(external_output(args.output_dir), args.worker_index, args.resume)
