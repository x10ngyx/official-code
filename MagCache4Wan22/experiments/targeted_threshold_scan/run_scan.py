#!/usr/bin/env python3
"""Fixed R/K per target, measured E calibration with the persistent batch runner."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime"))
from suite import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not any(arg == "--target-presets" or arg.startswith("--target-presets=") for arg in argv):
        argv = ["--target-presets", str(Path(__file__).with_name("presets.json"))] + argv
    main(scan=True, argv=argv)
