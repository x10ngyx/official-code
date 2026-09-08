#!/usr/bin/env python3
"""Compatibility entry for the persistent paired batch runner."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime"))
from prompts import load_prompts
from suite import main

if __name__ == "__main__":
    main()
