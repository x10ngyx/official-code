#!/usr/bin/env python3
"""Delegate to the package's single locked Wan2.2 Calflops profiler."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


for variable in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
):
    os.environ[variable] = "1"

TARGET = Path(__file__).resolve().parents[1] / "performance_t2v_a14b" / "profile_calflops.py"

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
