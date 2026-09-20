#!/usr/bin/env python3
"""Independent on-policy CNN+G1 ablation; no IQL checkpoint is accepted."""
import runpy
import sys
from pathlib import Path

experiment = Path(__file__).resolve().parent / 'experiments/reinforce_v1'
sys.path.insert(0, str(experiment))
if __name__ == '__main__':
    runpy.run_path(str(experiment / 'run.py'), run_name='__main__')
