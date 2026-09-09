#!/usr/bin/env python3
"""Wan21 online fine-tuning CLI; set thread limits before importing torch/NumPy."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from ours4wan21.online_pipeline import main
if __name__ == '__main__':
    main()
