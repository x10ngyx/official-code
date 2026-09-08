#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
exec /home/huteng/yes/envs/wan2.2/bin/python "$(dirname "$0")/rebalance_gpu123.py" "$@"
