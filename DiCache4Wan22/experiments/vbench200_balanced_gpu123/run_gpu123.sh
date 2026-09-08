#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${WAN22_PYTHON:-/home/huteng/yes/envs/wan2.2/bin/python}" "$script_dir/rebalance_gpu123.py" "$@"
