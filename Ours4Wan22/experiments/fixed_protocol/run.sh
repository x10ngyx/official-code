#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${WAN22_PYTHON:-python}" "$script_dir/../../main.py" "$@"
