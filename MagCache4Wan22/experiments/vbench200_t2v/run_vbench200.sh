#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${WAN22_PYTHON:-python}
exec "$python_bin" "$script_dir/run_vbench200.py" "$@"
