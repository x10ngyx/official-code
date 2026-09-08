#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${WAN22_PYTHON:-python}
"$python_bin" -c 'import sys; from pathlib import Path; assert Path(sys.prefix).name == "wan2.2", sys.prefix'
exec "$python_bin" -m unittest discover -s "$project_dir/tests" -v
