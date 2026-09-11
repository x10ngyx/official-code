#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${WAN22_PYTHON:-}" ]]; then
  exec "$WAN22_PYTHON" -m unittest discover -s "$project_dir/tests" -v
fi
exec conda run --no-capture-output -n wan2.2 python -m unittest discover -s "$project_dir/tests" -v
