#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 WAN21_ROOT [SeaCache4Wan21/generate.py arguments...]" >&2
  exit 2
fi

wan21_root=$1
shift
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${WAN21_PYTHON:-python}

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "$python_bin" "$script_dir/generate.py" --wan21_root "$wan21_root" "$@"
