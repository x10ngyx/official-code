#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 WAN21_ROOT [SeaCache4Wan21/generate.py arguments...]" >&2
  exit 2
fi

wan21_root=$1
shift
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -n ${WAN22_PYTHON:-} ]]; then
  python_cmd=("$WAN22_PYTHON")
else
  conda_bin=${CONDA_BIN:-$(command -v conda || true)}
  if [[ -z $conda_bin ]]; then
    echo "conda is unavailable; set WAN22_PYTHON to the wan2.2 environment Python" >&2
    exit 2
  fi
  python_cmd=("$conda_bin" run --no-capture-output -n wan2.2 python)
fi

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "${python_cmd[@]}" "$script_dir/generate.py" --wan21_root "$wan21_root" "$@"
