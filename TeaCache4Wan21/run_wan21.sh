#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 WAN21_ROOT WAN21_ARGUMENTS..." >&2
  exit 2
fi

wan21_root=$(readlink -f "$1")
shift
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
expected_commit=65386b2e03c490796eede31b0325a6a595cc684e
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

if [[ ! -f "$wan21_root/wan/__init__.py" || ! -f "$wan21_root/generate.py" ]]; then
  echo "not a Wan2.1 source tree: $wan21_root" >&2
  exit 1
fi
if [[ ! -e "$wan21_root/.git" ]]; then
  echo "Wan2.1 source must be a Git checkout pinned to $expected_commit" >&2
  exit 1
fi

actual_commit=$(git -C "$wan21_root" rev-parse HEAD)
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "Wan2.1 commit mismatch: expected $expected_commit, got $actual_commit" >&2
  exit 1
fi
if [[ -n "$(git -C "$wan21_root" status --porcelain --untracked-files=no)" ]]; then
  echo "Wan2.1 checkout has tracked modifications; exact reproduction requires a clean tree" >&2
  exit 1
fi

cd "$wan21_root"
export PYTHONPATH="$wan21_root${PYTHONPATH:+:$PYTHONPATH}"
export WAN21_ROOT="$wan21_root"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
exec "${python_cmd[@]}" "$project_dir/generate.py" "$@"
