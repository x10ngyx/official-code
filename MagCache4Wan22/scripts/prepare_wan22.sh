#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
[[ $# == 1 ]] || { echo "usage: $0 DESTINATION" >&2; exit 2; }
destination=$1
[[ ! -e "$destination" ]] || { echo "destination already exists" >&2; exit 1; }
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${WAN22_PYTHON:-python}
repository=${WAN22_REPOSITORY:-https://github.com/Wan-Video/Wan2.2.git}
commit=42bf4cfaa384bc21833865abc2f9e6c0e67233dc
git clone --no-checkout "$repository" "$destination"
git -C "$destination" checkout --detach "$commit"
"$python_bin" "$script_dir/validate_source.py" --source "$destination" --write-manifest
