#!/usr/bin/env bash
set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DESTINATION" >&2
  echo "The destination must not already exist." >&2
  exit 2
fi

destination=$1
if [[ -e "$destination" ]]; then
  echo "refusing to overwrite existing destination: $destination" >&2
  exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
repository=${WAN22_REPOSITORY:-https://github.com/Wan-Video/Wan2.2.git}
commit=42bf4cfaa384bc21833865abc2f9e6c0e67233dc
python_bin=${WAN22_SETUP_PYTHON:-python3}

mkdir -p "$(dirname "$destination")"
clone_args=(--no-checkout)
if [[ "$repository" == http://* || "$repository" == https://* ]]; then
  clone_args+=(--filter=blob:none)
fi
git clone "${clone_args[@]}" "$repository" "$destination"
if ! git -C "$destination" cat-file -e "${commit}^{commit}" 2>/dev/null; then
  git -C "$destination" fetch --depth 1 origin "$commit"
fi
git -C "$destination" checkout --detach "$commit"

"$python_bin" "$script_dir/validate_prepared_tree.py" \
  --source "$destination" \
  --mode upstream

git -C "$destination" apply "$project_dir/patches/wan22_42bf4cf_teacache.patch"
install -m 0644 "$project_dir/runtime/teacache.py" "$destination/wan/teacache.py"
install -m 0644 "$project_dir/runtime/inference_timing.py" "$destination/wan/inference_timing.py"

"$python_bin" "$script_dir/validate_prepared_tree.py" \
  --source "$destination" \
  --mode prepared \
  --write-manifest

echo "Prepared TeaCache4Wan22 source: $destination"
