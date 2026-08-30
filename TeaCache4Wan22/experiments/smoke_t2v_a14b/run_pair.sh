#!/usr/bin/env bash
set -euo pipefail

required=(WAN22_SOURCE WAN22_CKPT RESULT_ROOT CALFLOPS_SOURCE VBENCH_PYTHON)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done

experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$experiment_dir/../.." && pwd)
repository_dir=$(cd "$project_dir/.." && pwd)
python_bin=${WAN22_PYTHON:-python}
RESULT_ROOT=$(
  "$python_bin" - "$RESULT_ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).expanduser().resolve()
base = Path('/all/yiran07-disk3/huteng_data/exp').resolve()
try:
    root.relative_to(base)
except ValueError as error:
    raise SystemExit(f'RESULT_ROOT must be below {base}: {root}') from error
print(root)
PY
)
threshold=${SMOKE_THRESHOLD:-0.10}
prompt=${SMOKE_PROMPT:-Two anthropomorphic cats box on a spotlighted stage.}
baseline_id=smoke_baseline_seed42
teacache_id=smoke_teacache_seed42

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$RESULT_ROOT"
result_link="$project_dir/experiment_results/$(basename "$(readlink -f "$RESULT_ROOT")")"
mkdir -p "$project_dir/experiment_results"
if [[ -L $result_link ]]; then
  [[ $(readlink -f "$result_link") == $(readlink -f "$RESULT_ROOT") ]] || { echo "result symlink points elsewhere: $result_link" >&2; exit 2; }
elif [[ -e $result_link ]]; then
  echo "result index exists and is not a symlink: $result_link" >&2
  exit 2
else
  ln -s "$(readlink -f "$RESULT_ROOT")" "$result_link"
fi
if [[ ! -e "$RESULT_ROOT/README.md" ]]; then
  cp "$experiment_dir/RESULT_README.md" "$RESULT_ROOT/README.md"
fi

THRESHOLD=0 \
PROMPT="$prompt" \
RUN_ID="$baseline_id" \
bash "$project_dir/scripts/run_t2v_a14b.sh"

THRESHOLD="$threshold" \
PROMPT="$prompt" \
RUN_ID="$teacache_id" \
bash "$project_dir/scripts/run_t2v_a14b.sh"

mkdir -p "$RESULT_ROOT/flops"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
"$python_bin" "$project_dir/experiments/performance_t2v_a14b/profile_calflops.py" \
  --wan22-root "$WAN22_SOURCE" --checkpoint-dir "$WAN22_CKPT" \
  --calflops-source "$CALFLOPS_SOURCE" \
  --output "$RESULT_ROOT/flops/calflops_profile.json"

bash "$experiment_dir/finalize_pair.sh" \
  "$RESULT_ROOT" "$baseline_id" "$teacache_id" "$prompt"
