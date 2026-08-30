#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 VIDEOS_DIR WORK_DIR [PROMPT_MAP_JSON]" >&2
  exit 2
fi

videos_dir=$1
work_dir=$2
prompt_map=${3:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "$script_dir/../../.." && pwd)
python_bin=${PYTHON_BIN:-python}

export VBENCH_CACHE_DIR=${VBENCH_CACHE_DIR:-"$repository_root/models/VBench"}
export TORCH_HOME=${TORCH_HOME:-"$VBENCH_CACHE_DIR/torch"}
export HF_HOME=${HF_HOME:-"$VBENCH_CACHE_DIR/huggingface"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$VBENCH_CACHE_DIR/xdg"}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -e $work_dir ]]; then
  echo "refusing to overwrite custom VBench work directory: $work_dir" >&2
  exit 1
fi
evaluate_args=(
  --videos-dir "$videos_dir"
  --output-dir "$work_dir/scores"
)
if [[ -n $prompt_map ]]; then
  evaluate_args+=(--prompt-map "$prompt_map")
fi
"$python_bin" "$script_dir/evaluate_custom_vbench.py" "${evaluate_args[@]}"
"$python_bin" "$script_dir/aggregate_custom_vbench_scores.py" \
  --score-dir "$work_dir/scores" \
  --output "$work_dir/vbench_custom_aggregate_scores.json"
