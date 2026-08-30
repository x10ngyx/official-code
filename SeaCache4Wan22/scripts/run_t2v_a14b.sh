#!/usr/bin/env bash
set -euo pipefail

required=(WAN22_SOURCE WAN22_CKPT RESULT_ROOT THRESHOLD PROMPT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
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
run_id=${RUN_ID:-single_seed42}
video_path="$RESULT_ROOT/${run_id}.mp4"
trace_path="$RESULT_ROOT/${run_id}.seacache.json"
timing_path="$RESULT_ROOT/${run_id}.timing.json"
log_path="$RESULT_ROOT/${run_id}.log"
manifest_path="$RESULT_ROOT/${run_id}.manifest.json"
use_ret_steps=${SEACACHE_USE_RET_STEPS:-0}

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cache_mode=$(
  "$python_bin" - "$THRESHOLD" <<'PY'
import math
import sys
value = float(sys.argv[1])
if not math.isfinite(value) or value < 0:
    raise ValueError("THRESHOLD must be finite and non-negative")
print("seacache" if value > 0 else "none")
PY
)
if [[ "$use_ret_steps" != 0 && "$use_ret_steps" != 1 ]]; then
  echo "SEACACHE_USE_RET_STEPS must be 0 or 1" >&2
  exit 2
fi
if [[ "$cache_mode" == none && "$use_ret_steps" == 1 ]]; then
  echo "SEACACHE_USE_RET_STEPS=1 requires THRESHOLD>0" >&2
  exit 2
fi

if [[ ! -f "$WAN22_SOURCE/.seacache4wan22_prepared.json" ]]; then
  echo "source was not prepared by SeaCache4Wan22: $WAN22_SOURCE" >&2
  exit 1
fi
if [[ ! -d "$WAN22_CKPT" ]]; then
  echo "missing Wan2.2 checkpoint directory: $WAN22_CKPT" >&2
  exit 1
fi
"$python_bin" "$script_dir/validate_prepared_tree.py" \
  --source "$WAN22_SOURCE" --mode prepared

run_files=("$video_path" "$timing_path" "$log_path" "$manifest_path")
if [[ "$cache_mode" == seacache ]]; then
  run_files+=("$trace_path")
fi
for path in "${run_files[@]}"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite existing run file: $path" >&2
    exit 1
  fi
done
mkdir -p "$RESULT_ROOT"

cache_args=(--timestep_cache none)
manifest_args=()
if [[ "$cache_mode" == seacache ]]; then
  cache_args=(
    --timestep_cache seacache
    --seacache_threshold "$THRESHOLD"
    --seacache_trace "$trace_path"
  )
  manifest_args=(--trace "$trace_path")
  if [[ "$use_ret_steps" == 1 ]]; then
    cache_args+=(--seacache_use_ret_steps)
    manifest_args+=(--use-ret-steps)
  fi
fi

command=(
  "$python_bin" "$WAN22_SOURCE/generate.py"
  --task t2v-A14B
  --size '832*480'
  --frame_num 45
  --ckpt_dir "$WAN22_CKPT"
  --offload_model true
  --sample_solver dpm++
  --sample_steps 50
  --sample_shift 12
  --base_seed 42
  --convert_model_dtype
  --timing_trace "$timing_path"
  --prompt "$PROMPT"
  --save_file "$video_path"
  "${cache_args[@]}"
)

printf 'command=' > "$log_path"
printf '%q ' "${command[@]}" >> "$log_path"
printf '\n' >> "$log_path"
"${command[@]}" 2>&1 | tee -a "$log_path"

"$python_bin" "$script_dir/write_run_manifest.py" \
  --output "$manifest_path" \
  --source "$WAN22_SOURCE" \
  --checkpoint "$WAN22_CKPT" \
  --threshold "$THRESHOLD" \
  --prompt "$PROMPT" \
  --video "$video_path" \
  --timing "$timing_path" \
  --log "$log_path" \
  "${manifest_args[@]}"

echo "run complete: $manifest_path"
