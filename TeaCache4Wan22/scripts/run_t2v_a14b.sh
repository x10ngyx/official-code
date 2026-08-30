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
project_dir=$(cd "$script_dir/.." && pwd)
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
trace_path="$RESULT_ROOT/${run_id}.teacache.json"
timing_path="$RESULT_ROOT/${run_id}.timing.json"
log_path="$RESULT_ROOT/${run_id}.log"
manifest_path="$RESULT_ROOT/${run_id}.manifest.json"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

threshold_mode=$(
  "$python_bin" - "$THRESHOLD" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or value < 0:
    raise ValueError("THRESHOLD must be finite and non-negative")
print("teacache" if value > 0 else "none")
PY
)

if [[ ! -f "$WAN22_SOURCE/.teacache4wan22_prepared.json" ]]; then
  echo "source was not prepared by TeaCache4Wan22: $WAN22_SOURCE" >&2
  exit 1
fi
if [[ ! -d "$WAN22_CKPT" ]]; then
  echo "missing Wan2.2 checkpoint directory: $WAN22_CKPT" >&2
  exit 1
fi
"$python_bin" "$script_dir/validate_prepared_tree.py" \
  --source "$WAN22_SOURCE" \
  --mode prepared
if [[ -e "$video_path" || -e "$trace_path" || -e "$timing_path" || -e "$log_path" || -e "$manifest_path" ]]; then
  echo "refusing to overwrite existing run files for RUN_ID=$run_id" >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT"
cache_args=()
manifest_args=()
if [[ "$threshold_mode" == teacache ]]; then
  coefficients=${TEACACHE_COEFFICIENTS:-$project_dir/coefficients/wan22_t2v_a14b_50step_dpmpp_nonretention.json}
  if [[ ! -f "$coefficients" ]]; then
    echo "missing TeaCache coefficients: $coefficients" >&2
    exit 1
  fi
  cache_args=(
    --timestep_cache teacache
    --teacache_threshold "$THRESHOLD"
    --teacache_coefficients "$coefficients"
    --teacache_trace "$trace_path"
  )
  manifest_args=(--coefficients "$coefficients" --trace "$trace_path")
else
  cache_args=(--timestep_cache none)
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
