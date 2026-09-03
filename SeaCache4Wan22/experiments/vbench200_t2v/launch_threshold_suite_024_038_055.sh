#!/usr/bin/env bash
set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ $# -ne 1 ]]; then
  echo "usage: $0 SUITE_ROOT" >&2
  exit 2
fi

suite_root=$1
experiment_root=/all/yiran07-disk3/huteng_data/exp
case "$suite_root/" in
  "$experiment_root"/*/) ;;
  *)
    echo "SUITE_ROOT must be below $experiment_root: $suite_root" >&2
    exit 2
    ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/../.." && pwd)
workspace_root=$(cd "$project_dir/../../.." && pwd)
if [[ -n ${WAN22_PYTHON:-} ]]; then
  python_bin=$WAN22_PYTHON
else
  conda_bin=${CONDA_BIN:-$(command -v conda || true)}
  if [[ -z $conda_bin ]]; then
    echo "conda is unavailable; set WAN22_PYTHON to the wan2.2 environment Python" >&2
    exit 2
  fi
  python_bin=$("$conda_bin" run --no-capture-output -n wan2.2 python -c 'import sys; print(sys.executable)')
fi
vbench_python=${VBENCH_PYTHON:-"$project_dir/build/vbench-eval/bin/python"}
wan22_root=${WAN22_SOURCE:-"$project_dir/build/Wan2.2-42bf4cf-prepared"}
checkpoint=${WAN22_CKPT:-"$workspace_root/models/Wan2.2-T2V-A14B"}
baseline_source=${WAN22_BASELINE_SOURCE:-}
start_threshold=${WAN22_START_THRESHOLD:-0.24}
suite_name=$(basename "$suite_root")

case "$start_threshold" in
  0.24|0.38|0.55) ;;
  *)
    echo "WAN22_START_THRESHOLD must be 0.24, 0.38, or 0.55: $start_threshold" >&2
    exit 2
    ;;
esac
if [[ "$start_threshold" != 0.24 && -z "$baseline_source" ]]; then
  echo "WAN22_BASELINE_SOURCE is required when starting after threshold 0.24" >&2
  exit 2
fi

for path in "$python_bin" "$vbench_python" "$wan22_root/.seacache4wan22_prepared.json"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
done
if [[ ! -d "$checkpoint" ]]; then
  echo "missing checkpoint directory: $checkpoint" >&2
  exit 1
fi

output_024="$suite_root/${suite_name}_thr0p24"
output_038="$suite_root/${suite_name}_thr0p38"
output_055="$suite_root/${suite_name}_thr0p55"

run_threshold() {
  local threshold=$1
  local output_dir=$2
  local reusable_baseline=$3
  local resume_args=()
  local baseline_args=()
  if [[ -f "$output_dir/run_config.json" ]]; then
    resume_args=(--resume)
  fi
  if [[ -n "$reusable_baseline" ]]; then
    baseline_args=(--baseline-source "$reusable_baseline")
  fi
  echo "[$(date --iso-8601=seconds)] starting threshold=$threshold output=$output_dir"
  "$python_bin" "$script_dir/run_vbench200.py" \
    --output-dir "$output_dir" \
    --threshold "$threshold" \
    --wan22-root "$wan22_root" \
    --checkpoint-dir "$checkpoint" \
    --gpu-ids 0 1 2 3 \
    --worker-launch-wave-size 2 \
    --stagger-workers-gpu-memory-mib 8192 \
    --stagger-worker-timeout-seconds 1800 \
    --generation-python "$python_bin" \
    --video-metrics-python "$python_bin" \
    --vbench-python "$vbench_python" \
    --defer-evaluation \
    "${baseline_args[@]}" \
    "${resume_args[@]}"
  "$python_bin" - "$output_dir/status.json" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if status.get("status") != "inference_complete_evaluation_pending":
    raise SystemExit(f"unexpected inference status: {status}")
PY
}

if [[ "$start_threshold" == 0.24 ]]; then
  run_threshold 0.24 "$output_024" "$baseline_source"
  if [[ -z "$baseline_source" ]]; then
    baseline_source="$output_024/baseline"
  fi
fi
if [[ "$start_threshold" == 0.24 || "$start_threshold" == 0.38 ]]; then
  run_threshold 0.38 "$output_038" "$baseline_source"
fi
run_threshold 0.55 "$output_055" "$baseline_source"

echo "[$(date --iso-8601=seconds)] all inference thresholds complete; evaluation remains pending"
