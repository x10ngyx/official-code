#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
workspace=$(cd "$project_dir/../../.." && pwd)
exp_base=${EXP_BASE:-/mnt/hdd/xiongyuxiang/tmp/exp}
suite=${SUITE_NAME:-ours21_random3000_12groups_v1}
run_name=${RUN_NAME:-${suite}_vbench5_speed_targets_v1}
python_bin=${WAN22_PYTHON:-$workspace/data/environments/Wan2.2-conda-env/bin/python}
training_complete="$exp_base/${suite}_orchestration/COMPLETE.json"

[[ -x "$python_bin" ]] || { echo "Wan2.2 Python is not executable: $python_bin" >&2; exit 2; }
while [[ ! -f "$training_complete" ]]; do
  sleep 60
done

export OURS4WAN21_WORKSPACE="$workspace"
export OURS4WAN21_EXP_BASE="$exp_base"
export EXP_BASE="$exp_base"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
args=(
  --suite-name "$suite"
  --run-name "$run_name"
  --gpus 0 1 2 3
  --wan21-root "$workspace/data/source/Wan2.1-65386b2"
  --checkpoint-dir "$workspace/models/Wan2.1-T2V-1.3B"
  --flops-profile "$exp_base/wan21_seacache_threshold_collection_v1/calflops_profile.json"
  --prior-calibration "$exp_base/wan21_seacache_speedup_calibration_v1/analysis/speed_threshold_mapping.calibrated.json"
)
[[ -f "$exp_base/$run_name/config.json" ]] && args+=(--resume)
cd "$project_dir"
"$python_bin" experiments/vbench5_speed_targets_v1/run_pipeline.py "${args[@]}"
