#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
data_project=$(cd "$script_dir/../.." && pwd)

python_bin=${WAN21_PYTHON:-/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python}
wan21_root=${WAN21_ROOT:-/mnt/hdd/xiongyuxiang/tmp/data/source/Wan2.1-65386b2}
checkpoint_dir=${CHECKPOINT_DIR:-/mnt/hdd/xiongyuxiang/tmp/models/Wan2.1-T2V-1.3B}
baseline_root=${BASELINE_RUN_ROOT:-/mnt/hdd/xiongyuxiang/tmp/exp/wan21_seacache_threshold_collection_v1}
source_manifest=${SOURCE_MANIFEST:-$baseline_root/manifests/seacache_manifest.jsonl}
flops_profile=${FLOPS_PROFILE:-$baseline_root/calflops_profile.json}
exp_base=${EXP_BASE:-/mnt/hdd/xiongyuxiang/tmp/exp}
run_id=${RUN_ID:-wan21_seacache_speedup_calibration_v1}
output_root=$exp_base/$run_id
result_link=$data_project/experiment_results/$run_id

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$data_project/src${PYTHONPATH:+:$PYTHONPATH}

for required in "$python_bin" "$wan21_root/generate.py" "$source_manifest" "$flops_profile"; do
  if [[ ! -e $required ]]; then
    echo "required input is missing: $required" >&2
    exit 2
  fi
done
if [[ ! -d $checkpoint_dir ]]; then
  echo "checkpoint directory is missing: $checkpoint_dir" >&2
  exit 2
fi

"$python_bin" "$script_dir/run_calibration.py" prepare \
  --output-root "$output_root" \
  --baseline-root "$baseline_root" \
  --source-manifest "$source_manifest" \
  --flops-profile "$flops_profile"

if [[ -L $result_link ]]; then
  if [[ $(readlink -f "$result_link") != $(readlink -f "$output_root") ]]; then
    echo "existing result symlink points elsewhere: $result_link" >&2
    exit 2
  fi
elif [[ -e $result_link ]]; then
  echo "result index exists and is not a symlink: $result_link" >&2
  exit 2
else
  ln -s "$output_root" "$result_link"
fi

mkdir -p "$output_root/logs"
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu "$python_bin" "$script_dir/run_calibration.py" worker \
    --output-root "$output_root" \
    --shard-index "$gpu" \
    --wan21-root "$wan21_root" \
    --checkpoint-dir "$checkpoint_dir" \
    --flops-profile "$flops_profile" \
    --resume \
    > "$output_root/logs/worker_shard_${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ $status != 0 ]]; then
  echo "one or more calibration workers failed; see $output_root/logs" >&2
  exit "$status"
fi

"$python_bin" "$script_dir/analyze_calibration.py" --output-root "$output_root" \
  > "$output_root/logs/analysis.log" 2>&1
cat "$output_root/logs/analysis.log"
