#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
calibration_config=${CALIBRATION_CONFIG:?set CALIBRATION_CONFIG to the calibrated mapping output}
prompt_limit=${PROMPT_LIMIT:?set PROMPT_LIMIT to the staged prompt count}
exp_base=${EXP_BASE:-/mnt/hdd/xiongyuxiang/tmp/exp}
run_id=${RUN_ID:?set RUN_ID to the random-threshold archive name}
archive_root=${ARCHIVE_ROOT:-$exp_base/$run_id}
required_idle_polls=${REQUIRED_IDLE_POLLS:-6}
poll_seconds=${POLL_SECONDS:-10}
idle_polls=0

echo "$(date --iso-8601=seconds) waiting for $prompt_limit baselines, calibrated mapping, and four idle GPUs"
while (( idle_polls < required_idle_polls )); do
  baseline_count=0
  if [[ -d $archive_root/shared_baselines ]]; then
    baseline_count=$(find "$archive_root/shared_baselines" -name BASELINE_COMPLETE.json -type f | wc -l)
  fi
  baselines_ready=0
  if (( baseline_count == prompt_limit )); then
    baselines_ready=1
  fi
  mapping_ready=0
  if [[ -s $calibration_config ]]; then
    mapping_ready=1
  fi
  gpu_busy=0
  if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
      | awk -F, '$1 + 0 > 100 || $2 + 0 > 5 { found=1 } END { exit !found }'; then
    gpu_busy=1
  fi
  if (( baselines_ready == 1 && mapping_ready == 1 && gpu_busy == 0 )); then
    idle_polls=$((idle_polls + 1))
  else
    idle_polls=0
  fi
  if (( idle_polls < required_idle_polls )); then
    sleep "$poll_seconds"
  fi
done

echo "$(date --iso-8601=seconds) baselines and mapping are complete; GPUs were idle for $((required_idle_polls * poll_seconds)) seconds"
bash "$script_dir/launch_4gpu.sh" candidates
bash "$script_dir/launch_4gpu.sh" finalize
