#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
required_idle_polls=${REQUIRED_IDLE_POLLS:-6}
poll_seconds=${POLL_SECONDS:-10}
idle_polls=0

echo "$(date --iso-8601=seconds) waiting for all four GPUs to remain idle"
while (( idle_polls < required_idle_polls )); do
  random_running=0
  if pgrep -af 'ours4wan21_data.collector.*wan21_random_threshold_collection_v1_stage1' >/dev/null; then
    random_running=1
  fi
  gpu_busy=0
  if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
      | awk -F, '$1 + 0 > 100 || $2 + 0 > 5 { found=1 } END { exit !found }'; then
    gpu_busy=1
  fi
  if (( random_running == 0 && gpu_busy == 0 )); then
    idle_polls=$((idle_polls + 1))
  else
    idle_polls=0
  fi
  if (( idle_polls < required_idle_polls )); then
    sleep "$poll_seconds"
  fi
done

echo "$(date --iso-8601=seconds) four GPUs were idle for $((required_idle_polls * poll_seconds)) seconds; launching calibration"
exec "$script_dir/launch_4gpu.sh"
