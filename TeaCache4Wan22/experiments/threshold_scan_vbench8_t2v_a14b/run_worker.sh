#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 RESULT_ROOT WORKER_INDEX WORKER_COUNT PHYSICAL_GPU_ID ENABLE_WARMUP" >&2
  exit 2
fi

result_root=$1
worker_index=$2
worker_count=$3
physical_gpu=$4
enable_warmup=$5
experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${WAN22_PYTHON:-python}

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for name in WAN22_SOURCE WAN22_CKPT; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done

mkdir -p "$result_root/telemetry" "$result_root/worker_status"
telemetry="$result_root/telemetry/gpu_${physical_gpu}.csv"
if [[ ! -e "$telemetry" ]]; then
  echo "timestamp,index,temperature_gpu_c,utilization_gpu_percent,memory_used_mib,clocks_sm_mhz,power_draw_w,pstate" > "$telemetry"
fi

sample_telemetry() {
  while true; do
    nvidia-smi --id="$physical_gpu" \
      --query-gpu=timestamp,index,temperature.gpu,utilization.gpu,memory.used,clocks.sm,power.draw,pstate \
      --format=csv,noheader,nounits >> "$telemetry" || true
    sleep 30
  done
}

sample_telemetry &
telemetry_pid=$!
cleanup() {
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$python_bin" "$experiment_dir/scan_worker.py" \
  --result-root "$result_root" \
  --worker-index "$worker_index" \
  --worker-count "$worker_count" \
  --physical-gpu "$physical_gpu" \
  --enable-warmup "$enable_warmup"
