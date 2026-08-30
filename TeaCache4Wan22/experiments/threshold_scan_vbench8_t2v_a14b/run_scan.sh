#!/usr/bin/env bash
set -euo pipefail

experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$experiment_dir/../.." && pwd)
repository_dir=$(cd "$project_dir/.." && pwd)
python_bin=${WAN22_PYTHON:-python}
gpu_ids_text=${GPU_IDS:-1,2}
enable_warmup=${ENABLE_WARMUP:-1}

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
  IFS=',' read -r -a plan_gpus <<< "$gpu_ids_text"
  "$python_bin" "$experiment_dir/plan_scan.py" --worker-count "${#plan_gpus[@]}"
  exit 0
fi

for name in RESULT_ROOT WAN22_SOURCE WAN22_CKPT CALFLOPS_SOURCE VBENCH_PYTHON; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done

result_root=$(
  "$python_bin" - "$RESULT_ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).expanduser().resolve()
external = Path('/all/yiran07-disk3/huteng_data/exp').resolve()
try:
    root.relative_to(external)
except ValueError as error:
    raise SystemExit(f'RESULT_ROOT must be below {external}: {root}') from error
print(root)
PY
)
result_link="$project_dir/experiment_results/$(basename "$result_root")"
mkdir -p "$project_dir/experiment_results"
if [[ -L $result_link ]]; then
  if [[ $(readlink -f "$result_link") != "$result_root" ]]; then
    echo "result symlink points elsewhere: $result_link" >&2
    exit 2
  fi
elif [[ -e $result_link ]]; then
  echo "result index exists and is not a symlink: $result_link" >&2
  exit 2
else
  ln -s "$result_root" "$result_link"
fi

IFS=',' read -r -a gpu_ids <<< "$gpu_ids_text"
worker_count=${#gpu_ids[@]}
if ((worker_count < 1 || worker_count > 8)); then
  echo "GPU_IDS must contain between 1 and 8 GPU IDs" >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu in "${gpu_ids[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "invalid physical GPU ID: $gpu" >&2
    exit 2
  fi
  if [[ -n "${seen_gpus[$gpu]:-}" ]]; then
    echo "duplicate physical GPU ID: $gpu" >&2
    exit 2
  fi
  seen_gpus[$gpu]=1
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  if ((used > 1024)); then
    echo "physical GPU $gpu is not idle (${used} MiB used)" >&2
    exit 1
  fi
done

if [[ ! -f "$WAN22_SOURCE/.teacache4wan22_prepared.json" ]]; then
  echo "WAN22_SOURCE is not a prepared TeaCache4Wan22 tree: $WAN22_SOURCE" >&2
  exit 1
fi
if [[ ! -d "$WAN22_CKPT" ]]; then
  echo "missing Wan2.2 checkpoint directory: $WAN22_CKPT" >&2
  exit 1
fi
if [[ ! -f "$CALFLOPS_SOURCE/calflops/__init__.py" ]]; then
  echo "CALFLOPS_SOURCE is not a calculate-flops.pytorch checkout: $CALFLOPS_SOURCE" >&2
  exit 1
fi

"$python_bin" "$project_dir/scripts/validate_prepared_tree.py" \
  --source "$WAN22_SOURCE" \
  --mode prepared
mkdir -p "$result_root/logs"
for snapshot in scan_config.json prompts.jsonl; do
  if [[ -e "$result_root/$snapshot" ]]; then
    cmp "$experiment_dir/$snapshot" "$result_root/$snapshot"
  else
    install -m 0644 "$experiment_dir/$snapshot" "$result_root/$snapshot"
  fi
done
if [[ ! -e "$result_root/README.md" ]]; then
  install -m 0644 "$experiment_dir/RESULT_README.md" "$result_root/README.md"
fi
scan_plan_tmp="$result_root/scan_plan.json.tmp.$$"
"$python_bin" "$experiment_dir/plan_scan.py" \
  --worker-count "$worker_count" > "$scan_plan_tmp"
mv "$scan_plan_tmp" "$result_root/scan_plan.json"

calflops_profile="$result_root/calflops_profile.json"
if [[ ! -f "$calflops_profile" ]]; then
  profile_gpu=${gpu_ids[0]}
  CUDA_VISIBLE_DEVICES="$profile_gpu" \
    "$python_bin" "$project_dir/experiments/performance_t2v_a14b/profile_calflops.py" \
      --wan22-root "$WAN22_SOURCE" \
      --checkpoint-dir "$WAN22_CKPT" \
      --calflops-source "$CALFLOPS_SOURCE" \
      --output "$calflops_profile" \
      > "$result_root/logs/calflops_profile.log" 2>&1
fi

pids=()
for ((worker_index = 0; worker_index < worker_count; worker_index++)); do
  gpu=${gpu_ids[$worker_index]}
  worker_log="$result_root/logs/worker_${worker_index}_gpu_${gpu}.log"
  CUDA_VISIBLE_DEVICES="$gpu" \
    bash "$experiment_dir/run_worker.sh" \
      "$result_root" "$worker_index" "$worker_count" "$gpu" "$enable_warmup" \
      >> "$worker_log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if ((failed)); then
  echo "one or more scan workers failed; inspect $result_root/logs" >&2
  exit 1
fi

metric_gpu=${METRIC_GPU_ID:-${gpu_ids[0]}}
CUDA_VISIBLE_DEVICES="$metric_gpu" \
  "$python_bin" "$experiment_dir/finalize_scan.py" \
    --result-root "$result_root" \
    --calflops-profile "$calflops_profile" \
    --metric-device cuda:0 \
    --lpips-batch-size "${LPIPS_BATCH_SIZE:-2}" \
    --model-cache "${TORCH_HOME:-$repository_dir/../../models/torch-cache}" \
    --vbench-python "$VBENCH_PYTHON" \
    --vbench-cache "${VBENCH_CACHE_DIR:-$repository_dir/../../models/VBench}" \
    > "$result_root/logs/finalize_scan.log" 2>&1

"$python_bin" - "$result_root/COMPLETE.json" "$result_root/VALIDATION.json" <<'PY'
import datetime, hashlib, json, os, sys
target, validation_path = sys.argv[1:]
validation = json.load(open(validation_path, encoding='utf-8'))
if validation.get('status') != 'pass':
    raise SystemExit('scan validation did not pass')
payload = {
    'schema_version': 1,
    'status': 'complete',
    'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'validation_path': validation_path,
    'validation_sha256': hashlib.sha256(open(validation_path, 'rb').read()).hexdigest(),
}
temporary = f"{target}.tmp.{os.getpid()}"
with open(temporary, 'w', encoding='utf-8') as handle:
    json.dump(payload, handle, indent=2)
    handle.write('\n')
os.replace(temporary, target)
PY

echo "threshold scan complete: $result_root"
