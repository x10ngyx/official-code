#!/usr/bin/env bash
set -euo pipefail

phase=${1:-plan}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
data_project=$(cd "$script_dir/../.." && pwd)
official_code=$(cd "$data_project/../.." && pwd)

if [[ -n ${WAN22_PYTHON:-} ]]; then
  python_cmd=("$WAN22_PYTHON")
else
  conda_bin=${CONDA_BIN:-$(command -v conda || true)}
  if [[ -z $conda_bin ]]; then
    echo "conda is unavailable; set WAN22_PYTHON to the wan2.2 environment Python or set CONDA_BIN" >&2
    exit 2
  fi
  python_cmd=("$conda_bin" run --no-capture-output -n wan2.2 python)
fi
exp_base=${EXP_BASE:-/all/yiran07-disk3/huteng_data/exp}
if [[ -n ${ARCHIVE_ROOT:-} ]]; then
  archive_root=$ARCHIVE_ROOT
  run_id=${RUN_ID:-$(basename "$archive_root")}
else
  run_id=${RUN_ID:-wan21_random_threshold_v1_$(date +%Y%m%d_%H%M%S)}
  archive_root=$exp_base/$run_id
fi
prompt_pool=${PROMPT_POOL:-$data_project/resources/prompts/openvidhd_balanced_5000.upstream.jsonl}
calibration_config=${CALIBRATION_CONFIG:-$data_project/configs/speed_threshold_mapping.pending.json}
wan21_root=${WAN21_ROOT:-}
checkpoint_dir=${CHECKPOINT_DIR:-}
plan_manifest=$archive_root/manifests/random_plan.jsonl
runnable_manifest=$archive_root/manifests/random_runnable.jsonl
flops_profile=${FLOPS_PROFILE:-$archive_root/calflops_profile.json}
metrics_device=${METRICS_DEVICE:-auto}
lpips_batch_size=${LPIPS_BATCH_SIZE:-8}
metrics_model_cache=${METRICS_MODEL_CACHE:-${TORCH_HOME:-}}
vbench_python=${VBENCH_PYTHON:-}
vbench_cache=${VBENCH_CACHE_DIR:-$official_code/../../models/VBench}
vbench_runner=$official_code/VbenchEvaluation/run_custom_vbench.sh
profile_script=$data_project/experiments/calflops_profile_v1/profile_wan21_dit.py
result_link=$data_project/experiment_results/$run_id

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export OURS4WAN21_EXP_BASE=$exp_base
export PYTHONPATH=$data_project/src:$official_code/VideoMetrics:$official_code/CalflopsEvaluation${PYTHONPATH:+:$PYTHONPATH}

if [[ ! -x ${python_cmd[0]} ]]; then
  echo "wan2.2 Python/Conda launcher is not executable: ${python_cmd[0]}" >&2
  exit 2
fi

prepare_archive() {
  mkdir -p "$archive_root/manifests" "$archive_root/logs"
  if [[ ! -e $archive_root/README.md ]]; then
    printf '%s\n' \
      '# Ours4Wan21 random-threshold collection archive' \
      '' \
      'Generated artifacts for the frozen Wan2.1-T2V-1.3B random-threshold pipeline.' \
      'See manifests/, shared_baselines/, shards/, completed/, published/, audits/, and logs/.' \
      > "$archive_root/README.md"
  fi
  if [[ -L $result_link ]]; then
    if [[ $(readlink -f "$result_link") != $(readlink -f "$archive_root") ]]; then
      echo "existing result symlink points elsewhere: $result_link" >&2
      exit 2
    fi
  elif [[ -e $result_link ]]; then
    echo "result index path already exists and is not a symlink: $result_link" >&2
    exit 2
  else
    ln -s "$archive_root" "$result_link"
  fi
}

require_file() {
  if [[ ! -f $1 ]]; then
    echo "required file is missing: $1" >&2
    exit 2
  fi
}

require_model_inputs() {
  if [[ -z $wan21_root || -z $checkpoint_dir ]]; then
    echo "set WAN21_ROOT and CHECKPOINT_DIR before GPU phases" >&2
    exit 2
  fi
  require_file "$wan21_root/generate.py"
  if [[ ! -d $checkpoint_dir ]]; then
    echo "checkpoint directory is missing: $checkpoint_dir" >&2
    exit 2
  fi
}

require_gpu_inputs() {
  require_model_inputs
  require_file "$flops_profile"
}

preflight() {
  local requested_phase=$1
  local args=(
    -m ours4wan21_data.preflight --phase "$requested_phase"
    --exp-base "$exp_base" --archive-root "$archive_root"
  )
  case "$requested_phase" in
    profile|baselines|candidates)
      args+=(--wan21-root "$wan21_root" --checkpoint-dir "$checkpoint_dir")
      ;;
  esac
  case "$requested_phase" in
    baselines|candidates|finalize)
      args+=(--flops-profile "$flops_profile")
      ;;
  esac
  case "$requested_phase" in
    candidates|finalize)
      args+=(--runnable-manifest "$runnable_manifest")
      ;;
  esac
  if [[ $requested_phase == candidates && -n $metrics_model_cache ]]; then
    args+=(--metrics-model-cache "$metrics_model_cache")
  fi
  "${python_cmd[@]}" "${args[@]}"
}

make_plan() {
  require_file "$prompt_pool"
  if [[ -f $plan_manifest ]]; then
    "${python_cmd[@]}" -m ours4wan21_data.manifest validate --manifest "$plan_manifest"
  else
    "${python_cmd[@]}" -m ours4wan21_data.manifest plan \
      --prompt-pool "$prompt_pool" --output "$plan_manifest"
  fi
}

materialize() {
  require_file "$plan_manifest"
  require_file "$calibration_config"
  if [[ -f $runnable_manifest ]]; then
    "${python_cmd[@]}" -m ours4wan21_data.manifest validate --manifest "$runnable_manifest"
  else
    "${python_cmd[@]}" -m ours4wan21_data.manifest materialize \
      --plan "$plan_manifest" --calibration "$calibration_config" \
      --output "$runnable_manifest"
  fi
}

run_workers() {
  local mode=$1
  local manifest=$2
  local pids=()
  local metric_args=(--metrics-device "$metrics_device" --lpips-batch-size "$lpips_batch_size")
  if [[ -n $metrics_model_cache ]]; then
    metric_args+=(--metrics-model-cache "$metrics_model_cache")
  fi
  local gpu
  for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu "${python_cmd[@]}" -m ours4wan21_data.collector \
      --mode "$mode" --manifest "$manifest" --parent-root "$archive_root" \
      --shard-index "$gpu" --wan21-root "$wan21_root" \
      --checkpoint-dir "$checkpoint_dir" --flops-profile "$flops_profile" \
      "${metric_args[@]}" --resume \
      > "$archive_root/logs/${mode}_shard_${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ $mode == candidate ]]; then
    if ! "${python_cmd[@]}" -m ours4wan21_data.publisher \
      --manifest "$manifest" --parent-root "$archive_root" \
      > "$archive_root/logs/candidate_publication.log" 2>&1; then
      status=1
    fi
  fi
  return "$status"
}

run_vbench() {
  if [[ -z $vbench_python || ! -x $vbench_python ]]; then
    echo "set VBENCH_PYTHON to the VBench environment Python" >&2
    exit 2
  fi
  require_file "$vbench_runner"
  if [[ ! -d $vbench_cache ]]; then
    echo "VBench cache directory is missing: $vbench_cache" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES=${VBENCH_GPU:-0} "${python_cmd[@]}" \
    -m ours4wan21_data.vbench \
    --manifest "$runnable_manifest" --parent-root "$archive_root" \
    --runner "$vbench_runner" --vbench-python "$vbench_python" \
    --vbench-cache "$vbench_cache" \
    > "$archive_root/logs/vbench.log" 2>&1
}

preflight package
if [[ $phase == preflight ]]; then
  exit 0
fi
prepare_archive
case "$phase" in
  plan)
    preflight plan
    make_plan
    ;;
  profile)
    require_model_inputs
    preflight profile
    if [[ -e $flops_profile ]]; then
      echo "refusing to overwrite existing profile: $flops_profile" >&2
      exit 2
    fi
    CUDA_VISIBLE_DEVICES=${PROFILE_GPU:-0} "${python_cmd[@]}" "$profile_script" \
      --wan21-root "$wan21_root" --checkpoint-dir "$checkpoint_dir" \
      --output "$flops_profile" | tee "$archive_root/logs/calflops_profile.log"
    ;;
  baselines)
    make_plan
    require_gpu_inputs
    preflight baselines
    run_workers baseline "$plan_manifest"
    ;;
  materialize)
    materialize
    ;;
  candidates)
    materialize
    require_gpu_inputs
    preflight candidates
    run_workers candidate "$runnable_manifest"
    ;;
  publish)
    require_file "$runnable_manifest"
    "${python_cmd[@]}" -m ours4wan21_data.publisher \
      --manifest "$runnable_manifest" --parent-root "$archive_root"
    ;;
  vbench)
    require_file "$runnable_manifest"
    run_vbench
    ;;
  finalize)
    require_file "$runnable_manifest"
    require_file "$flops_profile"
    preflight finalize
    "${python_cmd[@]}" -m ours4wan21_data.publisher \
      --manifest "$runnable_manifest" --parent-root "$archive_root" --require-complete
    run_vbench
    mkdir -p "$archive_root/audits"
    audit_output=$archive_root/audits/archive_audit.json
    if [[ -e $audit_output ]]; then
      echo "refusing to overwrite audit: $audit_output" >&2
      exit 2
    fi
    audit_args=(
      -m ours4wan21_data.audit --manifest "$runnable_manifest"
      --parent-root "$archive_root" --flops-profile "$flops_profile"
      --output "$audit_output" --require-complete
    )
    if [[ ${DEEP_LATENTS:-0} == 1 ]]; then
      audit_args+=(--deep-latents)
    fi
    "${python_cmd[@]}" "${audit_args[@]}"
    ;;
  *)
    echo "usage: $0 {preflight|plan|profile|baselines|materialize|candidates|publish|vbench|finalize}" >&2
    exit 2
    ;;
esac
