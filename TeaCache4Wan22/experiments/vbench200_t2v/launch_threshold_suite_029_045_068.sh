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
experiment_root=/mnt/hdd/xiongyuxiang/tmp/exp
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

wan22_root=${WAN22_SOURCE:-"$project_dir/build/Wan2.2-42bf4cf-prepared"}
checkpoint=${WAN22_CKPT:-"$workspace_root/models/Wan2.2-T2V-A14B"}
coefficients=${TEACACHE_COEFFICIENTS:-"$project_dir/coefficients/wan22_t2v_a14b_50step_dpmpp_nonretention.json"}
baseline_source=${WAN22_BASELINE_SOURCE:-"/all/yiran07-disk3/huteng_data/exp/wan22_seacache_vbench200_thr024_038_055_fullwall_staggered_gpu0123_20260830_162817/wan22_seacache_vbench200_thr024_038_055_fullwall_staggered_gpu0123_20260830_162817_thr0p24/baseline"}
vbench_source=${VBENCH_SOURCE:-"/all/yiran07-disk3/huteng_data/exp/vbench_single_sample_smoke_fd18b3d_20260829_141200/upstream/VBench"}
vbench_extra_deps=${VBENCH_EXTRA_DEPS:-"/all/yiran07-disk3/huteng_data/exp/vbench_single_sample_smoke_fd18b3d_20260829_141200/python_deps"}
vbench_pythonpath="$vbench_source:$vbench_extra_deps"
vbench_cache=${VBENCH_CACHE_DIR:-"$workspace_root/models/VBench"}
video_metrics_cache=${VIDEO_METRICS_CACHE_DIR:-"$workspace_root/models/torch-cache"}
suite_name=$(basename "$suite_root")
output_029="$suite_root/${suite_name}_thr0p29"
output_045="$suite_root/${suite_name}_thr0p45"
output_068="$suite_root/${suite_name}_thr0p68"
reference_vbench="$output_029/evaluation/vbench_reference"
calflops_profile="$output_029/performance/calflops_profile.json"

for path in "$python_bin" "$wan22_root/.teacache4wan22_prepared.json" "$coefficients" "$vbench_source/vbench/__init__.py"; do
  if [[ ! -e $path ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
done
for path in "$checkpoint" "$baseline_source" "$vbench_extra_deps" "$vbench_cache" "$video_metrics_cache"; do
  if [[ ! -d $path ]]; then
    echo "missing required directory: $path" >&2
    exit 1
  fi
done

common_args=(
  --wan22-root "$wan22_root"
  --checkpoint-dir "$checkpoint"
  --coefficients "$coefficients"
  --baseline-source "$baseline_source"
  --gpu-ids 0 1 2 3
  --worker-launch-wave-size 2
  --stagger-workers-gpu-memory-mib 8192
  --stagger-worker-timeout-seconds 1800
  --generation-python "$python_bin"
  --video-metrics-python "$python_bin"
  --vbench-python "$python_bin"
  --vbench-pythonpath "$vbench_pythonpath"
  --vbench-cache-dir "$vbench_cache"
  --video-metrics-cache-dir "$video_metrics_cache"
)

echo "[$(date --iso-8601=seconds)] preflight: source, shared baseline, Calflops, VideoMetrics, all 16 VBench dimensions"
"$python_bin" "$project_dir/../VbenchEvaluation/validate_resources.py"
"$python_bin" "$project_dir/../VbenchEvaluation/validate_downloaded_weights.py" \
  --weights-dir "$vbench_cache"
"$python_bin" "$script_dir/run_vbench200.py" \
  --output-dir "$output_029" \
  --threshold 0.29 \
  --preflight-only \
  "${common_args[@]}"

mkdir -p "$suite_root"
suite_link="$project_dir/experiment_results/$suite_name"
if [[ -L $suite_link ]]; then
  if [[ $(readlink -f "$suite_link") != $(readlink -f "$suite_root") ]]; then
    echo "suite result link points elsewhere: $suite_link" >&2
    exit 1
  fi
elif [[ -e $suite_link ]]; then
  echo "suite result link path already exists: $suite_link" >&2
  exit 1
else
  ln -s "$suite_root" "$suite_link"
fi

run_threshold() {
  local phase=$1
  local threshold=$2
  local output_dir=$3
  local reference_source=${4:-}
  local profile_source=${5:-}
  local resume_args=()
  local reference_args=()
  local phase_args=()
  local profile_args=()
  if [[ -f $output_dir/run_config.json ]]; then
    resume_args=(--resume)
  fi
  if [[ -n $reference_source ]]; then
    reference_args=(--reference-vbench-source "$reference_source")
  fi
  if [[ $phase == generation ]]; then
    phase_args=(--defer-evaluation)
  fi
  if [[ -n $profile_source ]]; then
    profile_args=(--calflops-profile-source "$profile_source")
  fi
  echo "[$(date --iso-8601=seconds)] $phase threshold=$threshold output=$output_dir"
  "$python_bin" "$script_dir/run_vbench200.py" \
    --output-dir "$output_dir" \
    --threshold "$threshold" \
    "${common_args[@]}" \
    "${reference_args[@]}" \
    "${profile_args[@]}" \
    "${resume_args[@]}" \
    "${phase_args[@]}"
}

# Generate all 600 candidates first. Each threshold uses four persistent workers;
# GPUs 0/1 initialize first and GPUs 2/3 start only after the first wave is ready.
run_threshold generation 0.29 "$output_029"
run_threshold generation 0.45 "$output_045" "$reference_vbench" "$calflops_profile"
run_threshold generation 0.68 "$output_068" "$reference_vbench" "$calflops_profile"

# Resume without loading WanT2V: complete candidate sets are validated and skipped,
# then PSNR/SSIM/LPIPS, VBench, and per-threshold reports run automatically.
run_threshold evaluation 0.29 "$output_029"
run_threshold evaluation 0.45 "$output_045" "$reference_vbench" "$calflops_profile"
run_threshold evaluation 0.68 "$output_068" "$reference_vbench" "$calflops_profile"

if [[ ! -f $suite_root/suite_report.json ]]; then
  "$python_bin" "$script_dir/build_suite_report.py" \
    --suite-root "$suite_root" \
    --result-dir "$output_029" \
    --result-dir "$output_045" \
    --result-dir "$output_068"
fi
echo "[$(date --iso-8601=seconds)] TeaCache formal threshold suite complete: $suite_root"
