#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 ENV_ROOT WAN21_ROOT CHECKPOINT_DIR OUTPUT_DIR [GPU_INDEX]" >&2
  exit 2
fi

env_root=$(readlink -f "$1")
wan21_root=$(readlink -f "$2")
checkpoint_dir=$(readlink -f "$3")
output_dir=$4
gpu_index=${5:-0}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(readlink -f "$script_dir/../..")
repository_dir=$(readlink -f "$project_dir/..")
python_bin="$env_root/bin/python"
run_entrypoint="$project_dir/run_wan21.sh"
metrics_entrypoint="$repository_dir/VideoMetrics/evaluate.py"
expected_commit=65386b2e03c490796eede31b0325a6a595cc684e
exp_root=/mnt/hdd/xiongyuxiang/tmp/exp
prompt='A red toy car drives slowly across a clean white studio floor.'

if [[ ! -x "$python_bin" ]]; then
  echo "environment Python is unavailable: $python_bin" >&2
  exit 1
fi
if [[ ! -d "$checkpoint_dir" ]]; then
  echo "checkpoint directory is unavailable: $checkpoint_dir" >&2
  exit 1
fi
if [[ ! -f "$wan21_root/generate.py" || ! -d "$wan21_root/wan" ]]; then
  echo "Wan2.1 source tree is unavailable: $wan21_root" >&2
  exit 1
fi
if [[ "$(git -C "$wan21_root" rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "Wan2.1 source is not pinned to $expected_commit" >&2
  exit 1
fi

mkdir -p "$exp_root"
exp_root=$(readlink -f "$exp_root")
output_parent=$(dirname "$output_dir")
mkdir -p "$output_parent"
output_parent=$(readlink -f "$output_parent")
output_dir="$output_parent/$(basename "$output_dir")"
case "$output_dir" in
  "$exp_root"/*) ;;
  *) echo "output must be below $exp_root: $output_dir" >&2; exit 1 ;;
esac
if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite existing result: $output_dir" >&2
  exit 1
fi

mkdir "$output_dir"
mkdir "$output_dir/logs" "$output_dir/metrics" "$output_dir/latency" "$output_dir/flops"
cp "$script_dir/README.md" "$output_dir/README.md"

export CUDA_VISIBLE_DEVICES="$gpu_index"
export WAN21_PYTHON="$python_bin"
export PATH="$env_root/bin:$PATH"

"$python_bin" "$script_dir/capture_environment.py" \
  --wan21-root "$wan21_root" \
  --checkpoint-dir "$checkpoint_dir" \
  --entrypoint "$project_dir/generate.py" \
  --output "$output_dir/environment.json" \
  --prompt "$prompt"

common_args=(
  --task t2v-1.3B
  --size '832*480'
  --frame_num 5
  --ckpt_dir "$checkpoint_dir"
  --prompt "$prompt"
  --base_seed 42
  --offload_model True
  --t5_cpu
  --sample_solver unipc
  --sample_steps 4
  --sample_shift 5.0
  --sample_guide_scale 5.0
)

baseline="$output_dir/baseline.mp4"
candidate="$output_dir/teacache_threshold0.mp4"
baseline_inference_timing="$output_dir/latency/baseline_inference.json"
candidate_inference_timing="$output_dir/latency/teacache_threshold0_inference.json"
baseline_process_timing="$output_dir/latency/baseline_process.json"
candidate_process_timing="$output_dir/latency/teacache_threshold0_process.json"

baseline_command=(bash "$run_entrypoint" "$wan21_root" \
  --timing_json "$baseline_inference_timing" \
  "${common_args[@]}" \
  --save_file "$baseline")
"$python_bin" "$script_dir/run_timed_command.py" \
  --output "$baseline_process_timing" \
  --log "$output_dir/logs/baseline.log" \
  -- "${baseline_command[@]}"

candidate_command=(bash "$run_entrypoint" "$wan21_root" \
  --enable_teacache \
  --teacache_thresh 0 \
  --timing_json "$candidate_inference_timing" \
  "${common_args[@]}" \
  --save_file "$candidate")
"$python_bin" "$script_dir/run_timed_command.py" \
  --output "$candidate_process_timing" \
  --log "$output_dir/logs/teacache_threshold0.log" \
  -- "${candidate_command[@]}"

"$python_bin" "$script_dir/compare_outputs.py" \
  --baseline "$baseline" \
  --candidate "$candidate" \
  --output "$output_dir/comparison.json" \
  >"$output_dir/logs/comparison.log" 2>&1

"$python_bin" "$metrics_entrypoint" \
  --reference "$baseline" \
  --candidate "$candidate" \
  --video-id threshold_zero_smoke \
  --output-dir "$output_dir/metrics" \
  --metrics psnr ssim \
  --expected-frames 5 \
  >"$output_dir/logs/metrics.log" 2>&1

"$python_bin" "$script_dir/profile_wan21_dit.py" \
  --wan21-root "$wan21_root" \
  --checkpoint-dir "$checkpoint_dir" \
  --baseline-timing "$baseline_inference_timing" \
  --candidate-timing "$candidate_inference_timing" \
  --baseline-process-timing "$baseline_process_timing" \
  --candidate-process-timing "$candidate_process_timing" \
  --output "$output_dir/flops/calflops_profile.json" \
  --width 832 \
  --height 480 \
  --frame-num 5 \
  >"$output_dir/logs/calflops.log" 2>&1

echo "threshold-zero smoke test passed: $output_dir"
