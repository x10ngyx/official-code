#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 REFERENCE_VIDEOS CANDIDATE_VIDEOS OUTPUT_DIR [EXPECTED_SEEDS] [EXPECTED_FRAMES]" >&2
  exit 2
fi

reference_videos=$(readlink -f "$1")
candidate_videos=$(readlink -f "$2")
output_dir=$(readlink -m "$3")
expected_seeds=${4:-1}
expected_frames=${5:-81}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository_dir=$(cd "$script_dir/../../.." && pwd)
exp_root=/mnt/hdd/xiongyuxiang/tmp/exp
video_metrics_env=${VIDEO_METRICS_CONDA_ENV-wan2.2}
vbench_env=${VBENCH_CONDA_ENV-vbench-eval}
metrics_device=${METRICS_DEVICE:-cuda:0}

case "$output_dir/" in
  "$exp_root"/*) ;;
  *) echo "output must be below $exp_root: $output_dir" >&2; exit 1 ;;
esac
if [[ ! -d "$reference_videos" || ! -d "$candidate_videos" ]]; then
  echo "reference and candidate video directories must exist" >&2
  exit 1
fi

run_in_conda() {
  local environment=$1
  shift
  if [[ -n "$environment" ]]; then
    conda run -n "$environment" --no-capture-output "$@"
  else
    "$@"
  fi
}

mkdir -p "$output_dir"
if [[ ! -e "$output_dir/README.md" ]]; then
  printf '%s\n' \
    '# TeaCache4Wan21 T2V evaluation result' \
    '' \
    'Full-reference PSNR/SSIM/LPIPS and reference/candidate Vbench200 subset scores.' \
    'All metrics are produced by this repository’s VideoMetrics and VbenchEvaluation tools.' \
    > "$output_dir/README.md"
fi

run_in_conda "$video_metrics_env" \
  bash "$repository_dir/VideoMetrics/run_evaluation.sh" \
    --reference-dir "$reference_videos" \
    --candidate-dir "$candidate_videos" \
    --extension .mp4 \
    --expected-frames "$expected_frames" \
    --device "$metrics_device" \
    --output-dir "$output_dir/video_metrics"

run_in_conda "$vbench_env" \
  env PYTHON_BIN=python bash "$repository_dir/VbenchEvaluation/run_vbench200.sh" \
    "$reference_videos" "$output_dir/vbench_reference" "$expected_seeds"

run_in_conda "$vbench_env" \
  env PYTHON_BIN=python bash "$repository_dir/VbenchEvaluation/run_vbench200.sh" \
    "$candidate_videos" "$output_dir/vbench_candidate" "$expected_seeds"
