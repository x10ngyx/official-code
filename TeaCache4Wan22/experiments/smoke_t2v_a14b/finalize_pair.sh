#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 RESULT_ROOT BASELINE_RUN_ID TEACACHE_RUN_ID PROMPT" >&2
  exit 2
fi

result_root=$1
baseline_id=$2
teacache_id=$3
prompt=$4
experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$experiment_dir/../.." && pwd)
repository_dir=$(cd "$project_dir/.." && pwd)
python_bin=${WAN22_PYTHON:-python}
metrics_script=$repository_dir/VideoMetrics/evaluate.py
vbench_script=$repository_dir/VbenchEvaluation/run_custom_vbench.sh
profile=$result_root/flops/calflops_profile.json

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$python_bin" "$project_dir/experiments/performance_t2v_a14b/aggregate_performance.py" \
  --baseline-manifest "$result_root/${baseline_id}.manifest.json" \
  --teacache-manifest "$result_root/${teacache_id}.manifest.json" \
  --calflops-profile "$profile" \
  --output-dir "$result_root/performance"

"$python_bin" "$metrics_script" \
  --reference "$result_root/${baseline_id}.mp4" \
  --candidate "$result_root/${teacache_id}.mp4" \
  --video-id smoke_seed42 \
  --output-dir "$result_root/video_metrics" \
  --metrics psnr ssim lpips \
  --expected-frames 45 \
  --device cuda:0 \
  --model-cache "${TORCH_HOME:-$repository_dir/../../models/torch-cache}"

mkdir -p "$result_root/vbench_staging/baseline" "$result_root/vbench_staging/candidate"
ln -s "$(realpath --relative-to="$result_root/vbench_staging/baseline" "$result_root/${baseline_id}.mp4")" \
  "$result_root/vbench_staging/baseline/sample.mp4"
ln -s "$(realpath --relative-to="$result_root/vbench_staging/candidate" "$result_root/${teacache_id}.mp4")" \
  "$result_root/vbench_staging/candidate/sample.mp4"
"$python_bin" - "$result_root" "$prompt" <<'PY'
import json, sys
from pathlib import Path
root, prompt = Path(sys.argv[1]), sys.argv[2]
for condition in ("baseline", "candidate"):
    (root / f"vbench_{condition}_prompts.json").write_text(
        json.dumps({"sample.mp4": prompt}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
PY
export PYTHON_BIN=$VBENCH_PYTHON
export VBENCH_CACHE_DIR=${VBENCH_CACHE_DIR:-$repository_dir/../../models/VBench}
bash "$vbench_script" "$result_root/vbench_staging/baseline" \
  "$result_root/vbench_baseline" "$result_root/vbench_baseline_prompts.json"
bash "$vbench_script" "$result_root/vbench_staging/candidate" \
  "$result_root/vbench_candidate" "$result_root/vbench_candidate_prompts.json"

"$python_bin" "$experiment_dir/build_summary.py" \
  --performance "$result_root/performance/summary.json" \
  --metrics "$result_root/video_metrics/summary.json" \
  --baseline-vbench "$result_root/vbench_baseline/vbench_custom_aggregate_scores.json" \
  --candidate-vbench "$result_root/vbench_candidate/vbench_custom_aggregate_scores.json" \
  --output "$result_root/summary.json"

echo "smoke validation complete: $result_root"
