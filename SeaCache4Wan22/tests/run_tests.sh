#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
python_bin=${WAN22_PYTHON:-python}

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$project_dir"
"$python_bin" -m unittest discover -s tests -v
"$python_bin" -m py_compile \
  runtime/seacache.py runtime/inference_timing.py \
  scripts/validate_prepared_tree.py scripts/write_run_manifest.py \
  scripts/compare_runs.py \
  experiments/performance_t2v_a14b/profile_calflops.py \
  experiments/performance_t2v_a14b/aggregate_performance.py \
  experiments/vbench200_t2v/generate_vbench200.py \
  experiments/vbench200_t2v/profile_calflops.py \
  experiments/vbench200_t2v/aggregate_performance.py \
  experiments/vbench200_t2v/evaluate_results.py \
  experiments/vbench200_t2v/build_final_report.py \
  experiments/vbench200_t2v/run_vbench200.py
bash -n \
  scripts/prepare_wan22.sh scripts/run_t2v_a14b.sh \
  experiments/vbench200_t2v/launch_threshold_suite_024_038_055.sh
