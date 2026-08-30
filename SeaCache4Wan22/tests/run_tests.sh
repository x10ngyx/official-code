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
  experiments/performance_t2v_a14b/aggregate_performance.py
bash -n scripts/prepare_wan22.sh scripts/run_t2v_a14b.sh
