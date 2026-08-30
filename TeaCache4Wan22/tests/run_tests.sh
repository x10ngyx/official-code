#!/usr/bin/env bash
set -euo pipefail

test_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$test_dir/.." && pwd)

if [[ -n "${WAN22_PYTHON:-}" ]]; then
  python_cmd=("$WAN22_PYTHON")
elif python3 -c 'import torch' >/dev/null 2>&1; then
  python_cmd=(python3)
elif command -v conda >/dev/null 2>&1 \
  && conda run -n wan2.2 python -c 'import torch' >/dev/null 2>&1; then
  python_cmd=(conda run --no-capture-output -n wan2.2 python)
else
  echo "PyTorch Python not found. Activate conda env 'wan2.2' or set WAN22_PYTHON." >&2
  exit 2
fi

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${python_cmd[@]}" -m unittest discover -s "$test_dir" -p 'test_*.py' -v
"${python_cmd[@]}" -m py_compile \
  "$project_dir/runtime/teacache.py" \
  "$project_dir/runtime/inference_timing.py" \
  "$project_dir/scripts/validate_prepared_tree.py" \
  "$project_dir/scripts/write_run_manifest.py" \
  "$project_dir/scripts/package_coefficients.py" \
  "$project_dir/scripts/compare_runs.py" \
  "$project_dir/experiments/performance_t2v_a14b/profile_calflops.py" \
  "$project_dir/experiments/performance_t2v_a14b/aggregate_performance.py" \
  "$project_dir/experiments/threshold_scan_vbench8_t2v_a14b/plan_scan.py" \
  "$project_dir/experiments/threshold_scan_vbench8_t2v_a14b/finalize_scan.py" \
  "$project_dir/scripts/validate_calibration_source_equivalence.py" \
  "$project_dir/experiments/fit_t2vcompbench70_wan22_t2v_a14b/prepare_prompts.py" \
  "$project_dir/experiments/fit_t2vcompbench70_wan22_t2v_a14b/collect_distances.py" \
  "$project_dir/experiments/fit_t2vcompbench70_wan22_t2v_a14b/fit_polynomials.py" \
  "$project_dir/experiments/fit_t2vcompbench70_wan22_t2v_a14b/audit_fit.py" \
  "$project_dir/experiments/fit_t2vcompbench70_wan22_t2v_a14b/watch_and_fit.py"
bash -n "$project_dir/scripts/prepare_wan22.sh"
bash -n "$project_dir/scripts/run_t2v_a14b.sh"
bash -n "$project_dir/experiments/fit_t2vcompbench70_wan22_t2v_a14b/launch_shards.sh"
bash -n "$project_dir/experiments/smoke_t2v_a14b/run_pair.sh"
bash -n "$project_dir/experiments/smoke_t2v_a14b/finalize_pair.sh"
bash -n "$project_dir/experiments/threshold_scan_vbench8_t2v_a14b/run_scan.sh"
bash -n "$project_dir/experiments/threshold_scan_vbench8_t2v_a14b/run_worker.sh"

echo "TeaCache4Wan22 tests passed"
