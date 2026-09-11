#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
: "${EXP_BASE:?Set external experiment root}"
: "${OURS4WAN21_WORKSPACE:?Set workspace root}"
: "${COLLECTION_ROOT:?Set completed random collection root}"
: "${SELECTION_DIR:?Set frozen selection directory}"
suite="${SUITE_NAME:-ours21_random3000_12groups_v1}"
features="$EXP_BASE/${suite}_features"
python_bin="${WAN22_PYTHON:-$OURS4WAN21_WORKSPACE/data/environments/Wan2.2-conda-env/bin/python}"
[[ -x "$python_bin" ]] || { echo "Set WAN22_PYTHON to the wan2.2 Python interpreter" >&2; exit 2; }
read -r -a gpus <<< "${FEATURE_GPU_IDS:-0 1 2 3}"
num_workers="${#gpus[@]}"
(( num_workers > 0 )) || { echo "FEATURE_GPU_IDS must name at least one GPU" >&2; exit 2; }
common=(prepare_features.py --collection-root "$COLLECTION_ROOT" --selection "$SELECTION_DIR/selection.json" --output-dir "$features" --num-workers "$num_workers")
if [[ "${RESUME_FEATURES:-0}" != 1 ]]; then
  "$python_bin" "${common[@]}" --initialize-only
fi
pids=()
for ((worker=0; worker<num_workers; worker++)); do
  CUDA_VISIBLE_DEVICES="${gpus[$worker]}" "$python_bin" "${common[@]}" \
    --resume --worker-index "$worker" --device cuda \
    >"$features/workers/worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
(( failed == 0 )) || { echo "At least one feature worker failed; inspect $features/workers/*.log" >&2; exit 1; }
"$python_bin" "${common[@]}" --resume --finalize-only
