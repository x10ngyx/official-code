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
python_bin="${WAN22_PYTHON:-$OURS4WAN21_WORKSPACE/data/environments/Wan2.2-conda-env/bin/python}"
[[ -x "$python_bin" ]] || { echo "Set WAN22_PYTHON to the wan2.2 Python interpreter" >&2; exit 2; }
feature_complete="$EXP_BASE/${suite}_features/COMPLETE.json"
feature_session="${FEATURE_TMUX_SESSION:-ours21_features_v1}"
while [[ ! -f "$feature_complete" ]]; do
  if command -v tmux >/dev/null 2>&1 && ! tmux has-session -t "$feature_session" 2>/dev/null; then
    echo "Feature session ended without COMPLETE.json" >&2
    exit 1
  fi
  sleep 30
done
extra=()
[[ "${RESUME_SUITE:-0}" == 1 ]] && extra+=(--resume)
"$python_bin" experiments/latent_groups_v1/run_training_suite.py \
  --collection-root "$COLLECTION_ROOT" --selection-dir "$SELECTION_DIR" \
  --suite-name "$suite" --gpus ${TRAIN_GPU_IDS:-0 1 2 3} "${extra[@]}"
"$python_bin" experiments/latent_groups_v1/analyze_training_suite.py \
  --suite-name "$suite" --output-dir "$EXP_BASE/${suite}_training_readout"
"$python_bin" experiments/latent_groups_v1/build_training_report.py \
  --readout-dir "$EXP_BASE/${suite}_training_readout"
