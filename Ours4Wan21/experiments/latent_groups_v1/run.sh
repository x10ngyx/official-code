#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
stage="${1:?Usage: run.sh features|cache|train|analyze|vbench|evaluate [all|STATE_MODE]}"
selected="${2:-all}"
: "${EXP_BASE:?Set external experiment root}"
: "${OURS4WAN21_WORKSPACE:?Set remote workspace root}"
suite="${SUITE_NAME:-ours21_random3000_12groups_v1}"
features="$EXP_BASE/${suite}_features"
run_python() { conda run --no-capture-output -n wan2.2 python "$@"; }
if [[ "$stage" == features ]]; then
  : "${COLLECTION_ROOT:?}" "${SELECTION_DIR:?}"
  extra=()
  [[ "${RESUME_FEATURES:-0}" == 1 ]] && extra+=(--resume)
  run_python prepare_features.py --collection-root "$COLLECTION_ROOT" --selection "$SELECTION_DIR/selection.json" --output-dir "$features" --device "${FEATURE_DEVICE:-cpu}" "${extra[@]}"
  exit
fi
modes=(scalar5 sea7 sea7_dynamics_raw_sea128 sea7_cache_update192 sea7_local_drift1024 sea7_spatial_gradient96 sea7_channel_geometry240 sea7_distribution256 sea7_spectral_drift512 sea7_spectral_phase512 sea7_spectral_shape576 sea7_spectral_dynamics1024)
found=0
for mode in "${modes[@]}"; do
  [[ "$selected" == all || "$selected" == "$mode" ]] || continue
  found=1
  cache="$EXP_BASE/${suite}_${mode}_cache"
  training="$EXP_BASE/${suite}_${mode}_train"
  weights="$OURS4WAN21_WORKSPACE/models/${suite}_${mode}"
  analysis="$EXP_BASE/${suite}_${mode}_analysis"
  case "$stage" in
    cache)
      : "${COLLECTION_ROOT:?}" "${SELECTION_DIR:?}"
      extra=()
      [[ "$mode" == scalar5 || "$mode" == sea7 ]] || extra+=(--feature-cache "$features")
      run_python prepare_data.py --collection-root "$COLLECTION_ROOT" --selection "$SELECTION_DIR/selection.json" --state-mode "$mode" --output-dir "$cache" "${extra[@]}" ;;
    train)
      run_python train.py --dataset "$cache" --state-mode "$mode" --output-dir "$training" --checkpoint-dir "$weights" --device "${TRAIN_DEVICE:-cuda}" ;;
    analyze)
      run_python analyze_training.py --training-result "$training" --dataset "$cache" --checkpoint-dir "$weights" --output-dir "$analysis" --device "${ANALYSIS_DEVICE:-cuda}" ;;
    vbench|evaluate)
      : "${SKIP_BUDGET:?Set explicit K, e.g. 25; not an assumed speedup}" "${FLOPS_PROFILE:?}"
      candidate="$EXP_BASE/${suite}_${mode}_vbench200_K${SKIP_BUDGET}"
      if [[ "$stage" == vbench ]]; then
        : "${WAN21_ROOT:?}" "${CHECKPOINT_DIR:?}" "${OFFICIAL_CODE:?}"
        run_python generate.py --wan21-root "$WAN21_ROOT" --checkpoint-dir "$CHECKPOINT_DIR" --policy-checkpoint "$analysis/selected_model.pt" --state-mode "$mode" --skip-budget "$SKIP_BUDGET" --prompts "$OFFICIAL_CODE/Vbench200/prompts.jsonl" --flops-profile "$FLOPS_PROFILE" --output-dir "$candidate"
      else
        : "${VBENCH_BASELINE:?Use same protocol/prompts/physical GPU baseline}"
        run_python evaluate.py summarize --baseline-dir "$VBENCH_BASELINE" --candidate-dir "$candidate" --profile "$FLOPS_PROFILE"
        run_python evaluate.py evaluate --baseline-dir "$VBENCH_BASELINE" --candidate-dir "$candidate"
      fi ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
  esac
done
[[ "$found" == 1 ]] || { echo "Unknown state mode: $selected" >&2; exit 2; }
