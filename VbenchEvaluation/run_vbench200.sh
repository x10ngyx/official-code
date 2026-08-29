#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 VIDEOS_DIR WORK_DIR [EXPECTED_SEEDS_PER_PROMPT]" >&2
  exit 2
fi

VIDEOS_DIR=$1
WORK_DIR=$2
EXPECTED_SEEDS=${3:-1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

export VBENCH_CACHE_DIR=${VBENCH_CACHE_DIR:-"$REPOSITORY_ROOT/models/VBench"}
export TORCH_HOME=${TORCH_HOME:-"$VBENCH_CACHE_DIR/torch"}
export HF_HOME=${HF_HOME:-"$VBENCH_CACHE_DIR/huggingface"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$VBENCH_CACHE_DIR/xdg"}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$WORK_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_videos.py" \
  --videos-dir "$VIDEOS_DIR" \
  --staging-dir "$WORK_DIR/staged_videos" \
  --manifest "$WORK_DIR/staging_manifest.json" \
  --expected-seeds "$EXPECTED_SEEDS"

"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_vbench.py" \
  --videos-dir "$WORK_DIR/staged_videos" \
  --output-dir "$WORK_DIR/scores" \
  --load-ckpt-from-local

"$PYTHON_BIN" "$SCRIPT_DIR/aggregate_vbench_scores.py" \
  --score-dir "$WORK_DIR/scores" \
  --output "$WORK_DIR/vbench200_aggregate_scores.json"
