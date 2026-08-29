#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RESULT_ROOT SESSION_PREFIX" >&2
  exit 2
fi

result_root=$(readlink -f "$1")
session_prefix=$2
experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${WAN22_PYTHON:-python}
manifest="$result_root/prompts.jsonl"
mkdir -p "$result_root/logs"

if [[ ! -s "$manifest" ]]; then
  echo "missing manifest: $manifest" >&2
  exit 1
fi

for gpu in 0 1 2 3; do
  session_name="${session_prefix}_g${gpu}"
  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "tmux session already exists: $session_name" >&2
    exit 1
  fi
done

for gpu in 0 1 2 3; do
  session_name="${session_prefix}_g${gpu}"
  command="cd '$experiment_dir' && env CUDA_VISIBLE_DEVICES='$gpu' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 '$python_bin' collect_distances.py --manifest '$manifest' --output-root '$result_root' --shard-index '$gpu' --num-shards 4 >> '$result_root/logs/shard_${gpu}.log' 2>&1"
  tmux new-session -d -s "$session_name" "$command"
  echo "$session_name"
done
