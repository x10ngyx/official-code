#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RESULT_ROOT SESSION_PREFIX" >&2
  exit 2
fi

result_root=$(readlink -f "$1")
session_prefix=$2
experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$experiment_dir/../.." && pwd)
python_bin=${WAN22_PYTHON:-python}
case "$result_root" in
  /all/yiran07-disk3/huteng_data/exp/*) ;;
  *) echo "RESULT_ROOT must be below /all/yiran07-disk3/huteng_data/exp: $result_root" >&2; exit 2 ;;
esac
result_link="$project_dir/experiment_results/$(basename "$result_root")"
mkdir -p "$project_dir/experiment_results"
if [[ -L $result_link ]]; then
  [[ $(readlink -f "$result_link") == "$result_root" ]] || { echo "result symlink points elsewhere: $result_link" >&2; exit 2; }
elif [[ -e $result_link ]]; then
  echo "result index exists and is not a symlink: $result_link" >&2
  exit 2
else
  ln -s "$result_root" "$result_link"
fi
manifest="$result_root/prompts.jsonl"
mkdir -p "$result_root/logs"

for name in WAN22_SOURCE WAN22_CKPT; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done

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
  command="cd '$experiment_dir' && env CUDA_VISIBLE_DEVICES='$gpu' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 '$python_bin' collect_distances.py --manifest '$manifest' --output-root '$result_root' --wan-repo '$WAN22_SOURCE' --ckpt-dir '$WAN22_CKPT' --shard-index '$gpu' --num-shards 4 >> '$result_root/logs/shard_${gpu}.log' 2>&1"
  tmux new-session -d -s "$session_name" "$command"
  echo "$session_name"
done
