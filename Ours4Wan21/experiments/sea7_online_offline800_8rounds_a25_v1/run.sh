#!/usr/bin/env bash
set -euo pipefail
export WAN22_PYTHON=/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exp_root=/mnt/hdd/xiongyuxiang/tmp/exp
run_dir="$exp_root/ours21_sea7_e328_online_offline800_8rounds_a25_v1"
setup_dir="$exp_root/ours21_sea7_e328_online_offline800_8h_v1_setup"
reference_dir="$exp_root/ours21_online_eval20_from_vbench50_random42_v1"
cd "$project_dir"
case "${1:-run}" in
  prepare)
    exec "$WAN22_PYTHON" online.py prepare \
      --dataset "$exp_root/ours21_random3000_12groups_v1_sea7_cache" \
      --start-checkpoint /mnt/hdd/xiongyuxiang/tmp/models/ours21_random3000_12groups_v1_sea7/checkpoints/epoch_328.pt \
      --train-prompts "$setup_dir/train_baselines/train_prompts.jsonl" \
      --training-bundle "$setup_dir/train_baselines" \
      --eval-prompts "$reference_dir/eval_prompts.jsonl" --evaluation-bundle "$reference_dir" \
      --prompt-registry "$project_dir/data_collection/resources/prompts/openvidhd_balanced_5000.upstream.jsonl" \
      --calibration "$setup_dir/calibration.json" \
      --flops-profile "$exp_root/wan21_seacache_threshold_collection_v1/calflops_profile.json" \
      --wan21-root /mnt/hdd/xiongyuxiang/tmp/data/source/Wan2.1-65386b2 \
      --checkpoint-dir /mnt/hdd/xiongyuxiang/tmp/models/Wan2.1-T2V-1.3B \
      --gpus 0,1,2,3 --rounds 8 --iql-profile aggressive_a2_a3_v1 --output-dir "$run_dir" \
      --weights-dir /mnt/hdd/xiongyuxiang/tmp/models/ours21_sea7_e328_online_offline800_8rounds_a25_v1
    ;;
  run)
    exec "$WAN22_PYTHON" online.py run --run-dir "$run_dir"
    ;;
  *) echo 'Usage: run.sh prepare|run' >&2; exit 2 ;;
esac
