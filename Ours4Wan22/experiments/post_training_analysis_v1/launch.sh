#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=GPU-9dc656f1-c874-1c7b-a8d6-8216d3748cba
exec /home/huteng/yes/envs/wan2.2/bin/python -u /home/huteng/xiongyuxiang/tmp/work/offical-code/Ours4Wan22/experiments/post_training_analysis_v1/analyze.py --root /all/yiran07-disk3/huteng_data/exp/ours22_random2323_cnn_g1_20260911_223523 --wait > /all/yiran07-disk3/huteng_data/exp/ours22_random2323_cnn_g1_20260911_223523/postprocess.console.log 2>&1
