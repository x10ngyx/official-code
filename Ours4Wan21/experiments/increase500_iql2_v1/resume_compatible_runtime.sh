#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
compat_lib=/mnt/hdd/xiongyuxiang/tmp/exp/ours21_increase500_iql2_v1/runtime/nvidia580_173/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH="$compat_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python -u /home/star/xiongyuxiang/tmp/work/official-code/Ours4Wan21/experiments/increase500_iql2_v1/pipeline.py run
