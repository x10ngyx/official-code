#!/usr/bin/env bash
set -euo pipefail
if [[ $(cat /proc/driver/nvidia/version) == *580.173.02* ]]; then
 export LD_LIBRARY_PATH=/mnt/hdd/xiongyuxiang/tmp/exp/ours21_increase500_iql2_v1/runtime/nvidia580_173/usr/lib/x86_64-linux-gnu
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python -u /home/star/xiongyuxiang/tmp/work/official-code/Ours4Wan21/experiments/increase_vbench20_v1/pipeline.py "$@"
