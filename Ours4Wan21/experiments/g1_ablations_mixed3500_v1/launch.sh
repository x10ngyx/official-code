#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/mnt/hdd/xiongyuxiang/tmp/exp/ours21_increase500_iql2_v1/runtime/nvidia580_173/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python_bin=/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$python_bin" -u "$script_dir/pipeline.py" run
