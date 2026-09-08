#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${WAN22_PYTHON:-python}
exec "$python_bin" "$script_dir/../targeted_threshold_scan/run_scan.py" \
  --gpu-ids 1 2 3 --worker-launch-wave-size 1 \
  --presets "$script_dir/presets.json" "$@"
