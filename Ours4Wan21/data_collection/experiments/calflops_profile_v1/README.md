# Wan2.1 real-shape Calflops profile

`profile_wan21_dit.py` profiles one Wan2.1-T2V-1.3B DiT forward at the
frozen `832x480 / 81-frame` shape.  It records a full forward and an
always-on (transformer blocks removed) forward with `calflops==0.3.2`, then
adds the dense self/cross-attention core operations that the custom
FlashAttention kernel hides from module hooks. The audited formula is imported
from sibling `../../../CalflopsEvaluation/`; it is not duplicated here.

The output is an operation-count profile.  `TFLOPs` means FLOPs divided by
`1e12`, not hardware throughput.  The collection runtime trace-weights this
profile using the actual block count of all 100 cond/uncond calls. Output must
be under the configured result base (`EXP_BASE`, defaulting to
`/all/yiran07-disk3/huteng_data/exp`) and is never overwritten.

Example:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
OURS4WAN21_EXP_BASE=/remote/exp python profile_wan21_dit.py \
  --wan21-root /path/to/locked/Wan2.1 \
  --checkpoint-dir /path/to/models/Wan2.1-T2V-1.3B \
  --output /remote/exp/<run>/calflops_profile.json
```
