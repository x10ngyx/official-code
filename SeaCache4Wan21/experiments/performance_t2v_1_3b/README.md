# Wan2.1 T2V-1.3B performance measurement

This experiment follows the TeaCache performance-counting convention for the
locked `832x480`, 81-frame, 50-step Wan2.1 T2V-1.3B protocol.

- `profile_calflops.py` profiles one real-shape full DiT forward and the
  always-on/no-Transformer-block path with Calflops 0.3.2, then adds the dense
  FlashAttention core analytically because the custom CUDA kernel is not
  visible to Calflops. It also profiles two fixed-shape UMT5 encoder forwards
  and one VAE decode per output video.
- `aggregate_performance.py` consumes matched baseline/SeaCache timing
  directories and weights those two profiles by each call's actual
  `blocks_executed` trace. Latency and FLOPs speedups are reported from matched
  runs; the FLOPs speedup is a ratio of sums.

Formal outputs must be below `/all/yiran07-disk3/huteng_data/exp` and linked
from `experiment_results/`.

## Profile once

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python profile_calflops.py \
  --wan21-root /path/to/Wan2.1 \
  --checkpoint-dir /path/to/models/Wan2.1-T2V-1.3B \
  --calflops-source /path/to/calculate-flops.pytorch \
  --output /all/yiran07-disk3/huteng_data/exp/<run>/calflops_profile.json
```

The source checkout must be
`calculate-flops.pytorch@027e89a24daf23ee7ed79ca4abee3fb59b5b23cd`;
an installed `calflops==0.3.2` is also accepted.

## Aggregate matched traces

```bash
python aggregate_performance.py \
  --baseline-dir /all/yiran07-disk3/huteng_data/exp/<run>/baseline \
  --seacache-dir /all/yiran07-disk3/huteng_data/exp/<run>/seacache \
  --calflops-profile /all/yiran07-disk3/huteng_data/exp/<run>/calflops_profile.json \
  --output-dir /all/yiran07-disk3/huteng_data/exp/<run>/performance
```

Each condition directory must contain `timings/*.json` with identical sample
IDs. The default aggregate requires 200 matched VBench samples.

The headline is estimated DiT TFLOPs. T5 encoder and VAE decode TFLOPs are
recorded separately; scheduler, MP4 export, and SeaCache's SEA
filter/controller/residual addition remain outside these component counts.
None is labeled complete-method or end-to-end FLOPs.
