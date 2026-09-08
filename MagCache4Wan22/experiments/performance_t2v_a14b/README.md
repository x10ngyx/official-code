# Component performance

`profile_calflops.py` profiles the original MagCache forward with full blocks
and an empty block list (embeddings/head-only cost), using real A14B shapes.
Each profiling invocation freshly initializes the official method at a forced
full call. High/low and cond/uncond are measured separately; the profiling
controller never selects a cached call. Dense FlashAttention-core operations
are added explicitly. T5 and VAE use shared ComponentMetrics profiling.

`aggregate_performance.py` weights these costs by the actual per-CFG-call
block execution trace. It checks source, model protocol, paired prompts,
artifact hashes, cache trace, measured timing and required component metrics.

```bash
conda activate wan2.2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python profile_calflops.py \
  --wan22-root /path/to/MagCache4Wan22/build/Wan2.2-42bf4cf \
  --checkpoint-dir /path/to/workspace/models/Wan2.2-T2V-A14B \
  --output /all/yiran07-disk3/huteng_data/exp/magcache_profile/calflops.json

python aggregate_performance.py \
  --baseline-manifest /path/to/baseline/manifest.json \
  --magcache-manifest /path/to/magcache/manifest.json \
  --calflops-profile /path/to/profile/calflops.json \
  --output /all/yiran07-disk3/huteng_data/exp/magcache_comparison/performance.json
```

Both manifest flags may be repeated for matching prompt sets. The profile
requires locked `calflops==0.3.2`, or `--calflops-source` pointing at its locked
source checkout. Full profile, source preparation, and run must share the same
source manifest. These scripts do not constitute a GPU performance result
until executed on real model weights.
