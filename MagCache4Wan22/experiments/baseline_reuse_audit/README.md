# Cross-method baseline reuse audit

`audit_baselines.py` checks the existing MagCache GPU1/2/3 baseline videos, the independently generated historical TeaCache eight-prompt baseline, and the shared SeaCache/TeaCache formal 200-prompt baseline. It verifies actual MP4 bytes/SHA256, prompts, seed, fixed protocol, source references, all 200 full-compute timing records, and 11 video headers. It does not run inference or modify the generation runtime.

The output contains a report, a per-prompt comparison, a frozen 200-video reuse manifest, and symlink references to the shared baseline and the 11-prompt subset. Historical timing is kept separate from the MagCache calibration timing; the report quantifies the effect of changing the denominator without treating it as a new benchmark.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/huteng/yes/envs/wan2.2/bin/python experiments/baseline_reuse_audit/audit_baselines.py \
  --output-dir /all/yiran07-disk3/huteng_data/exp/magcache4wan22_baseline_reuse_audit_20260906
```
