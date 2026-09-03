# Timing-scope comparison

This CPU-only audit compares the two Wan2.2 latency definitions on the same
archived inference runs:

- legacy headline: `inference_compute_elapsed_seconds`;
- current headline-equivalent scope: `generation_wall_elapsed_seconds`, which
  covers the complete `WanT2V.generate()` call and therefore includes model
  transfers/offload performed inside `generate()`.

The archived logs also contain
`inference_weight_transfer_elapsed_seconds`, allowing the wall-time difference
to be decomposed into transfer time and the small remaining host-side span.

`compare_timing_scopes.py` validates all baseline/candidate logs, emits
per-run and per-threshold tables, recomputes matched ratio-of-sums speedups, and
linearly interpolates the threshold implied by requested speedups. It uses only
the Python standard library.

Example:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 python compare_timing_scopes.py \
  --source-root /all/yiran07-disk3/huteng_data/exp/wan22_seacache_vbench30_9thresholds_gpu0123_queued_20260825_005057 \
  --output-dir /all/yiran07-disk3/huteng_data/exp/wan22_seacache_timing_scope_comparison_v1_YYYYMMDD_HHMMSS
```

The full-wall measurement in the archive has the same inclusion boundary as
the current `pipeline_generate_wall_seconds`, but it does not measure any
additional overhead introduced by the current component/DiT instrumentation.
That limitation is stated in every generated report.
