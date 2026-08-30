# Tests

CPU-only tests cover the immutable OpenVid plans, pending/calibrated mapping
barrier, dynamic-threshold independent-CFG SeaCache controller, direct
differential agreement with the locked corrected `SeaCache4Wan21` controller,
filtered-distance/accumulator runtime persistence, trace-weighted TFLOPs,
inference-only speedup formulas, completion-prefix publication, and launcher
thread limits. They also freeze the 1,000-prompt fixed SeaCache selection, the
Wan2.2-derived nine-threshold grid, three-distinct-threshold sampling, and
250-baseline/750-candidate four-GPU shard contract. The publication contract requires PSNR, SSIM, and LPIPS
per-frame/per-video/full-summary artifacts. Tests also verify the shared `VideoMetrics/`,
`CalflopsEvaluation/`, and parent upstream-lock boundary. Run from
`data_collection/` with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src conda run --no-capture-output -n Wan2.1 \
python -m unittest discover -s tests -v
```
