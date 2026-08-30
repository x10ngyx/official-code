# Tests

CPU-only tests cover the immutable OpenVid plan, pending/calibrated mapping
barrier, dynamic-threshold independent-CFG SeaCache controller, direct
differential agreement with the locked corrected `SeaCache4Wan21` controller,
filtered-distance/accumulator runtime persistence, trace-weighted TFLOPs,
inference-only speedup formulas, completion-prefix publication, and launcher
thread limits. The publication contract requires PSNR, SSIM, and LPIPS
per-frame/per-video/full-summary artifacts. Tests also verify the shared `VideoMetrics/`,
`CalflopsEvaluation/`, and parent upstream-lock boundary. Run from
`data_collection/` with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src conda run --no-capture-output -n Wan2.1 \
python -m unittest discover -s tests -v
```
