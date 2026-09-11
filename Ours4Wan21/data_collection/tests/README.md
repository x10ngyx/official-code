# Tests

CPU-only tests cover the immutable OpenVid plans, pending/calibrated mapping
barrier, dynamic-threshold independent-CFG SeaCache controller, direct
differential agreement with the locked corrected `SeaCache4Wan21` controller,
filtered-distance/accumulator runtime persistence, trace-weighted TFLOPs,
inference-only speedup formulas, completion-prefix publication, and launcher
thread limits. They also freeze the 1,000-prompt fixed SeaCache selection, the
Wan2.2-derived nine-threshold grid, three-distinct-threshold sampling, and
250-baseline/750-candidate four-GPU shard contract. The no-offload runtime must
place T5 on the visible GPU before the captured inference span. The publication
contract also enforces that the randomized plan's first 1,000-prompt release
has an 800/100/100 split, balanced 250/750 GPU shards, full prompt-category
coverage, and bounded categorical/uniform-distribution drift versus the full
3,000/9,000 plan. The candidate artifact contract requires PSNR, SSIM, and LPIPS
per-frame/per-video/full-summary artifacts. Tests also verify the shared `VideoMetrics/`,
`CalflopsEvaluation/`, and parent upstream-lock boundary. Run from
`data_collection/` with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src conda run --no-capture-output -n wan2.2 \
python -m unittest discover -s tests -v
```
