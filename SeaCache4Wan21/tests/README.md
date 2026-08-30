# Tests

Run the CPU-only regression suite from this directory with:

```bash
python -m unittest discover -s tests -v
```

The suite includes 50-step synthetic differential coverage of the corrected
filtered-boundary state machine across both retention modes, multiple
thresholds, and independent cond/uncond trajectories. The SEA filter formula
is independently transcribed from the official Wan2.1 utility.

It also validates the per-DiT-call timing trace and trace-weighted TFLOPs
aggregation used by `experiments/performance_t2v_1_3b/`.
