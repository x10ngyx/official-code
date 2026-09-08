# Paired benchmark compatibility entry

`run_benchmark.py` now delegates to the persistent batch implementation in
`../vbench200_t2v/`. Existing single-condition flags remain available; use
`--gpu-ids`, `--resume` and `--defer-evaluation` for batch execution.
See [the batch README](../vbench200_t2v/README.md) for commands, result layout,
shared baseline, timing/FLOPs and VideoMetrics/VBench evaluation.
