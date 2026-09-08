# Paired benchmark

`run_benchmark.py` is a compatibility entry to the persistent paired suite, sharing arguments with `../vbench200_t2v/run_vbench200.py`. It generates a same-GPU baseline and DiCache candidate per prompt, then reports component performance and VideoMetrics/VBench scores. `--dry-run` prints the plan; `--defer-evaluation` stops at quality pending; `--resume` checks completed hashes before continuing.
