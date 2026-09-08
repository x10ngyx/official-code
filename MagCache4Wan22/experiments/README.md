# experiments

performance_t2v_a14b/ profiles and aggregates component FLOPs/time. paired_benchmark/ generates matching baselines/candidates and computes all required quality scores. smoke_official_core/ compares the original and adapted functions on small CUDA/BF16 WanModel instances. Experiment outputs belong on the external experiment disk.

- `vbench200_t2v/`: persistent paired workers, one model initialization per worker,
  fixed GPU shards, warmup, strict resume and complete performance/quality pipeline.
- `parameter_scan/`: joint official threshold/K/retention grid, 11 fixed prompts,
  shared baseline, exact schedule deduplication and measured 1.8x/2.4x/3.0x selection.
- `targeted_threshold_scan/`: fixed R/K for each target, .001-spaced E grids around
  unmeasured starting points, per-target selection and default relative ±2% tolerance.
- `resume_gpu123_queue/`: waits for the preceding CNN generation and quality pipeline
  to finish before resuming the unchanged three-target scan on GPU1/2/3.
