# runtime

official.py loads unchanged pinned core functions and manages per-video installation/observations. common.py enforces environment, source, model and result conventions. inference_timing.py measures complete generate and T5/DiT/VAE.

`generation.py` provides one fixed native generation call for both single and
persistent workers. `batch.py` owns worker sharding, one-time model loading,
warmup, per-video lifecycle, strict resume and process supervision. `suite.py`
coordinates profiling, generation and resumable quality evaluation. `prompts.py`
validates prompt identity and VBench dimension coverage. `scan.py` executes the
unchanged official gate statements for grid planning and selects target speeds
only from measured complete-generation latency.

Preset scans fix R/K per target and expand exact decimal E grids. Target selection
is restricted to the corresponding measured group, with explicit tolerance and
unmet results. The original generic Cartesian scan remains available.
