# Experiments

Use the fixed runner in `scripts/` and the repository's shared
`VideoMetrics/`, `Vbench200/`, and `VbenchEvaluation/` projects. Any new
runnable experiment must live in its own subdirectory and store large outputs
externally.

`performance_t2v_a14b/` contains fixed-protocol component timing, trace-weighted
DiT Calflops, and separately recorded T5/VAE Calflops profiles.

`vbench200_t2v/` orchestrates the full 200-prompt baseline/SeaCache generation,
reuses that locked profiler, runs repository VideoMetrics and VbenchEvaluation,
and emits one auditable report containing time, TFLOPs, PSNR, SSIM, LPIPS, and
Vbench200 subset scores.
