# Experiments

SeaCache4Wan21 uses the shared `VideoMetrics/`, `Vbench200/`, and
`VbenchEvaluation/` projects in the repository. New runnable experiments must
be placed in their own subdirectory and store large outputs externally.

`performance_t2v_1_3b/` contains fixed-protocol component timing, trace-weighted
DiT Calflops, and separately recorded T5/VAE Calflops profiles.

`vbench200_t2v/` orchestrates the full 200-prompt baseline/SeaCache generation,
reuses that locked profiler, runs repository VideoMetrics and VbenchEvaluation,
and emits one auditable report containing time, TFLOPs, PSNR, SSIM, LPIPS, and
Vbench200 subset scores.
