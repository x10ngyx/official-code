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
# Five-prompt Ours comparison

`vbench5_ours_comparison_v1/` adds fixed-threshold SeaCache to the same five prompts
and three nominal speed targets used by the Ours twelve-group diagnostic.
No VBench scoring or adaptive threshold search is performed.

- `linear_increase_matched_v1/`：5组线性递增与标定固定threshold，4prompt配对40条正式轨迹；已完成三项质量及计量，无VBench报告。

- `increase_e391_trace_comparison_v1/`：只读绘制increase/e391同4prompt每档的实测trace，按skip数量排序，核对实际blocks。
