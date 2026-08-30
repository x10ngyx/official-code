# Wan2.1 fixed-threshold SeaCache collection v1

This experiment samples 1,000 prompts uniformly without replacement from the
same frozen OpenVidHD balanced-5000 pool used by the random-threshold pipeline.
For every selected prompt it samples three **distinct** fixed thresholds from
the frozen local Wan2.2 SeaCache list:

`0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.60 0.70`

Threshold sampling is uniform without replacement inside each prompt. The
prompt-selection seed (`2026073001`) and manifest seed (`20260722`) match the
random-data pipeline. The prompt-level split is 800/100/100 train/val/test,
giving 3,000 candidates split 2,400/300/300. Four deterministic shards contain
250 baselines and 750 candidates each.

All other contracts are shared with `random_threshold_collection_v1`: the
frozen Wan2.1 inference protocol, one full-compute baseline per prompt, 50
input latents, cond/uncond SeaCache branch traces, complete generate and
T5/DiT/VAE timing, DiT/T5/VAE TFLOPs, canonical PSNR/SSIM/LPIPS, custom-input
VBench, prefix publication, and archive audit. Fixed-threshold rows deliberately
store `target_speedup=null` and `q=null`; measured speedup remains an outcome.

Run `launch_4gpu.sh <phase>` in this order:

1. `preflight`
2. `plan`
3. `profile` (or point `FLOPS_PROFILE` at a compatible existing profile)
4. `baselines`
5. `candidates`
6. `publish` (optional intermediate snapshot)
7. `vbench` (optional standalone scoring)
8. `finalize`

Set `RUN_ID`, `WAN21_ROOT`, `CHECKPOINT_DIR`, `EXP_BASE`, metric-cache and
VBench variables as documented by the parent project. Results are written only
under `/all/yiran07-disk3/huteng_data/exp` (or explicit `EXP_BASE`) and exposed
through a symlink in `experiment_results/`.
