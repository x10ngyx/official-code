# MagCache VBench200 with a reused baseline

`run_gpu123.sh` launches three independent persistent batch workers:

| GPU | requested speed | R | E | K |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.8x | 0.2 | 0.072 | 2 |
| 2 | 2.4x | 0.2 | 0.199 | 4 |
| 3 | 3.0x | 0.1 | 0.198 | 5 |

Each worker generates all 200 VBench candidates with the fixed Wan2.2 protocol and one persistent model pipeline. The runner validates and links the audited shared baseline instead of generating it. Performance reports use `reported_timings.jsonl`, including the committed GPU0 healthy-card correction. After generation, each target runs PSNR, SSIM, LPIPS, and VBench evaluation on its assigned GPU. All large outputs must be placed under `/all/yiran07-disk3/huteng_data/exp`; the project result index is a symlink.

Use `--dry-run` for a full source, profile, prompt, parameter, and baseline validation without creating output. Use `--resume` to validate and continue an interrupted output directory.

`rebalance_gpu123.sh` converts an in-progress one-GPU-per-target run into a fixed, auditable cross-target assignment. It preserves all completed manifests, archives incomplete attempts, estimates each remaining video's cost from completed measurements, and assigns remaining work to GPU1/2/3 with deterministic longest-processing-time greedy scheduling. Each replacement worker uses one persistent pipeline, one unreported warmup, fresh per-video profiling, and may process all three target parameter sets. The finalizer accepts only validated persistent results from GPU1/2/3 and then evaluates the three completed targets in parallel.

If a quality evaluator is interrupted after generation, resume an individual target with `rebalance_gpu123.py --finalize-target TARGET --evaluation-gpu GPU`. The override changes only the device used by VideoMetrics/VBench and is recorded in the target report; generation provenance and timings remain unchanged. After every target report is complete, `--complete-root` validates their committed hashes before restoring the suite completion marker.
