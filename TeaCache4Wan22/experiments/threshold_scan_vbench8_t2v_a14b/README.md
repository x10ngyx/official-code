# VBench 16-dimension-coverage threshold scan for Wan2.2 T2V-A14B

This experiment locates TeaCache thresholds nearest `1.8x`, `2.4x`, and
`3.0x` inference speedup under the frozen Wan2.2 protocol. The legacy folder
name is retained for stable paths, but the frozen subset now contains eleven
deterministic Vbench200 prompts covering all 16 VBench dimensions.

## Frozen scan

- Model/protocol: `Wan2.2-T2V-A14B`, `832x480`, 45 frames, 50 DPM++ steps,
  shift 12, low/high CFG `(3,4)`, boundary `.875`, BF16, seed 42.
- Thresholds: `.15, .175, .20, .225, .25, .275, .30, .325, .35, .375,
  .40, .45, .50, .60, .70, .80`.
- Headline latency: CUDA-synchronized `WanT2V.generate()` wall time. Speedup
  is the ratio of summed baseline time to summed candidate time over the same
  eleven prompts.
- Computation: real-shape Calflops 0.3.2 full/always-on DiT profiles plus the
  dense FlashAttention correction, accumulated from each run's actual
  100-call block-execution trace.
- Quality: `rgb_full_reference_v1` PSNR, SSIM, and spatial AlexNet LPIPS
  against each prompt's same-seed no-cache MP4, plus the official
  16-dimension Vbench200 subset aggregate `total_score` (`vbench_score`).

Every prompt stays on one physical GPU for its baseline and all thresholds.
Each worker constructs one persistent WanT2V pipeline, then first runs one
complete no-cache warm-up that is retained for audit but excluded from
results. GPU telemetry is sampled every 30 seconds. This 251 GiB/no-swap host
uses two workers: an observed three-worker model-conversion peak reduced
available memory to 54 GiB, so that warm-up was stopped before any measured
sample.

This is a threshold-location pilot, not a full Vbench200 calibration. A final
threshold must be rechecked on a larger held-out prompt set before being
treated as a benchmark setting.

## End-to-end runner

All outputs must be under `/all/yiran07-disk3/huteng_data/exp`.

```bash
export WAN22_PYTHON=/path/to/wan2.2/bin/python
export WAN22_SOURCE=/path/to/prepared/Wan2.2
export WAN22_CKPT=/path/to/models/Wan2.2-T2V-A14B
export CALFLOPS_SOURCE=/path/to/calculate-flops.pytorch-at-027e89a
export RESULT_ROOT=/all/yiran07-disk3/huteng_data/exp/<scan-run>
export GPU_IDS=1,2
export METRIC_GPU_ID=0
export VBENCH_PYTHON=/path/to/vbench-python
export VBENCH_CACHE_DIR=/path/to/models/VBench
bash run_scan.sh
```

The runner is resumable at completed-manifest granularity. Existing complete
runs are hash-validated before being skipped; incomplete partial files fail
closed and must be inspected rather than silently overwritten. One-time
pipeline initialization and worker progress are recorded separately under
`worker_status/`; per-run speed uses only CUDA-synchronized `generate()` time.

Use `PLAN_ONLY=1 bash run_scan.sh` to validate the frozen prompt/threshold
inputs and print worker assignments without requiring GPUs or writing results.

## Outputs

Raw generation artifacts preserve every MP4, log, timing JSON, decision trace,
and manifest. Threshold-level folders preserve full performance JSON/JSONL and
per-frame/per-video metric CSVs. The root `per_video_summary.csv` provides one
row per prompt and threshold with initialization/generation time, DiT CUDA
time, estimated DiT TFLOPs, full/reuse counts, PSNR, SSIM, and LPIPS.

`target_thresholds.json` reports the nearest measured threshold for each
requested speedup and a bracketed linear interpolation diagnostic. The
nearest measured value is the auditable recommendation; interpolation is not
treated as an observed experiment.
