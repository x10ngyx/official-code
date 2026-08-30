# Fixed-protocol integration smoke

This experiment runs one matched no-cache/TeaCache pair through the prepared
Wan2.2 source, validates manifests and traces, computes inference-only
ratio-of-sums speedup, component latency/TFLOPs, canonical RGB
PSNR/SSIM/LPIPS against the no-cache video, and a custom-input VBench score.

It is an integration check, not a threshold recommendation or a formal
quality benchmark. The default TeaCache threshold is `0.10` and the fixed
prompt is:

> Two anthropomorphic cats box on a spotlighted stage.

Prepare the source and coefficient file first, then run on one visible GPU:

```bash
export WAN22_SOURCE=/path/to/prepared-Wan2.2
export WAN22_CKPT=/path/to/Wan2.2-T2V-A14B
export WAN22_PYTHON=/path/to/wan2.2/bin/python
export RESULT_ROOT=/all/yiran07-disk3/huteng_data/exp/teacache4wan22_smoke
export CALFLOPS_SOURCE=/path/to/calculate-flops.pytorch
export VBENCH_PYTHON=/path/to/vbench/bin/python
export VBENCH_CACHE_DIR=/path/to/models/VBench
CUDA_VISIBLE_DEVICES=0 bash experiments/smoke_t2v_a14b/run_pair.sh
```

Override `SMOKE_THRESHOLD`, `SMOKE_PROMPT`, or `TEACACHE_COEFFICIENTS` through
the environment. All generated artifacts remain under `RESULT_ROOT`; the
source project should retain only a symlink in `experiment_results/`.
