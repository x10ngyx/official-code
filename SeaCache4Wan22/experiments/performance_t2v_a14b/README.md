# Wan2.2 T2V-A14B performance measurement

This folder reproduces the SeaCache4Wan21 latency/Calflops workflow for the
fixed SeaCache4Wan22 protocol.

- `profile_calflops.py` independently profiles high/low x cond/uncond DiT
  full-forward and always-on/no-block paths at the real `832x480`, 45-frame
  tensor shape, checks the expected branch/stage equivalence, and adds an
  analytical dense FlashAttention-core correction to Calflops 0.3.2. It also
  profiles two fixed-shape UMT5 encoder forwards and one VAE decode per video.
- `aggregate_performance.py` validates matched baseline/SeaCache manifests,
  consumes the Wan2.1-style timing trace's 100 actual DiT call rows, and
  accumulates per-video TFLOPs from the measured full/reuse path.

Formal outputs must be placed below `/all/yiran07-disk3/huteng_data/exp` and
linked from `SeaCache4Wan22/experiment_results/`; this source folder contains
no model weights or generated results.

## 1. Profile high/low DiT paths once

Use a prepared Wan2.2 tree and the pinned Calflops source checkout. The shared
`wan2.2` environment does not need to be modified:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python profile_calflops.py \
  --wan22-root /path/to/prepared-Wan2.2 \
  --checkpoint-dir /path/to/models/Wan2.2-T2V-A14B \
  --calflops-source /path/to/calculate-flops.pytorch \
  --output /all/yiran07-disk3/huteng_data/exp/<run>/calflops_profile.json
```

The Calflops checkout must be commit
`027e89a24daf23ee7ed79ca4abee3fb59b5b23cd`; an installed
`calflops==0.3.2` is also accepted.

## 2. Aggregate matched runs

```bash
python aggregate_performance.py \
  --baseline-manifest /path/to/prompt001_baseline.manifest.json \
  --seacache-manifest /path/to/prompt001_seacache.manifest.json \
  --calflops-profile /all/yiran07-disk3/huteng_data/exp/<run>/calflops_profile.json \
  --output-dir /all/yiran07-disk3/huteng_data/exp/<run>/performance
```

Repeat both manifest options in matched order for multiple prompts. The
headline latency speedup and DiT FLOPs speedup are both ratios of sums.

Latency uses CUDA-synchronized `pipeline_generate_wall_seconds`: text encoding
+ denoising/cache/CFG/scheduler + cache-state release + transfers/offload
inside `generate()` + VAE decode. It excludes pipeline construction, MP4
export, file I/O, and evaluation. The Calflops headline covers the DiT forward
path; T5 encoder and VAE decode TFLOPs are saved separately. The SeaCache SEA
filter, controller, residual addition, scheduler, and export are outside these
component counts. The DiT headline must therefore be labeled estimated DiT
TFLOPs, not complete-method or end-to-end FLOPs.
