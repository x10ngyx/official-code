# VideoMetrics

Reproducible full-reference PSNR, SSIM, and LPIPS evaluation for videos.
The public project, Python package, CLI, and protocol use method-neutral names.
Formula provenance, source hashes, and compatibility decisions are recorded in
`upstream_lock.json` and `NOTICE.md`.

This package evaluates how closely a cached result matches the same-seed
full-compute model output. It does not replace reference-free generation
quality evaluation such as VBench.

## Frozen protocol: `rgb_full_reference_v1`

All methods being compared must use the same model checkpoint, prompt, seed,
scheduler, number of denoising steps, CFG, resolution, frame count, and export
settings. The full-compute/no-cache video is the reference.

| Item | Frozen definition |
| --- | --- |
| Decode | `imageio` FFmpeg backend to RGB8, then float32 RGB `[0,1]` |
| Alignment | Frame `t` against frame `t`; identical `T,C,H,W` required |
| PSNR | Per-frame RGB all-channel MSE; `20 log10(1/sqrt(MSE))`; MSE `<1e-10` becomes `100 dB` |
| SSIM | Per-frame RGB channel mean; `11x11` Gaussian, sigma `1.5`, valid crop, `K1=.01`, `K2=.03` |
| LPIPS | `lpips==0.1.4`, AlexNet, model version `0.1`, RGB `[-1,1]`, `spatial=True`, spatial-map mean |
| Aggregate | Mean over frames within each video, then equal-weight mean over videos |

Higher PSNR/SSIM is better; lower LPIPS is better. PSNR values capped at
`100 dB` are retained in the mean and counted. No perfect frame is excluded.

## Why these metric definitions

### Evaluation target

This protocol measures **paired fidelity**: how closely a cached or otherwise
accelerated output reproduces the full-compute output produced with the same
prompt, seed, model, and sampling configuration. The three metrics are kept
together because they expose different error types:

- PSNR measures absolute pixel reconstruction error.
- SSIM measures local luminance, contrast, and structural similarity.
- LPIPS measures deep-feature perceptual similarity and is less tied to exact
  pixel correspondence than PSNR or SSIM.

None of the three is treated as a standalone measure of general video quality.
They do not directly score prompt alignment, motion realism, temporal
flickering, or aesthetics; use VBench or other temporal/reference-free metrics
for those properties.

### Decode and alignment

Metrics are computed on the final exported video rather than an internal model
tensor, so both sides are decoded through one fixed ImageIO/FFmpeg RGB8 path.
This makes the measured object explicit and reproducible, but also means that
reference and candidate videos must use identical encoder and export settings.
Frame `t` is compared only with frame `t`; shape mismatches fail instead of
being hidden by resizing, cropping, padding, or truncation.

The current implementation validates decoded `T,C,H,W`, but does not yet
validate FPS, duration, time base, or timestamps. Inputs must therefore be
produced by the same fixed-FPS pipeline.

### PSNR choice

The protocol uses full-RGB, all-channel, per-frame MSE because its target is the
rendered RGB output used by text-to-video cache evaluations. This avoids an
implicit dependency on a particular YUV conversion or luma/chroma weighting.
It is intentionally named `psnr_rgb_db`; it is not interchangeable with
Y-PSNR, weighted YUV PSNR, or a single PSNR computed from pooled whole-video
MSE.

Each frame is converted to dB first, then frame scores are averaged within the
video. This preserves the locked cache-evaluation convention and makes
per-frame degradation inspectable. Perfect or numerically identical frames
would have infinite mathematical PSNR; the `100 dB` cap is retained as a
finite compatibility and serialization rule, and capped-frame counts are
reported explicitly.

### SSIM choice

SSIM follows the classical local-window construction: an `11x11` Gaussian
window with sigma `1.5`, `K1=.01`, and `K2=.03`, evaluated on each RGB channel
and then averaged. The Gaussian window and constants follow the standard SSIM
formulation, while RGB channel averaging is made explicit because some video
compression evaluations report luma-only SSIM instead. Results from those
different color-domain definitions must not be mixed.

### LPIPS choice

LPIPS uses the official `lpips==0.1.4` implementation, the AlexNet backbone,
model version `0.1`, learned perceptual calibration weights, and RGB inputs
mapped from `[0,1]` to `[-1,1]`. AlexNet is the official package's standard
forward-metric choice and is also the backbone used by the locked upstream
cache evaluator.

The `spatial=True` setting requires special care. It is an official LPIPS mode,
but the library constructor defaults to `spatial=False`:

- `spatial=False` averages each calibrated feature-difference map at its native
  resolution and then sums the resulting layer scalars, returning `N,1,1,1`.
- `spatial=True` bilinearly upsamples every layer map to input resolution and
  sums the maps, returning `N,1,H,W`; this protocol then averages that spatial
  map to one value per frame.

The two scalar results are normally very close, but are not guaranteed to be
identical because bilinear upsampling does not preserve the global mean for
every input/output size. `spatial=True` followed by map averaging is retained
to reproduce the locked upstream call, and the output name
`lpips_alex_v0_1_spatial` records this non-default choice. It must not be mixed
silently with default scalar LPIPS results. See the
[official LPIPS implementation](https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py)
and the
[locked upstream LPIPS call](https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/eval/teacache/common_metrics/calculate_lpips.py).

### Aggregation choice

The aggregation hierarchy is frame mean within each video, followed by an
equal-weight mean over videos. Equal video weight prevents results from being
determined by evaluation batch boundaries and avoids overweighting a short
final batch. It is also equivalent to equal-prompt aggregation when every
prompt has one seed, or the same number of seeds. A dataset with unequal seed
counts per prompt requires an explicit seed-mean-then-prompt-mean hierarchy and
must not use the directory aggregate as if it were prompt-balanced.

For reporting, use the exact names `psnr_rgb_db`, `ssim_rgb`, and
`lpips_alex_v0_1_spatial`, together with protocol ID
`rgb_full_reference_v1`. Results produced by different color spaces, LPIPS
spatial modes, preprocessing, alignment, or aggregation rules belong to
different protocols.

## Formula provenance and orchestration corrections

The metric kernels are numerically aligned with the locked upstream evaluation
formulas. Dataset orchestration makes three audited corrections relative to
that evaluator:

1. TeaCache first averages each 16-video batch and then averages batch means.
   Its final short batch is therefore overweighted. This package stores every
   video mean and gives every video equal weight.
2. Frame count and spatial shape must match exactly. The evaluator never
   resizes, crops, truncates, or pads an input.
3. Pair order, per-frame values, per-video values, input SHA256 hashes,
   dependency versions, and aggregation are saved explicitly.

These corrections change orchestration, not the locked metric definitions.

## Environment

The project environment is `wan2.2`. LPIPS uses AlexNet weights; in this
workspace they remain under `models/torch-cache/` as required by the project.
The CLI detects that cache automatically unless `TORCH_HOME` or
`--model-cache` is supplied.

```bash
conda activate wan2.2
pip install -r requirements.txt
```

Every NumPy/BLAS evaluation process must explicitly limit its thread pools.
The supplied shell entry point does this automatically:

```bash
bash run_evaluation.sh --help
```

## Evaluate one video pair

Write results to the external experiment root, not to this source directory:

```bash
conda activate wan2.2
bash run_evaluation.sh \
  --reference /path/to/full_compute.mp4 \
  --candidate /path/to/cached.mp4 \
  --video-id sample_001 \
  --expected-frames 45 \
  --device cuda:0 \
  --output-dir /path/to/experiment-results/sample_001_video_metrics
```

For CPU-only LPIPS, use `--device cpu`. `--device auto` selects CUDA when it
is available. `--metrics psnr ssim` can be used when LPIPS is not required.

## Evaluate matching directories

Reference and candidate directories must contain exactly the same filenames.
Files are paired and processed in sorted filename order.

```bash
conda activate wan2.2
bash run_evaluation.sh \
  --reference-dir /path/to/references \
  --candidate-dir /path/to/candidates \
  --extension .mp4 \
  --expected-frames 45 \
  --device cuda:0 \
  --output-dir /path/to/experiment-results/vbench200_video_metrics
```

The output directory contains:

- `per_frame.csv`: one row per aligned frame and all selected metric values.
- `per_video.csv`: equal-weight video means, dispersion, hashes, dimensions,
  capped/exact frame counts, and evaluation-only timing.
- `summary.json`: dataset aggregates, full protocol lock, software versions,
  thread settings, decoder version, LPIPS device, and output schema version.

Metric/decode timing is recorded separately and must never be included in
inference speedup.

## Standalone PSNR entry point

The repository includes a standalone compatibility entry point:

```bash
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python compute_psnr.py \
  --reference /path/to/full_compute.mp4 \
  --candidate /path/to/cached.mp4 \
  --output /path/to/psnr.json
```

## Tests

```bash
conda activate wan2.2
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
TORCH_HOME=/path/to/torch-cache \
python -m unittest discover -s tests -p 'test_*.py' -v
```

The tests compare PSNR and SSIM directly against the locked upstream formulas,
compare batched LPIPS against locked per-frame spatial-map calls, exercise real
MP4 decoding, validate the combined CLI, and verify both protocols exposed by
the repository-wide PSNR compatibility entry point.

## Files

- `video_metrics/core.py`: frozen metric kernels and reusable LPIPS model.
- `video_metrics/video.py`: strict ImageIO/FFmpeg RGB decoder.
- `video_metrics/evaluator.py`: pairing, hierarchical aggregation, and artifacts.
- `video_metrics/compat.py`: `work/compute_psnr.py` compatible JSON contract.
- `evaluate.py` / `run_evaluation.sh`: source-tree entry points.
- `compute_psnr.py`: standalone PSNR-only entry point.
- `tests/`: numerical and end-to-end regression tests.
- `upstream_lock.json`: upstream commit, hashes, and compatibility boundary.
- `NOTICE.md`: upstream attribution and license notice.
