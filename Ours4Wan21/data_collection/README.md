# Ours4Wan21 behavior-data collection

This subproject creates Wan2.1-T2V-1.3B behavior trajectories by randomizing
the per-denoising-step threshold supplied to the clean SeaCache gate.  It does
not directly randomize or force `reuse/recompute`; those actions remain real
environment outcomes from the native gate state.

It also contains a matched fixed-threshold SeaCache pipeline. That pipeline
selects 1,000 prompts from the same 5,000-prompt pool and samples three
distinct thresholds per prompt from the frozen local Wan2.2 grid
`0.15/0.20/0.25/0.30/0.35/0.40/0.50/0.60/0.70`. It produces 1,000 shared
baselines and 3,000 candidates. Everything after manifest construction uses
the same collection, metric, timing, TFLOPs, VBench, publication, and audit
contracts as randomized behavior data.

The transfer unit is the complete sibling `offical-code/` tree. This subproject
bundles only its unique frozen prompt snapshot, collection code, tests and
launcher; canonical PSNR/SSIM/LPIPS and manual FLOP formulas are reused from
sibling `VideoMetrics/` and `CalflopsEvaluation/`, while notices and upstream
locks are owned by `Ours4Wan21/`. Model weights, the exact locked Wan2.1 checkout, CUDA
environment, external result disk, and the not-yet-fitted threshold mapping
remain explicit external inputs. See `REMOTE_DEPLOYMENT.md`.

## Frozen protocol

- Model: Wan2.1-T2V-1.3B, locked Wan2.1 commit
  `65386b2e03c490796eede31b0325a6a595cc684e`.
- Generation: `832x480`, 81 frames, 16 fps, 50-step UniPC, shift 5, CFG 5,
  seed 42, batch 1, BF16 DiT, one 48GB GPU, `offload_model=False`, and
  `t5_cpu=False`. All model components stay on the GPU; no offload is allowed.
- Prompt pool: immutable OpenVidHD balanced-5000 snapshot.  The plan samples
  3,000 prompts uniformly without replacement and assigns an 80/10/10
  prompt-level train/val/test split.  It retains the established prompt
  selection seed `2026073001` and random-manifest seed `20260722`.
- Random trajectories: three per selected prompt (`9,000` total).  Target
  speedup is sampled independently from `Uniform[1.50, 3.50]`; perturbation
  strength is sampled independently from `Uniform[0.20, 1.00]`.
- The target-speedup to mean-threshold mapping is intentionally absent until
  a compatible calibration experiment is fitted.  A plan manifest can be
  built now, but runnable candidate materialization and GPU candidate launch
  fail closed while the mapping status is `pending`.

## Two-stage manifest

1. `python -m ours4wan21_data.manifest plan ...` freezes prompt selection,
   split, target speedup, q, control points, normalized residual paths, seeds,
   release order, and four-GPU shards.  Mean threshold and threshold path are
   null.
2. After calibration, create a new `calibrated` mapping file and run
   `python -m ours4wan21_data.manifest materialize ...`.  The materializer
   interpolates the fitted mapping, constructs bounded smooth 50-step paths,
   records the mapping checksum, and produces the only candidate-runnable
   manifest.

The pending config in `configs/speed_threshold_mapping.pending.json` contains
no placeholder thresholds or invented relationship.

The fixed-threshold SeaCache pipeline does not need calibration. Its
single-stage manifest freezes 1,000 prompts, an 800/100/100 prompt split, and
three thresholds sampled uniformly without replacement for each prompt. It
uses the same prompt-selection seed `2026073001` and manifest seed `20260722`;
`target_speedup` and `q` remain null because measured speedup is an outcome,
not a conditioning variable. The grid provenance is frozen in
`configs/seacache_thresholds.wan22_v1.json`.

## Collection and metrics

Every selected prompt first receives one matched full-compute reference with
its MP4, timing/TFLOPs, trace and 50 input latents. Candidate collection is
blocked until all 3,000 baseline bundles pass. Each candidate stores MP4,
50 input latents paired step-for-step with the baseline latents, requested-threshold/action trace,
FFprobe metadata, canonical `rgb_full_reference_v1` PSNR/SSIM/LPIPS,
CUDA-synchronized pipeline-generate timing, per-DiT-call timing/block counts,
and trace-weighted estimated DiT TFLOPs.

The candidate gate follows the locked corrected `SeaCache4Wan21` state
machine: `cond` and `uncond` own independent previous-feature, accumulator,
decision and residual states; calls are recorded in `cond` then `uncond`
order. Every comparison feature, including forced boundaries, is SEA-filtered.
For each branch call the trace directly stores:

- `filtered_relative_l1`: current filtered feature versus the immediately
  preceding filtered feature of the same CFG branch;
- `accumulated_distance_before`: accumulator on entry;
- `accumulated_distance_with_current`: the exact threshold operand before a
  possible recompute reset;
- `accumulated_distance_after`: persistent state after reuse/recompute;
- feature/reference/metric provenance and the executed action.

The raw trace and completion marker contain 100 branch-call records plus 50
step aggregates with explicit cond/uncond columns. Publication emits
`branch_transitions.jsonl/csv` as the lossless RL-facing table alongside
`step_transitions.jsonl/csv`; no filtered distance is reconstructed from the
saved raw input latents.

Each candidate also stores
`video_metrics/per_frame.csv`, `per_video.csv`, the full shared-protocol
`summary.json`, and compact `metrics.json`; trajectory and step tables retain
the three video-level means, while trajectory tables also retain per-frame
standard deviation/min/max values.

The complete archive additionally runs the ten official VBench dimensions
supported for arbitrary `custom_input` videos over all baselines and all
candidates. Because upstream defines no official custom-input aggregate, the
reported archive-level `vbench_score` is explicitly the unweighted mean of
those ten raw dimension scores and is not presented as a leaderboard score.

The headline inference field is `pipeline_generate_wall_seconds`, matching
`TeaCache4Wan21`: it includes text encoding, denoising/cache/CFG/scheduler and
VAE decode, and excludes model construction, MP4 export, latent/file I/O,
FFprobe, PSNR/SSIM/LPIPS, and aggregation. The locked 1.3B protocol performs no
model or CPU offload. Latency speedup is
matched baseline inference time divided by candidate inference time.  DiT
FLOPs speedup is reported separately as a ratio of estimated operation sums.
`TFLOPs` means `10^12` operations; `TFLOP/s` is only a throughput diagnostic.

## Structure

- `src/ours4wan21_data/`: implementation modules.
- `resources/`: immutable bundled OpenVid prompt pool and integrity metadata.
- `configs/`: pending calibration contract.
- `experiments/random_threshold_collection_v1/`: 3,000×3 randomized-path launcher.
- `experiments/seacache_threshold_collection_v1/`: 1,000×3 fixed SeaCache launcher.
- `experiments/calflops_profile_v1/`: one-time real-shape DiT profile.
- `tests/`: CPU contract tests.
- `experiment_results/`: external-result symlink index.
- `PROGRESS.md`, `logs/`: subproject status and handoffs.
- `REMOTE_DEPLOYMENT.md`: full-`offical-code/` remote setup and checks.

All NumPy/BLAS-capable shells and Python workers must explicitly set
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`NUMEXPR_NUM_THREADS=1`.
