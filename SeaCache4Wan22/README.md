# SeaCache4Wan22

Clean, reproducible SeaCache integration for Wan2.2 T2V-A14B.

The package contains a canonical stage-aware controller, an auditable minimal
patch against one Wan2.2 commit, source preparation and validation scripts, a
fixed single-GPU runner, full decision trace, inference-only timing, and CPU
tests. It deliberately contains no experimental block cache, CFG cache, ZEUS,
TeaCache, or learned controller.

## Supported protocol

- `Wan2.2-T2V-A14B` / `t2v-A14B`;
- `832x480`, 45 frames, 16 fps;
- 50-step DPM++, shift `12`, seed `42`;
- low/high CFG `(3,4)`, model boundary `.875`;
- BF16 DiT, one GPU, model offload enabled, T5 on GPU;
- timestep SeaCache only.

The runtime rejects model-side protocol mismatches. Threshold and
`use_ret_steps` are method parameters and must be recorded for each run.

## Algorithm boundary

At each step the active high- or low-noise model computes the first block's
timestep-modulated normalized input. The controller reshapes it to the video
grid, applies the scheduler-aligned SEA spectral filter, and evaluates the
accumulated relative-L1 threshold. Cond publishes one decision and uncond
consumes it; the two branches keep separate residual tensors:

```text
residual = hidden_after_all_transformer_blocks - hidden_before_first_block
```

On reuse, only the Transformer block stack is skipped. Embeddings, head,
unpatchify, CFG, scheduler, model switching, and VAE still execute. High and
low stages keep separate state, the first step of each stage recomputes, and
completed-stage tensors are released immediately. By default the global first
and last steps recompute. Retention mode recomputes the first five global steps
and removes the final-step cutoff.

## Prepare pinned Wan2.2

```bash
bash scripts/prepare_wan22.sh /path/to/prepared-Wan2.2
```

The script checks out `Wan-Video/Wan2.2@42bf4cf`, verifies original hashes,
applies only `patches/wan22_42bf4cf_seacache.patch`, installs the two runtime
files, compiles modified Python, and writes
`.seacache4wan22_prepared.json`. Model weights remain in the workspace
`models/` root.

## Run

```bash
export WAN22_SOURCE=/path/to/prepared-Wan2.2
export WAN22_CKPT=/path/to/models/Wan2.2-T2V-A14B
export WAN22_PYTHON=/path/to/wan2.2/bin/python
export RESULT_ROOT=/all/yiran07-disk3/huteng_data/exp/seacache_wan22
export THRESHOLD=0.20
export PROMPT='Two anthropomorphic cats box on a spotlighted stage.'
export RUN_ID=prompt001_threshold020
bash scripts/run_t2v_a14b.sh
```

`THRESHOLD=0` runs the patched-tree no-cache baseline without constructing a
controller. Set `SEACACHE_USE_RET_STEPS=1` only with a positive threshold.
The example threshold is not a universal recommendation; calibrate it with
matched prompt/seed/protocol pairs.

Each run writes MP4, raw log, timing JSON, manifest, and—when enabled—a full
SeaCache trace. `pipeline_generate_wall_seconds` is CUDA-synchronized around
`WanT2V.generate()`: it includes text encoding, denoising/cache/CFG/scheduler,
in-generate transfers/offload, and VAE decode, while excluding pipeline
construction, video export, file I/O, and evaluation. Compute speedup from the
ratio of matched-prompt summed inference times. The timing schema separately
retains T5, DiT, and VAE decode CUDA/host spans.

The workflow in `experiments/performance_t2v_a14b/` profiles the real high/low
DiT shapes with Calflops 0.3.2 and weights full versus reuse forwards from the
100-call timing/decision trace. It reports estimated DiT TFLOPs, achieved
estimated DiT TFLOP/s, and FLOPs speedup as a ratio of sums, using TeaCache's
counting convention; UMT5 encoder and VAE decode TFLOPs are retained
separately. SEA filtering, controller logic, residual addition, scheduler, and
export are outside these component counts and are not claimed to be negligible.

Use sibling `VideoMetrics/` for paired PSNR/SSIM/LPIPS and
`VbenchEvaluation/` for Vbench200. Store all large result bundles under
`/all/yiran07-disk3/huteng_data/exp`; `experiment_results/` is only an index of
local symlinks.

The complete 200-prompt baseline/SeaCache benchmark, including inference time,
trace-weighted estimated DiT TFLOPs, paired PSNR/SSIM/LPIPS, both Vbench200
subset scores, and a final JSON/CSV/Markdown report, is under
`experiments/vbench200_t2v/`.

## Validation

```bash
WAN22_PYTHON=/path/to/wan2.2/bin/python bash tests/run_tests.sh
python scripts/validate_prepared_tree.py \
  --source /path/to/prepared-Wan2.2 --mode prepared
```

The suite covers stage-local gates, synchronized CFG actions,
branch-separated residuals, stage/final recompute boundaries, protocol
rejection, trace completeness, exact patch scope, shell syntax, and exclusion
of other cache implementations.
