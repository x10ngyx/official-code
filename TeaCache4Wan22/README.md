# TeaCache4Wan22

Reproducible non-retention TeaCache integration for Wan2.2 T2V-A14B.

The repository contains the TeaCache runtime, an auditable patch against an
exact Wan2.2 commit, source preparation and validation scripts, the fixed
inference protocol, the complete 70-prompt coefficient-calibration workflow,
CPU tests, run manifests, and evaluation entry points. Model weights and
generated videos remain external because of their size.

## Supported protocol

The first release intentionally supports one configuration:

- model/task: `Wan2.2-T2V-A14B` / `t2v-A14B`;
- output: `832x480`, 45 frames, 16 fps;
- sampler: 50-step DPM++, shift `12`;
- CFG: low/high `(3, 4)`, model boundary `.875`;
- seed: `42` for the supplied reproduction runner;
- precision/runtime: BF16, one GPU, model offload enabled, T5 on GPU;
- TeaCache: `use_ret_steps=False`, one public threshold, no simultaneous
  ZEUS/SeaCache/block/CFG cache.

Coefficient files are protocol-bound. Runtime validation rejects mismatched
geometry, step count, solver, shift, CFG, boundary, dtype, or retention mode.

## Algorithm boundary

At every timestep, Wan2.2 still executes patch/time/text embeddings, the model
head, and unpatchify. TeaCache may skip only the Transformer block stack. On a
fresh step it stores:

```text
block_residual = hidden_after_all_blocks - hidden_before_first_block
```

On a cache hit it adds the previous residual to the new block input. The gate
uses adjacent relative-L1 of the compact timestep embedding `e`, followed by a
stage-specific quartic mapping and the standard accumulated-threshold rule.

Wan2.2 uses separate high- and low-noise models, so the implementation keeps:

- one gate/accumulator per model stage;
- one shared cond/uncond decision per timestep;
- separate cond and uncond block residuals;
- an explicit recompute at each stage's first step and at the global final
  step;
- explicit release of the completed stage's GPU residual before model switch.

Only the first token of token-wise `e` is retained because scalar T2V
timesteps are identical across tokens. This avoids cloning the multi-GiB
token-wise `e0` tensor.

## Prepare the pinned Wan2.2 source

The canonical algorithm lives in `runtime/teacache.py`. The preparation script
checks out the locked upstream commit, verifies original file hashes, applies
the integration patch, installs the runtime, compiles all modified files, and
writes `.teacache4wan22_prepared.json`.

```bash
bash scripts/prepare_wan22.sh /path/to/prepared-Wan2.2
```

The pinned upstream is `Wan-Video/Wan2.2@42bf4cf`. Install its dependencies
following the generated checkout's `INSTALL.md` and `requirements.txt`. The
checkpoint directory must contain the official `Wan2.2-T2V-A14B` weights.

## Run

All generation-side NumPy/BLAS thread counts are fixed to one by the supplied
runner. A positive `THRESHOLD` enables TeaCache; `THRESHOLD=0` runs the exact
patched-tree no-cache baseline without constructing a controller.

```bash
export WAN22_SOURCE=/path/to/prepared-Wan2.2
export WAN22_CKPT=/path/to/models/Wan2.2-T2V-A14B
export WAN22_PYTHON=/path/to/wan2.2/bin/python
export RESULT_ROOT=/path/to/experiment/results
export THRESHOLD=0.10
export PROMPT='Two anthropomorphic cats box on a spotlighted stage.'
export RUN_ID=prompt001_threshold010
bash scripts/run_t2v_a14b.sh
```

`0.10` above is an integration-smoke example, not a recommended operating
point. Select a threshold only from matched-seed speed/fidelity evaluation;
the coefficient fit by itself does not validate a threshold.

Each run saves the video, raw log, a structured timing JSON, and a run
manifest with artifact SHA256 values. When TeaCache is enabled it also writes
a JSON trace containing every shared gate decision, both branch actions,
polynomial inputs/outputs, accumulator values, coefficient SHA256, runtime
protocol. The standalone timing JSON follows the same single schema and outer
pipeline profiler as TeaCache4Wan21. It stores all 100 DiT calls (`50 steps x
cond/uncond`) with stage, branch, CUDA-event seconds, host span, actual
Transformer blocks executed, and full/reuse flags.

## Timing and quality

`pipeline_generate_wall_seconds` is CUDA-synchronized wall time around
`WanT2V.generate()`. It includes text encoding, denoising/cache/CFG/scheduler,
cache release, transfers/offload performed inside `generate()`, and VAE
decode. Pipeline construction, MP4 export, file I/O, and evaluation are
outside this boundary. Speedup uses matched-prompt sums of this field, never
end-to-end process wall time.

`model_forward_cuda_seconds` is a separate DiT-only diagnostic: it is the sum
of the 100 per-call CUDA-event spans and does not replace the headline
pipeline-generation latency above. The same schema separately records the two
T5 encoder calls and one VAE decode with CUDA-event and host spans.

Validate one or more matched pairs and compute the required ratio-of-sums
speedup with:

```bash
python scripts/compare_runs.py \
  --baseline-manifest /path/to/baseline.manifest.json \
  --teacache-manifest /path/to/teacache.manifest.json \
  --output /path/to/matched_speedup.json
```

Repeat both manifest options in matching order to aggregate multiple prompts.
The validator checks all recorded hashes, the fixed protocol, 50-step trace,
32/18 stage split, shared CFG actions, forced recompute boundaries, and all
100 timed DiT calls before reporting speedup.

## Calflops and computation

`experiments/performance_t2v_a14b/` mirrors the TeaCache4Wan21 computation
workflow. It independently profiles the real `832x480`, 45-frame high/low x
cond/uncond DiT full-forward and always-on/no-block paths with Calflops 0.3.2,
checks their expected equivalence, adds a manual dense FlashAttention-core
correction, then weights those costs by the timing trace's actual 100-call
block path.

```bash
python experiments/performance_t2v_a14b/profile_calflops.py \
  --wan22-root /path/to/prepared-Wan2.2 \
  --checkpoint-dir /path/to/models/Wan2.2-T2V-A14B \
  --calflops-source /path/to/calculate-flops.pytorch \
  --output /all/yiran07-disk3/huteng_data/exp/<run>/calflops_profile.json

python experiments/performance_t2v_a14b/aggregate_performance.py \
  --baseline-manifest /path/to/baseline.manifest.json \
  --teacache-manifest /path/to/teacache.manifest.json \
  --calflops-profile /all/yiran07-disk3/huteng_data/exp/<run>/calflops_profile.json \
  --output-dir /all/yiran07-disk3/huteng_data/exp/<run>/performance
```

Repeat the two manifest options in matched order to aggregate prompts. Both
latency speedup and FLOPs speedup are ratios of summed baseline/candidate
quantities. `TFLOPs` means decimal `10^12` floating-point operations per
forward/video, not `TFLOP/s`; achieved estimated DiT `TFLOP/s` is reported as
a separate diagnostic. The headline remains estimated DiT TFLOPs, while the
same profile separately saves two UMT5 encoder forwards and one VAE decode per
video. Scheduler, video export, and TeaCache controller/residual operations
remain outside these component counts; none is labeled complete end-to-end
FLOPs.

For paired full-reference quality, use this repository's sibling
`VideoMetrics/` package with protocol `rgb_full_reference_v1`. VBench200 and
the official 16-dimension evaluation adapter are available in `Vbench200/`
and `VbenchEvaluation/`.

## Coefficient reproduction

`experiments/fit_t2vcompbench70_wan22_t2v_a14b/` reproduces TeaCache's
70-prompt calibration protocol with a deterministic, category-balanced sample
from the pinned historical T2V-CompBench suite. It performs full-compute
denoising, collects scalar `e`, post-block `H`, and residual `H-Z` statistics,
then fits independent high/low quartics.

After the raw experiment finishes:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 python \
  experiments/fit_t2vcompbench70_wan22_t2v_a14b/audit_fit.py \
  --result-root /path/to/calibration-result

python scripts/package_coefficients.py \
  --result-root /path/to/calibration-result \
  --output coefficients/wan22_t2v_a14b_50step_dpmpp_nonretention.json
```

The packager requires all 70 samples, `7000` forward records, `6720`
within-stage transitions, and `6580` gate-eligible records. It embeds all 70
sample hashes and source/protocol locks in the public coefficient file.
The included default coefficient SHA256 is
`8b9550f7bd190aafcfac7871bb12d387fbfb4f6afe61e285d08a5b641b0c2970`.

Independent audit exactly reproduced both quartics and reported in-sample /
leave-one-prompt-out / leave-one-category-out R² of
`0.278456 / 0.273534 / 0.267062` for high noise and
`0.630407 / 0.622728 / 0.628646` for low noise. The high-stage mapping is a
weak but cross-validation-stable TeaCache heuristic, not a precise
per-forward predictor. Raw-power quartics must not be extrapolated beyond the
protocol and `x` domains stored in the coefficient file.

## Validation

```bash
WAN22_PYTHON=/path/to/wan2.2/bin/python bash tests/run_tests.sh
```

The CPU suite covers shared-gate semantics, branch-separated residuals,
stage/final recompute boundaries, per-forward timing, trace-weighted FLOPs,
protocol mismatch rejection, source-URL normalization, trace writing, Python
compilation, and shell syntax.
`prepare_wan22.sh` additionally performs
an end-to-end patch-application validation against the exact upstream tree.

Before reporting a speed-quality result, also verify:

1. no-cache output equivalence on the prepared tree;
2. high/low stage split and recompute boundaries;
3. identical cond/uncond actions with distinct residual tensors;
4. complete 50-step trace and finite video output;
5. peak GPU memory and absence of a retained high-stage residual;
6. paired PSNR/SSIM/LPIPS and inference-only timing.

## Directory structure

- `runtime/`: canonical TeaCache controller and DiT forward profiler.
- `patches/`: reviewable integration patch for the pinned Wan2.2 source.
- `scripts/`: source preparation, validation, fixed runner, manifests, and
  coefficient packaging.
- `configs/`: frozen runtime protocol.
- `coefficients/`: validated protocol-bound polynomial files.
- `experiments/`: coefficient calibration, fixed-protocol smoke, and
  performance profiling/aggregation.
- `tests/`: CPU and installation checks.
- `experiment_results/`: ignored local symlinks to external result roots.
- `upstream_lock.json` / `NOTICE.md`: source commits, original hashes,
  artifact hashes, licenses, and attribution.

## Scope

v0.1 deliberately rejects I2V/TI2V/S2V, retention mode, distributed FSDP/SP,
and combinations with other caches. These paths require separate algorithmic
integration and calibration and must not silently reuse the included
coefficients.
