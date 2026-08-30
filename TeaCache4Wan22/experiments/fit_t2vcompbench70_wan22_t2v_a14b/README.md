# Wan2.2 T2V-A14B TeaCache polynomial calibration

This experiment performs full-compute Wan2.2 T2V-A14B denoising on 70
T2V-CompBench prompts and fits one quartic TeaCache mapping for each of the
high- and low-noise stages.

## Prompt set

- Source: official `KaiyueSun98/T2V-CompBench` repository.
- Pinned commit: `4fa8be2c46d49796a16678c245ea16e3f12bc4c1`
  (the 2024 700-prompt suite available when TeaCache was reported).
- Selection: 10 prompts from each of seven categories, sampled without
  replacement by one `random.Random(42)` stream; selected source indices are
  sorted and saved in `prompts.jsonl`.
- TeaCache does not publish its exact 70 prompt indices, so this is a
  deterministic reproduction of its stratified calibration protocol, not a
  claim that the prompt identities are byte-for-byte identical.

## Inference and fitting protocol

The run uses the project protocol: `832x480`, 45 frames, 50-step DPM++, shift
12, low/high CFG `(3, 4)`, boundary `.875`, seed 42, BF16,
`offload_model=True`, and `t5_cpu=False`. No cache is enabled during data
collection, and VAE decode/video export is skipped because only the denoising
trajectory is required.

Forward hooks collect scalar statistics without changing Wan2.2 model code:

- `x`: adjacent relative-L1 of non-retention timestep embedding `e`;
- `y`: adjacent relative-L1 of full-compute post-block/pre-head hidden state
  `H`;
- diagnostic: adjacent relative-L1 of the block residual `H-Z`.

High/low stages never share previous state. Cond/uncond observations are
pooled for one polynomial per stage and evaluated separately by branch. The
primary fit excludes stage-first points (which have no within-stage previous
state) and global step 49 (forced recompute under `use_ret_steps=False`). The
all-within-stage fit is retained as a diagnostic.

## Files

- `prepare_prompts.py`: downloads the pinned source files and creates the
  manifest.
- `collect_distances.py`: resumable full-compute collection worker.
- `launch_shards.sh`: launches one worker on each of GPUs 0--3 via tmux.
- `watch_and_fit.py`: waits for 70 validated prompt JSON files, stops on any
  recorded failure, and invokes the fitter once the set is complete.
- `fit_polynomials.py`: validates all 70 samples and fits high/low quartics.
- `audit_fit.py`: independently recomputes the fit, checks data grain and
  source equivalence, and reports leave-one-prompt/category-out diagnostics.

All NumPy/BLAS processes are explicitly constrained to one CPU thread. Large
results must live outside the repository; the project may keep a local symlink
in `experiment_results/`.

## Reproduce

First prepare the pinned Wan2.2 source with the repository-level
`scripts/prepare_wan22.sh`, install the official dependencies, and make the
official T2V-A14B checkpoint available. Then run:

```bash
export WAN22_SOURCE=/path/to/prepared-Wan2.2
export WAN22_CKPT=/path/to/Wan2.2-T2V-A14B
export WAN22_PYTHON=/path/to/wan2.2/bin/python
export RESULT_ROOT=/path/outside/the/repository/teacache4wan22_fit

"$WAN22_PYTHON" prepare_prompts.py --output-dir "$RESULT_ROOT"
bash launch_shards.sh "$RESULT_ROOT" teacache4wan22_fit

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 "$WAN22_PYTHON" watch_and_fit.py \
  --result-root "$RESULT_ROOT" \
  --manifest "$RESULT_ROOT/prompts.jsonl"
```

`launch_shards.sh` is intentionally fixed to GPUs 0--3. The watcher may run in
the foreground or in a separately named tmux session. Completed workers are
resumable: rerunning the launcher skips sample JSON files that already pass
the worker's completeness check.

Finally package the validated raw fit from the TeaCache4Wan22 project root:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 "$WAN22_PYTHON" \
  experiments/fit_t2vcompbench70_wan22_t2v_a14b/audit_fit.py \
  --result-root "$RESULT_ROOT"

"$WAN22_PYTHON" scripts/package_coefficients.py \
  --result-root "$RESULT_ROOT" \
  --output coefficients/wan22_t2v_a14b_50step_dpmpp_nonretention.json
```

The packager accepts calibration traces only from the locked upstream tree or
the exact tree produced by `prepare_wan22.sh`. For a legacy result collected
from another full-compute source, it deliberately stops unless one prompt has
been rerun on an approved source and all 100 scalar records pass
`scripts/validate_calibration_source_equivalence.py`. This exception is
recorded, hashed, and embedded in the packaged coefficient provenance; it is
not a way to bypass source validation silently.
