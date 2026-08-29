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

All NumPy/BLAS processes are explicitly constrained to one CPU thread. Large
results must live outside the repository; the project may keep a local symlink
in `experiment_results/`.
