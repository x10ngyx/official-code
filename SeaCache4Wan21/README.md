# SeaCache4Wan21

Corrected filtered-boundary SeaCache integration for locked Wan2.1 T2V inference.

The baseline path executes the original Wan2.1 `generate.py` directly. The
SeaCache path is enabled explicitly and replaces only the T2V sampler/model
forward needed to skip the full Transformer-block stack. It does not include
experimental block cache, CFG cache, ZEUS, TeaCache, or learned controllers.
The release supports one GPU without FSDP or sequence parallelism. It retains
the branch-local gate, SEA filter, accumulated relative-L1, and residual reuse
from `jiwoogit/SeaCache@8dcf490` Wan2.1, but intentionally corrects its forced
boundary storage so raw and filtered representations are never mixed.

## Algorithm boundary

For each denoising step, the conditional and unconditional branches each
compute their own gate from the first block's timestep-modulated normalized
input. Every branch independently maintains its previous feature, accumulated
relative L1, boolean decision, and cached residual tensor. At every step the
feature is reshaped to the video grid and filtered with the scheduler-aligned
SEA spectral filter. Outside forced boundary steps, that filtered feature is
compared with the branch's preceding filtered feature:

```text
residual = hidden_after_all_transformer_blocks - hidden_before_first_block
```

On reuse, only the Transformer blocks are skipped; the patch/time/text path,
head, unpatchify, CFG combination, scheduler, and VAE still run. The first and
last steps recompute by default. Forced boundary steps do not evaluate
relative-L1, but they do run the SEA filter and save the filtered feature as
the next comparison reference. `--use_ret_steps` instead recomputes the first
five steps and removes the final-step cutoff.

The implementation is a clean controller/integration split rather than a
byte-for-byte copy of the official monolithic script. `upstream_lock.json`
records the official SeaCache commit and reference-file hashes as provenance.
The official code skips filtering on forced steps and then stores a raw
feature; this release deliberately differs because the next gate would
otherwise compare `SEA(current)` with `raw(previous)`. The CPU suite checks the
corrected, always-filtered comparison state with an independent filter
transcription.

## Locked upstream

Use the exact Wan2.1 commit recorded in `upstream_lock.json`:

```bash
git clone https://github.com/Wan-Video/Wan2.1.git /path/to/Wan2.1
git -C /path/to/Wan2.1 checkout 65386b2e03c490796eede31b0325a6a595cc684e
python validate_reproduction.py --wan21-root /path/to/Wan2.1
python validate_reproduction.py --seacache-root /path/to/SeaCache
```

Model weights stay under the workspace `models/` root and are not copied into
this repository.

All commands use the project-wide conda environment `wan2.2`. Here and below,
`Wan2.1` names only the locked model/source version. Activate `wan2.2`, or set
`WAN22_PYTHON` to that environment's Python.

## Run

No-cache baseline:

```bash
WAN22_PYTHON=/path/to/wan2.2/bin/python bash run_wan21.sh /path/to/Wan2.1 \
  --task t2v-1.3B --size '832*480' --frame_num 81 \
  --sample_steps 50 --sample_solver unipc --sample_shift 5 \
  --sample_guide_scale 5 --base_seed 42 --offload_model False \
  --ckpt_dir /path/to/models/Wan2.1-T2V-1.3B \
  --prompt 'Two anthropomorphic cats box on a spotlighted stage.' \
  --timing_json /path/to/results/baseline.timing.json \
  --save_file /path/to/results/baseline.mp4
```

SeaCache candidate:

```bash
WAN22_PYTHON=/path/to/wan2.2/bin/python bash run_wan21.sh /path/to/Wan2.1 \
  --enable_seacache --seacache_thresh 0.20 \
  --seacache_trace /path/to/results/seacache.trace.json \
  --timing_json /path/to/results/seacache.timing.json \
  --task t2v-1.3B --size '832*480' --frame_num 81 \
  --sample_steps 50 --sample_solver unipc --sample_shift 5 \
  --sample_guide_scale 5 --base_seed 42 --offload_model False \
  --ckpt_dir /path/to/models/Wan2.1-T2V-1.3B \
  --prompt 'Two anthropomorphic cats box on a spotlighted stage.' \
  --save_file /path/to/results/seacache.mp4
```

The example threshold is not a cross-model recommendation. Calibrate it with
matched prompt/seed/protocol pairs.

`pipeline_generate_wall_seconds` is CUDA-synchronized around
`WanT2V.generate()`: it includes text encoding, denoising/cache/CFG/scheduler,
and VAE decode, while excluding pipeline construction, video export, file I/O,
and evaluation. The locked 1.3B protocol uses one 48GB GPU with
`offload_model=False` and `t5_cpu=False`; no model or CPU offload is permitted.
Compute speedup from matched-prompt sums of this inference field.

When `--timing_json` is enabled, the timing artifact also records every DiT
forward's CUDA span and actual Transformer blocks executed, plus separate T5
and VAE decode CUDA/host spans. The workflow in
`experiments/performance_t2v_1_3b/` combines that trace with a real-shape
Calflops 0.3.2 profile of the full and always-on/no-block paths. It reports
estimated DiT TFLOPs, achieved estimated DiT TFLOP/s, and FLOPs speedup as a
ratio of sums, using the same counting convention as TeaCache; UMT5 encoder
and VAE decode TFLOPs are retained separately. SEA filtering, controller
logic, residual addition, scheduler, and export are outside these component
counts and are not claimed to be negligible.

Use sibling `VideoMetrics/` for paired PSNR/SSIM/LPIPS and
`VbenchEvaluation/` for Vbench200. Store large results under
`/all/yiran07-disk3/huteng_data/exp` and only add a symlink under
`experiment_results/`.

The complete 200-prompt baseline/SeaCache benchmark, including inference time,
trace-weighted estimated DiT TFLOPs, paired PSNR/SSIM/LPIPS, both Vbench200
subset scores, and a final JSON/CSV/Markdown report, is under
`experiments/vbench200_t2v/`.

## Validation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 python -m unittest discover -s tests -v
python validate_reproduction.py --wan21-root /path/to/Wan2.1
```

The CPU suite checks independent cond/uncond gates and residuals,
filtered-feature forced boundaries, 50-step corrected-state equivalence, trace
completeness, explicit enablement, and absence of other cache implementations.
