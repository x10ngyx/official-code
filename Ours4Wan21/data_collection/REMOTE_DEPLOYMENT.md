# Remote deployment

The transfer unit is the complete `work/offical-code/` tree, not this
`data_collection/` directory alone. The collection project intentionally uses
these shared resources after transfer:

- `VideoMetrics/video_metrics/` for canonical `rgb_full_reference_v1`
  PSNR, SSIM, and LPIPS;
- `CalflopsEvaluation/calflops_eval/` for audited manual attention FLOPs;
- `Ours4Wan21/upstream_lock.json` and `NOTICE.md` for Wan2.1 compatibility and
  provenance;
- `TeaCache4Wan21/LICENSE.upstream.txt` for the shared Apache-2.0 text.

The 5,000-row OpenVid prompt snapshot remains under
`Ours4Wan21/data_collection/resources/` because its originating workspace
source is outside `offical-code/`.

The following large or environment-specific inputs remain external:

1. The project-wide `wan2.2` conda environment with working Python/CUDA.
   `Wan2.1` below names the model/source version only, never an environment.
   Install the shared metric and Calflops requirements when they are not
   already present. LPIPS is locked to `0.1.4`.
2. The Wan2.1 source tree at commit
   `65386b2e03c490796eede31b0325a6a595cc684e`. Four compatibility file hashes
   from `Ours4Wan21/upstream_lock.json` are checked before every GPU phase.
3. The `Wan2.1-T2V-1.3B` checkpoint directory.
4. A writable external result root. It defaults to
   `/all/yiran07-disk3/huteng_data/exp`; set `EXP_BASE` on another machine.
5. A fitted `calibrated` speed-to-mean-threshold mapping before randomized-path
   candidates can run. The pending mapping is intentionally blank and fails
   closed. The fixed-threshold SeaCache pipeline does not use this mapping.
6. The locked AlexNet weight used by LPIPS under a Torch model cache. Set
   `METRICS_MODEL_CACHE` when it cannot be inferred from an ancestor
   `models/torch-cache` directory.

## Transfer and verify

From the source machine:

```bash
rsync -a work/offical-code/ user@remote:/path/offical-code/
```

On the remote machine:

```bash
cd /path/offical-code
conda run --no-capture-output -n wan2.2 python -m pip install \
  -r VideoMetrics/requirements.txt -r CalflopsEvaluation/requirements.txt
cd Ours4Wan21/data_collection
export PYTHONPATH="$PWD/src:$PWD/../../VideoMetrics:$PWD/../../CalflopsEvaluation${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
conda run --no-capture-output -n wan2.2 python -m ours4wan21_data.preflight --phase package
conda run --no-capture-output -n wan2.2 python -m unittest discover -s tests -v
```

Preflight must report the expected `offical-code/` root, prompt SHA256
`fb5d5d73f86b84d10d8e55154b789ac8549c74e90f33c1d4d2a02d67a5cde3e5`,
5,000 prompt rows, FFmpeg/FFprobe, and status `ok`.

## Runtime variables

```bash
export WAN21_ROOT=/remote/path/Wan2.1-locked-65386b2
export CHECKPOINT_DIR=/remote/models/Wan2.1-T2V-1.3B
export EXP_BASE=/remote/large-disk/exp
export RUN_ID=wan21_random_threshold_v1
export METRICS_MODEL_CACHE=/remote/models/torch-cache
# Optional: auto selects the worker's visible CUDA device; CPU is supported.
export METRICS_DEVICE=auto
export LPIPS_BATCH_SIZE=8
# Optional explicit interpreter override; it must be the wan2.2 environment Python:
# export WAN22_PYTHON=/absolute/path/to/wan2.2/bin/python
```

## Phase order

```bash
bash experiments/random_threshold_collection_v1/launch_4gpu.sh preflight
bash experiments/random_threshold_collection_v1/launch_4gpu.sh plan
bash experiments/random_threshold_collection_v1/launch_4gpu.sh profile
bash experiments/random_threshold_collection_v1/launch_4gpu.sh baselines
```

After calibration, point `CALIBRATION_CONFIG` at the fitted mapping and run:

```bash
export CALIBRATION_CONFIG=/remote/path/speed_threshold_mapping.calibrated.json
bash experiments/random_threshold_collection_v1/launch_4gpu.sh materialize
bash experiments/random_threshold_collection_v1/launch_4gpu.sh candidates
bash experiments/random_threshold_collection_v1/launch_4gpu.sh finalize
```

The four-GPU launcher assigns one shard to each visible GPU 0–3. Generated
artifacts remain under `$EXP_BASE/$RUN_ID`; only a symlink is created in
`experiment_results/`.

For the fixed-threshold SeaCache dataset, use a separate `RUN_ID` and run:

```bash
bash experiments/seacache_threshold_collection_v1/launch_4gpu.sh plan
bash experiments/seacache_threshold_collection_v1/launch_4gpu.sh profile
bash experiments/seacache_threshold_collection_v1/launch_4gpu.sh baselines
bash experiments/seacache_threshold_collection_v1/launch_4gpu.sh candidates
bash experiments/seacache_threshold_collection_v1/launch_4gpu.sh finalize
```

This path is runnable immediately after manifest creation: it samples 1,000
prompts and three distinct values per prompt from the frozen nine-threshold
Wan2.2 grid, producing 1,000 baselines and 3,000 candidates.
