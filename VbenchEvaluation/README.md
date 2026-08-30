# VbenchEvaluation

This directory contains the reproducible evaluation resources for the fixed
`Vbench200` test set. Metric implementations are installed from the official
VBench repository at the commit recorded in `upstream_lock.json`; this
directory supplies the dataset adapter, locked score constants, aggregation,
and an end-to-end runner.

## Protocol boundary

VBench officially exposes two relevant modes:

- `vbench_standard` supports all 16 dimensions using the prompt metadata in a
  full-info JSON file. It expects video filenames of the form
  `<original prompt>-0.mp4` through `<original prompt>-4.mp4`.
- `custom_input` accepts arbitrary filenames/prompts and, in the locked
  implementation, supports ten dimensions; the six dimensions requiring
  benchmark-specific auxiliary labels (object class, multiple objects, scene,
  appearance style, color, and spatial relationship) are excluded.

This package uses standard mode with `VBench200_full_info.json`. The staging
script maps open-source-friendly IDs such as `vbench200_001.mp4` to the
prompt-based filenames expected by VBench using symlinks, without duplicating
videos. All 16 dimensions can therefore be evaluated on the selected official
records.

The resulting values must be called **Vbench200 subset scores**. They are not
the official full-suite VBench leaderboard score because only 200 of the 944
unique prompts are evaluated.

## Files

- `prepare_videos.py`: validates ID-named outputs and creates prompt-named
  symlinks for standard mode.
- `build_subset_full_info.py`: when `--allow-missing` is requested, filters
  full-info metadata to exactly the staged prompts and rejects subsets that do
  not cover all 16 dimensions. This prevents official VBench from receiving
  empty video lists for the other Vbench200 prompts.
- `evaluate_vbench.py`: runs selected or all 16 official metric dimensions.
- `aggregate_vbench_scores.py`: applies the official normalization ranges,
  dimension weights, and 4:1 Quality/Semantic aggregation.
- `run_vbench200.sh`: end-to-end one-command wrapper.
- `run_custom_vbench.sh`: arbitrary-prompt runner for the ten dimensions
  supported by VBench `custom_input`; its `vbench_score` is an explicitly
  local raw-score mean because upstream defines no official custom aggregate.
- `dimensions.json`: auditable score constants pinned to the official source.
- `upstream_lock.json`: VBench commit, version, license, and source hashes.
- `model_resources.json`: metric checkpoint URLs, target cache paths, and
  dimension ownership extracted from the locked VBench implementation.
- `download_sources.json` and `download_vbench_weights.py`: mirror-first,
  resumable downloads with origin fallback, size checks, and SHA256 auditing.
- `validate_resources.py`: CPU-only validation of the dataset, dimensions,
  upstream lock, and metric-resource coverage.
- `validate_downloaded_weights.py`: CPU-only size, SHA256, extracted-file, and
  pinned-source validation for a completed weight bundle.
- `requirements.txt`: installable pinned VBench dependency.

## Installation

Use a dedicated environment. The locked VBench package is version 0.1.5 and
its installer accepts CUDA 11.6, 11.7, 11.8, or 12.1 builds of PyTorch.

```bash
conda create -n vbench-eval python=3.10 -y
conda activate vbench-eval
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install 'detectron2 @ git+https://github.com/facebookresearch/detectron2.git'
```

VBench downloads several metric checkpoints. Choose one external cache
directory and keep it outside version control:

```bash
export VBENCH_CACHE_DIR=/path/to/VBench
export TORCH_HOME=$VBENCH_CACHE_DIR/torch
export HF_HOME=$VBENCH_CACHE_DIR/huggingface
export XDG_CACHE_HOME=$VBENCH_CACHE_DIR/xdg
```

The runner sets these paths automatically for the current repository layout
and enables VBench local-checkpoint mode. Override them explicitly in other
layouts. No metric weights are vendored in the open-source code directory.

Model weights are not committed to this repository. To reproduce the download
in a directory of your choice:

```bash
python download_vbench_weights.py \
  --output-dir /path/to/vbench-weights
```

The completed bundle contains 15 extracted checkpoint files totaling
`9,052,714,209` bytes (about 8.431 GiB), plus the pinned DINO source tree and
audit metadata. `weights/VALIDATION.json` contains every final SHA256 and
`weights/SOURCE_PROVENANCE.json` records the sources observed during the
mirror-first download.

Validate the local package without downloading models or using a GPU:

```bash
python validate_resources.py
```

After downloading, validate every checkpoint and the pinned DINO source tree:

```bash
python validate_downloaded_weights.py --weights-dir weights
```

## Input video names

For one generated video per prompt, place these files in one directory:

```text
vbench200_001.mp4
vbench200_002.mp4
...
vbench200_200.mp4
```

For multiple samples per prompt, use zero-based indices:

```text
vbench200_001-0.mp4
vbench200_001-1.mp4
...
```

All inputs must use the same extension. The adapter supports `.mp4` and
`.gif`. The one-seed protocol is valid for matched internal comparisons but
must not be compared as though it used the official five-sample protocol.

## Run

Experiment outputs should be written under the external experiment root, not
inside the code repository:

```bash
bash run_vbench200.sh \
  /path/to/id_named_videos \
  /path/to/experiment-results/my_vbench200_eval \
  1
```

The last argument is the expected number of videos per prompt (`1` to `5`).
For a deliberately selected Vbench200 subset, append `--allow-missing`; the
selected records must still cover all 16 dimensions before aggregate scoring
is valid.
The working directory will contain:

```text
staged_videos/                  # symlinks only
staging_manifest.json           # exact ID/prompt/source mapping
subset_full_info.json            # present for deliberate partial subsets
scores/                          # per-dimension official VBench JSON files
vbench200_aggregate_scores.json # raw, normalized, and aggregate subset scores
```

To distribute dimensions across GPUs, invoke `evaluate_vbench.py` separately
with disjoint `--dimensions` lists and a shared score directory, then run the
aggregator once all 16 dimensions are complete.

## Single-video smoke test

The reproducible CPU smoke test under `experiments/single_sample_smoke/`
executes both the checkpoint-free `temporal_flickering` dimension and the
CLIP ViT-B/32-backed `background_consistency` dimension. It uses a short
repository sample only to validate decoding, standard-mode prompt staging,
official VBench dispatch, local checkpoint loading and JSON output. Its scores
are not Vbench200 benchmark results because the sample was not generated from
the staged prompts.

A completed smoke run should contain `VALIDATION.json`. Generated smoke
artifacts are intentionally excluded from version control.

## Score definition

Each raw dimension score is normalized using the official min/max range. The
Dynamic Degree dimension has weight 0.5 and every other dimension has weight
1. Quality Score is the weighted average of seven quality dimensions;
Semantic Score is the average of nine semantic dimensions; Total Score is
`(4 * Quality + Semantic) / 5`. The official implementation does not clamp
normalized values, so this wrapper also leaves them unclamped.

See the official [VBench README](https://github.com/Vchitect/VBench) and the
locked [`scripts/constant.py`](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/scripts/constant.py)
for the upstream definitions.
