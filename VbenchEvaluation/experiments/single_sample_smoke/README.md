# Single-sample VBench smoke test

This experiment verifies the real official VBench execution path with one
video. The first script selects the first Vbench200 record assigned to
`temporal_flickering`, stages the input under VBench's required
`<prompt>-0.mp4` name, and calls `evaluate_vbench.py` in official standard
mode on CPU. The optional second script derives a short 16-frame clip and runs
the checkpoint-backed `background_consistency` dimension with the downloaded
OpenAI CLIP ViT-B/32 weight.

The selected metric does not load a checkpoint, so it can run without taking
a GPU away from another experiment. The resulting number is only a pipeline
smoke-test value: the bundled sample video was not generated from the selected
Vbench200 prompt and this is not a Vbench200 aggregate score.

Usage:

```bash
./run_single_sample_smoke.sh \
  /path/to/sample.mp4 \
  /path/to/experiment-results/my_vbench_smoke \
  /path/to/VBench-source-at-the-locked-commit
```

The script requires the `wan2.2` Python environment by default. Override it
with `WAN22_PYTHON=/path/to/python` if needed. `VBENCH_CACHE_DIR` defaults to
the resolved `VbenchEvaluation/weights` target.

For the weight-backed CPU check, install `openai-clip` into an isolated target
directory and run:

```bash
PYTHON_DEPS=/path/to/python_deps ./run_background_consistency_smoke.sh \
  /path/to/sample.mp4 \
  /path/to/experiment-results/my_vbench_smoke \
  /path/to/VBench-source-at-the-locked-commit
```

Run `validate_results.py RESULT_ROOT` after both commands. It independently
recomputes temporal flickering from decoded frames, checks the weighted result
aggregation, and verifies the sample, CLIP checkpoint and VBench commit.
