# VBench evaluation experiments

Each experiment lives in its own folder. Write large outputs to an external
experiment directory; `../experiment_results/` may contain local symlinks to
those result directories, but generated artifacts are not versioned.

- `single_sample_smoke/`: CPU-only official VBench scoring smoke test on one
  video and one metric dimension.
