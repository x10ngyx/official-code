# CPU integration validation

validate.py runs unit tests and creates clearly marked synthetic collection metadata, followed by two two-epoch offline runs (scalar5 and sea7), policy checkpoint reloads and 50-step exact-K controller replays. CUDA is hidden, all BLAS thread limits are 1, smoke checkpoints are forbidden in production inference. No Wan videos or production training.

```bash
conda run --no-capture-output -n wan2.2 python experiments/rl_validation_v1/validate.py --output-dir /all/yiran07-disk3/huteng_data/exp/UNIQUE_OURS21_CPU_VALIDATION
```

The result directory contains unit_tests.txt, synthetic_collection/, training subdirectories and VALIDATION.json; weights are under models/<result-name>/. Result directories have project experiment_results symlinks.
