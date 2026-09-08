# Original core CUDA smoke

`run_smoke.py` uses the pinned WanModel class with two small random-weight
experts on CUDA/BF16. It compares 100 outputs from direct official installation
with scoped/observed installation under the default .06/K2/retention.4 method.
It does not load or save model weights and produces no videos or quality scores.
This is an implementation check, not an A14B performance measurement.

```bash
conda activate wan2.2
python run_smoke.py --source /path/to/pristine/Wan2.2-42bf4cf \
  --output-dir /all/yiran07-disk3/huteng_data/exp/magcache_core_smoke
```

The script explicitly sets all four BLAS thread variables to 1 before loading
NumPy/PyTorch. Results are indexed by a project symlink.
