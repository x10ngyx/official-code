# Local consistency audit

`audit.py` compares the frozen IQL/Exact-K functions with the local Wan22 trainer, checks CPU SEA filtering numerically, and records a source dataset provenance sample. Architecture and latent feature selection are outside this audit's scope. No models or videos are generated.

Run in conda `wan2.2`, with all four BLAS thread variables set to 1:

```bash
CUDA_VISIBLE_DEVICES='' python experiments/local_consistency_audit/audit.py --output /all/yiran07-disk3/huteng_data/exp/ours22_local_consistency_audit_20260911
```

The output contains `validation.json`, source hashes and a README; the project registers an `experiment_results/` symlink.
