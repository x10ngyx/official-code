# Shared baseline scan correction

This post-processing workflow replaces the baseline side of the completed MagCache E scan with the matching 11 prompts from the corrected shared Wan2.2 baseline. The resulting mean is 989.138134 seconds/video; it does not substitute the full 200-prompt mean for an 11-prompt comparison. GPU1/2/3 MagCache candidate measurements, all video files, quality metrics, FLOPs and official action schedules remain unchanged.

`normalization.py` loads hash-pinned shared reported timings and the video reuse index, then normalizes one measured performance record while retaining `raw_same_gpu_performance`. `apply_correction.py` backs up affected files, updates all three target performance/report/selection records and completion hashes, reruns the correction-aware CPU scan audit, and refreshes the complete 1.8x E-grid readout. Existing same-GPU baseline manifests stay valid and unchanged.

Run from this project, using a fresh external output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/huteng/yes/envs/wan2.2/bin/python experiments/shared_baseline_scan_correction/apply_correction.py \
  --scan-dir /all/yiran07-disk3/huteng_data/exp/magcache4wan22_targeted_e_scan_gpu123_20260905_165530 \
  --correction-dir /all/yiran07-disk3/huteng_data/exp/wan22_gpu0_healthy_mean_correction_20260906 \
  --reuse-index /all/yiran07-disk3/huteng_data/exp/magcache4wan22_baseline_reuse_audit_20260906/baseline_reuse_manifest.json \
  --output-dir /all/yiran07-disk3/huteng_data/exp/magcache4wan22_shared_baseline_scan_correction_20260906
```

Original report artifacts and their hashes are retained in `backups/`; the raw generation and quality evidence remains at its original location. This modifies reported normalization, not the model or inference implementation. The new speedups are estimates against the corrected shared baseline.
