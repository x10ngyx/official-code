# Completed scan readout

`audit_scan.py` performs a CPU-only audit of an existing three-target scan. It validates source and artifact hashes, all measured run contracts and official action schedules, recomputes performance and target selection, checks frame/video quality aggregation and all 16 VBench dimensions, and compares the three independent baselines. An optional `--baseline-videos` directory also checks cross-method MP4 identity.

It writes `final_validation.json`, `scan_summary.json`, `scan_candidates.csv`, and `RESULTS.md` inside the existing external result directory. Original generation, timing, evaluation, and completion files remain intact. No inference or quality models are rerun.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/huteng/yes/envs/wan2.2/bin/python experiments/scan_readout/audit_scan.py \
  --result-dir /all/yiran07-disk3/huteng_data/exp/magcache4wan22_targeted_e_scan_gpu123_20260905_165530
```

When `timing_correction.json` is present, the audit also loads the hash-pinned corrected shared baseline, checks prompt/video identity, reconstructs the preserved original same-GPU performance, and verifies the normalized performance and target selection. Corrected speedups are labeled explicitly in RESULTS.md. The inference source lock is unchanged.
