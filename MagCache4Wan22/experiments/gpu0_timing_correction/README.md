# GPU0 timing correction

`apply_correction.py` applies the user-authorized direct healthy-card mean substitution to the seven frozen Wan2.2 VBench200 cohorts: one shared baseline and six SeaCache/TeaCache candidates. Each GPU0 video receives its own cohort's GPU1/2/3 arithmetic mean for complete generate, DiT, T5 and VAE timing fields. This deliberately uses the condition mean, not a multiplicative or action-matched correction.

The script stages and verifies all changes first, backs up every existing file it replaces, then updates formal performance rows, summaries, benchmark reports, the TeaCache suite report and the shared baseline reuse index. Original timing JSON, calls, traces, video files, operation counts and quality data stay intact. Corrected rows retain `raw_timing`; source condition directories expose `reported_timings.jsonl` and `timing_correction.json` links to the external correction package. Re-aggregation from raw timings must reapply this recorded correction.

All artifacts, backups, staging and validation live under the external experiment root. Run in the project directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/huteng/yes/envs/wan2.2/bin/python experiments/gpu0_timing_correction/apply_correction.py \
  --audit-dir /all/yiran07-disk3/huteng_data/exp/wan22_gpu0_timing_audit_20260906 \
  --output-dir /all/yiran07-disk3/huteng_data/exp/wan22_gpu0_healthy_mean_correction_20260906 \
  --apply
```

The output directory must be fresh. Without `--apply`, the package is only staged. On an application/validation failure, already written formal files are restored from backup and newly installed links are removed.

`verify_correction.py --result-dir <correction-package>` independently verifies installed JSON/CSV/Markdown reports, all time distributions using NumPy, exact healthy-row preservation, raw field restoration, protected SHA values and the shared baseline reuse index. Run with the same environment/thread variables.
