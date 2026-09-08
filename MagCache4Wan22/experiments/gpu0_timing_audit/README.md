# GPU0 timing audit

CPU-only diagnosis of the shared Wan2.2 VBench200 baseline and the three formal SeaCache/TeaCache candidate conditions. `audit.py` reads physical GPU ownership, generation manifests, all 1,400 timing records and all 140,000 CFG calls; matches GPU0 to GPUs 1/2/3 by the exact 100-call full/reuse sequence; estimates a separate latency multiplier per condition; and checks temporal variation, component attribution, peer consistency, and the independent MagCache baseline repeats.

The factor is a **time multiplier**, `r = GPU0 seconds / expected healthy-GPU seconds`. Time inflation is `r-1`; throughput loss is `1-1/r`. The optional correction arithmetic is `t/r` for GPU0 only. The audit preserves all original experiment files and does not publish corrected benchmark results.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/huteng/yes/envs/wan2.2/bin/python experiments/gpu0_timing_audit/audit.py \
  --output-dir /all/yiran07-disk3/huteng_data/exp/wan22_gpu0_timing_audit_20260906
```

The external result directory holds the reviewed video table, source SHA manifest, estimates, matched-path strata, per-GPU and time-bin diagnostics, a runnable companion notebook, and the report. The reference assumption is GPUs 1/2/3 within the same experiment condition; candidate prompts differ between GPUs, so matching controls measured computation but is not a randomized thermal intervention. The independent same-prompt baseline repeat is a separate check across dates.

`audit.py` requires a fresh result directory. For an existing audit, `build_evidence.py --result-dir <result-directory>` independently checks it using SQLite, verifies source hashes, writes `VALIDATION.json` and `audit.ipynb`, and authors canonical `artifact.json`. Execute with the same interpreter and four thread variables as above. The current environment lacks Jupyter dependencies; the notebook code cells were executed sequentially with Python, not through a Jupyter kernel. `audit.sqlite` materializes only the derived report datasets; original measurements remain untouched.

The Data Analytics plugin's `npm run report:deliver -- --input <artifact.json> --output <report.html>` packages the portable report. Numerical validation passed. No installed Chromium headless-shell was available, so the delivered HTML passed structural checks and retains semantic chart tables; interactive browser QA was unavailable.

`prompt_variation.py --result-dir <audit-directory>` is a bounded follow-up that measures per-prompt runtime dispersion on GPUs1/2/3. It writes `healthy_prompt_variation.json` and by-GPU/by-path CSVs; same-card/schedule leave-one-out comparisons exclude strata smaller than five. Use the same interpreter and thread environment.
