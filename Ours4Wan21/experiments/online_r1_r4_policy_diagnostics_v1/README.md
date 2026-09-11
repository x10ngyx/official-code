# R1–R4 collection policy diagnostics

`analyze.py` audits 400 sealed collection traces, quality and timing records, then exports trajectory-level and round/budget summaries under the existing online run's `analysis/r1_r4_policy_diagnostics/`. No live training source changes or new GPU jobs.

Primary definitions: steps 1–25 / 26–50, total skip count K, longest consecutive skip within each window (runs crossing step 25 are clipped), full-path longest run, number of skip runs, per-trajectory mean run length, skip-position centroid. Exact-K standardization uses the pooled four-round K distribution restricted to common support. It controls budget mix, not prompt difficulty or action-sampling noise. Budget bands 17–23 / 24–30 / 31–37 provide readable supporting cuts.

Report contract: technical audience, portable canonical HTML; summary and exact mean table, front/back allocation, run concentration, same-K robustness and limitations. Charts: grouped bars for front/back counts (8 rows); line profiles across 50 denoising steps (200 rows, probability 0–1); grouped run-length distribution bins. Round identity uses categorical palette with visible legend; all charts retain counts and denominators. Four rounds alone are not treated as a time-series trend. Companion notebook independently recounts CSV metrics top-to-bottom. Source hashes and QA remain supporting artifacts.
