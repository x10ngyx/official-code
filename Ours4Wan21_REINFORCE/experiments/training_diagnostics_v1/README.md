# Training diagnostics

Read-only analysis of committed REINFORCE batches: PSNR trends, exact-K standardized early/late comparison, repeated-prompt/same-K comparisons and optimization diagnostics. analyze.py writes CSV, summary JSON, source hashes and a scientific PNG/PDF figure below the run's diagnostics directory. No GPU inference, checkpoint edits or training-source changes.

`recent.py <snapshot-dir>` reads the frozen trajectories.csv and writes recent_summary.json: adjacent recent windows, K-fixed-effect descriptive slopes, and matched prompt/K pairs between adjacent epochs.
