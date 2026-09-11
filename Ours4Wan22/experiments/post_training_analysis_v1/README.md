# Automatic checkpoint selection and training plots

`analyze.py --root RESULT --wait` waits for all 200 training epochs, then uses
the released GPU for a fixed full-validation free-decision-state census over
all 200 checkpoints. It runs no video generation or quality evaluation.

Q is min(online Q1, online Q2), with both actions flattened for mean and midpoint
P25/P75/IQR. V, action gap, actor agreement and adjacent D_Q/IQR are also recorded.
Plots include train/validation Q/V/actor loss, mean Q, Q-IQR, mean V, action gap,
normalized Q drift and actor stability. PNG/SVG/PDF, CSV and NPZ are retained.

All e180--e200 are selection candidates. The Wan21 actor-agreement gate and
normalized-D_Q ranking are reused; endpoints explicitly use one available
in-window neighbor, while interior candidates use both. No e179/e201 or
video-quality criteria are used. `selected.pt` is created under the model root;
an existing selection or analysis directory is never overwritten.

`POSTPROCESS_JOB.json` records waiting/running/completed/failed status. The
training process and completion marker are checked while waiting. Failures
stop this job and are reported without altering the original training outputs.
Use conda wan2.2 and all four BLAS thread limits set to 1.
