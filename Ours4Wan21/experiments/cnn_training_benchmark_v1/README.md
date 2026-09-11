# Three-branch CNN training benchmark

`run.py prepare` extracts all four feature variants from one real training trajectory.
`run.py worker --group G1` runs the candidate from the static arithmetic estimate,
with a full-size in-memory timing fixture made by repeating that trajectory.
Four concurrent workers measure 2 full train/validation epochs after warmup.
The unchanged local IQL update includes target critics, actor weighting, gradient
clipping, optimizer updates and reporting synchronizations. Checkpoint serialization
is measured separately. This is throughput testing, not a valid learning experiment.
Results/cache live in HDD exp; disposable timing weights live in the shared models root.

Three-dimensional pooling and full-time mean/variance followed by spatial pooling
are separate branches. Variance uses correction=0 before spatial pooling.
CNN architecture matches cnn_compute_estimate_v1: 3D/2D 32/32/64 channels,
fusion 768->128, SEA7, MLP 135->256->256->output. All IQL networks are independent.
