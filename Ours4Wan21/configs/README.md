# Configuration reference

training.json exports the frozen local 400-epoch settings; it is not a CLI override. speed_to_k.pending.json documents an empty Wan21 empirical mapping and must fail inference until genuinely calibrated. Ten concrete sea7_<group> modes are registered in ours4wan21/latent_features.py; their dimensions and causal contracts are frozen in checkpoints. The generic sea7_latent placeholder still requires an explicit group.
