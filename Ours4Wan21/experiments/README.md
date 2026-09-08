# Ours4Wan21 experiments

- rl_training_v1/: frozen local offline IQL launcher.
- rl_inference_v1/: fixed Wan21 resident inference launcher.
- rl_validation_v1/: synthetic CPU end-to-end validation.

Collection experiments remain under ../data_collection/experiments/. All results live in the external experiment root; model weights live under models/.

- rl_cache_v1/: build scalar caches from the frozen 3000-random selection.
- rl_analysis_v1/: all-epoch metrics/plots and post300 validation-only checkpoint selection.

- `latent_groups_v1/`: remote scalar5/SEA7 controls plus ten individual SEA7+latent groups; one-pass feature cache, training, selection, VBench and CPU validation.
