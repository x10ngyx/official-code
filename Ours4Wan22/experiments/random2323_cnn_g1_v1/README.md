# Random2323 CNN+G1 training

Uses only the frozen 2323 pure-random trajectories (1851 train/472 validation; 80/20 prompt split preserved). Builds exact raw G1 pooled states from existing lossless packed latents, recomputes terminal RGB PSNR from existing videos with VideoMetrics, fits train-only normalization and launches 200epoch aggressive_a1_v1 IQL matching formal Wan21. No generation/data collection.

Run launch.sh under wan2.2; four thread limits and GPU UUID are explicit. Source hashes, cache completion, reward completion and TRAINING_STARTED.json guard phase transitions. Weights go to models, outputs to external exp with a project symlink.
