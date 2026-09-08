# Persistent Vbench200 suite

`run_vbench200.sh`/`.py` orchestrates the suite. `generate_vbench200.py` is an internal worker entry. Each assigned GPU constructs one pipeline, optionally warms up once, and processes batch-size-1 paired conditions; caches and profilers reset per video. Prompts are statically sharded so all conditions of one prompt use the same GPU.

Use `--source SOURCE --checkpoint CHECKPOINT --output-dir EXTERNAL_DIR --gpu-ids 0 --threshold .08 --retention-ratio .2`. Standard mode defaults to the sibling Vbench200 table and requires coverage of all 16 dimensions; `--prompts TABLE --vbench-mode custom` explicitly labels the custom ten-dimension raw score. PSNR/SSIM/LPIPS use sibling VideoMetrics. Resume validates inputs, source, profile, warmup, GPU assignment and output hashes. `COMPLETE.json` requires all quality stages.
