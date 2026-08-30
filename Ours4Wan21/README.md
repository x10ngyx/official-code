# Ours4Wan21

Wan2.1-T2V-1.3B experiments for the learned cache controller.  The first
implemented subproject is `data_collection/`, which builds randomized
per-step SeaCache-threshold behavior trajectories for offline training and a
matched fixed-threshold SeaCache dataset from the same OpenVid prompt pool.

Large generated artifacts must live below
`/all/yiran07-disk3/huteng_data/exp`; `experiment_results/` contains only
README files or symlinks to those external archives.  Model weights remain
under the workspace `models/` root.

## Structure

- `data_collection/`: immutable plan/runnable manifests, Wan2.1 runtime,
  timing/TFLOPs aggregation, PSNR/SSIM/LPIPS collection, publication, audit,
  tests, and experiment launchers.
- `upstream_lock.json`, `NOTICE.md`: Wan2.1, OpenVid snapshot, SeaCache-method,
  metric, and Calflops provenance.
- `experiments/`: future Ours4Wan21 experiment families not owned by the data
  collection subproject.
- `experiment_results/`: external-result index.
- `PROGRESS.md`, `logs/`: local status and concise handoff records.
