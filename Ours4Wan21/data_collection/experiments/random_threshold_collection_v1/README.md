# Wan2.1 random-threshold collection v1

This experiment freezes a 3,000-prompt OpenVid sample and three randomized
SeaCache threshold trajectories per prompt.  The four shards are deterministic:
750 baselines and 2,250 candidates per GPU.  A pipeline is loaded once per
shard and reused serially.

Run `launch_4gpu.sh <phase>`. Supported phases are:

- `preflight`: verify this project and its shared `offical-code/` resources.
- `plan`: freeze the 9,000-row pending plan; no GPU is used.
- `profile`: create the real-shape Calflops profile on one GPU.
- `baselines`: collect the 3,000 shared full-compute references on four GPUs.
- `materialize`: combine the pending plan with a separately fitted calibrated
  speedup-to-mean-threshold mapping.  The checked-in pending mapping is
  intentionally rejected.
- `candidates`: require the runnable manifest and every baseline, then collect
  9,000 random trajectories on four GPUs.
- `publish`: atomically publish the longest contiguous completion prefix.
- `finalize`: require all 9,000 rows, publish, and run the archive audit.

Set `RUN_ID`, `WAN21_ROOT`, `CHECKPOINT_DIR`, and after fitting
`CALIBRATION_CONFIG` as needed. Set `EXP_BASE` when the remote result root is
not `/all/yiran07-disk3/huteng_data/exp`; the project result directory receives
only a symlink. Worker logs are part of that external result archive.

The bundled prompt pool is the default. PSNR/SSIM/LPIPS and manual FLOP
accounting use sibling `VideoMetrics/` and `CalflopsEvaluation/`. One LPIPS
AlexNet model is loaded per candidate worker and reused across that shard;
configure `METRICS_MODEL_CACHE`, `METRICS_DEVICE`, and `LPIPS_BATCH_SIZE` when
needed. Metric/decode time remains outside inference speedup. See
`REMOTE_DEPLOYMENT.md` for the external-input checklist.

The frozen source checker rejects a Wan2.1 tree whose four compatibility files
do not match commit `65386b2e03c490796eede31b0325a6a595cc684e`.

Candidate trace schema v2 records 100 ordered CFG branch calls. Each branch
row contains the SEA-filtered adjacent-step relative-L1 and accumulator
`before`, `with_current` (the threshold operand before reset), and `after`.
Published snapshots expose these rows in `tables/branch_transitions.jsonl` and
`tables/branch_transitions.csv`, in addition to the 50-row step view.
