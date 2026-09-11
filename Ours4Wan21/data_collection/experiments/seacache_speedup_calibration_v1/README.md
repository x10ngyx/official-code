# SeaCache speedup calibration

This experiment calibrates fixed SeaCache threshold and observed skip count
against full-pipeline inference speedup for the locked Wan2.1-T2V-1.3B
protocol. It uses ten deterministic OpenVidHD samples and a threshold scan on
four GPUs. Existing matched full-compute baselines from
`wan21_seacache_threshold_collection_v1` are referenced through a symlink; they
are not copied or regenerated.

The initial grid is `0.04/0.06/0.08/0.10/0.12/0.15/0.18/0.21/0.24`. If that
grid does not bracket the `1.5x-3.5x` target domain, the resumable high-range
extension scans `0.30/0.36/0.42/0.50/0.60/0.70` and merges both roots into one
fit. Every candidate records
CUDA-synchronized full-pipeline and T5/DiT/VAE timing, trace-weighted DiT
TFLOPs, T5/VAE TFLOPs, and the exact cond/uncond reuse decisions. Calibration
does not save videos or latents and does not run quality metrics because those
artifacts are outside the requested speed/skip fit.

Run the four-GPU scan with:

```bash
export WAN21_PYTHON=/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python
export WAN21_ROOT=/mnt/hdd/xiongyuxiang/tmp/data/source/Wan2.1-65386b2
export CHECKPOINT_DIR=/mnt/hdd/xiongyuxiang/tmp/models/Wan2.1-T2V-1.3B
export BASELINE_RUN_ROOT=/mnt/hdd/xiongyuxiang/tmp/exp/wan21_seacache_threshold_collection_v1
export RUN_ID=wan21_seacache_speedup_calibration_v1
./launch_4gpu.sh
```

Run the high-range extension after the initial scan with:

```bash
./launch_high_extension_4gpu.sh
```

The launcher is resumable. Results are written below
`/mnt/hdd/xiongyuxiang/tmp/exp/$RUN_ID` and indexed from
`../../experiment_results/$RUN_ID`. `analysis/threshold_summary.csv` is the
fit source; `analysis/linear_fits.json`, `analysis/calibration_report.md`, and
`analysis/calibration_fits.png` contain the fitted results. The generated
`analysis/speed_threshold_mapping.calibrated.json` is compatible with the
random-threshold manifest materializer.

Files:

- `run_calibration.py`: prepares the ten-sample manifest and runs one reusable
  Wan pipeline per GPU shard.
- `analyze_calibration.py`: validates all results, aggregates the scan, fits
  both linear relationships, and writes the calibrated mapping.
- `launch_4gpu.sh`: prepares, launches four workers, and runs analysis.
- `launch_high_extension_4gpu.sh`: supplements an unbracketed initial scan and
  analyzes both result roots together.
- `queue_launch.sh`: waits for four GPUs to remain idle for 60 seconds before
  invoking the launcher; it never signals or modifies another job.
- `test_analysis.py`: CPU-only regression tests for fitting and fit-point
  selection.
