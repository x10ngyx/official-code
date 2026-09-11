# VBench10 full-metrics threshold scan

This experiment calibrates TeaCache thresholds against complete Wan2.1 pipeline
inference time on ten fixed VBench200 prompts. It retains matched baseline and
candidate videos, synchronized module timing traces, actual TeaCache block
traces, Calflops-based DiT TFLOPs/TFLOP/s, and per-frame/per-video RGB PSNR,
SSIM, and AlexNet LPIPS.

The locked threshold grid is dense near the expected 1.8× and 2.4× regions and
sparse in the high-threshold tail:

```text
0.15 0.16 0.17 0.18 0.19 0.20
0.22 0.24 0.26 0.28 0.30 0.32 0.34 0.36
0.44 0.56 0.68 0.80
```

`run_scan.py` is the resumable orchestration entry point. It profiles DiT FLOPs,
generates the baseline and all threshold conditions on four GPUs, evaluates all
matched video pairs, and invokes `analyze_scan.py`. The analyzer rejects missing
videos, timing traces, calls, module stages, metric rows, or frames before it
writes aggregate results.

```bash
/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python \
  run_scan.py \
  --output-root /mnt/hdd/xiongyuxiang/tmp/exp/teacache_wan21_vbench10_full_threshold_scan
```

Resume an interrupted run with the same command plus `--resume`. Individual
phases can be selected with `--phases profile`, `generate`, `metrics`, or
`analyze`. `--dry-run` prints all generation/evaluation commands without writing
results.

The headline speedup is
`sum(baseline pipeline_generate_wall_seconds) / sum(candidate pipeline_generate_wall_seconds)`.
It includes text encoding, denoising, and VAE decode, while excluding model
loading, MP4 export, and metric evaluation. TFLOPs are DiT-only operation counts;
T5 and VAE are represented by their measured wall time.

The completed result is indexed by
`../../experiment_results/vbench10_full_threshold_scan`. The validated outputs
are under `analysis/`: `summary.csv` contains all 18 aggregate rows,
`per_prompt.csv` contains matched prompt-level timings, `summary.json` preserves
the complete protocol and target selection, `data_quality.json` records the
completeness checks, and `REPORT.md` is the concise readout.

For the requested end-to-end latency definition, threshold 0.20 is the nearest
observed 2.4x point (2.3930x), and 3.0x is interpolated at approximately 0.48
between thresholds 0.44 and 0.56. The 1.8x target is not bracketed: even the
lowest requested threshold, 0.15, reaches 2.0780x. Because T5 time varied across
sequential conditions despite being threshold-independent, the report also
saves a fixed-module-normalized sensitivity analysis; it is diagnostic and does
not replace the requested raw end-to-end timing result.
