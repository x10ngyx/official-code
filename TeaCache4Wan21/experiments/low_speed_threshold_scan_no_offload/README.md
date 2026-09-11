# Low-speed TeaCache threshold scan (GPU T5, no offload)

这套实验在 Wan2.1-T2V-1.3B 的默认 GPU-T5、无 model offload 配置下，用正式扫描
相同的 10 个固定 VBench200 prompt 补扫 0.15 以下的 threshold，以完整推理时间校准
约 1.8× 的低加速档位。

默认网格为 `0.06, 0.07, 0.075, 0.08, 0.085, 0.09, 0.10`。分辨率、帧数、采样器、
步数、CFG 和 seed 与完整扫描一致。脚本复用已经实测的 no-offload baseline 和
Calflops profile，只生成新增 TeaCache 条件；每个条件保存视频、分模块时间、逐调用
cache trace、DiT TFLOPs/TFLOP/s，以及逐帧/逐视频 PSNR、SSIM、LPIPS。

运行：

```bash
/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python \
  run_experiment.py --resume
```

大型结果写入 `/mnt/hdd/xiongyuxiang/tmp/exp/teacache_wan21_vbench10_low_speed_no_offload_scan`；
项目内的 `experiment_results/low_speed_threshold_scan_no_offload` 只保存符号链接。

扫描完成后，`build_report_artifact.py` 从验证后的 `analysis/summary.csv` 和
`summary.json` 生成 canonical `artifact.json`、可复核 SQLite/SQL source，以及供
portable report builder 打包的技术报告输入。

`build_paper_comparison_artifact.py` 进一步把本次低阈值扫描与 SeaCache、BAG、SenCache
等论文中对 TeaCache 的复测按 equivalent NFE、相同 threshold 和相近 speedup 三种口径
对齐，输出独立的 `paper_comparison_artifact.json`、`paper_comparison_source.sql` 和
`paper_comparison_report.html`。该补充报告同时检查 DiT TFLOPs reduction、DiT CUDA
speedup 与完整推理 speedup 是否符合固定模块开销的 Amdahl 预期。
