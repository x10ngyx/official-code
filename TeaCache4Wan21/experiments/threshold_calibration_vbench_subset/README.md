# VBench200 子集 threshold 校准

本实验用少量、固定且跨类型的 VBench200 prompt 实测 Wan2.1-T2V-1.3B TeaCache
threshold 到推理时间加速比的映射。结果必须写入
`/all/yiran07-disk3/huteng_data/exp/teacache_wan21_threshold_calibration/`；代码目录不保存
视频或大体积日志。

## 固定协议

- 模型：Wan2.1-T2V-1.3B，参数 bfloat16；
- 生成：832×480、81 帧、16 fps、50-step UniPC、shift 5、CFG 5、seed 42；
- TeaCache：锁定官方正阈值实现，关闭 retention steps；
- prompt：固定 11 条 Vbench200 样本
  `001/026/051/056/076/085/101/126/151/176/177`，联合覆盖全部 16 个
  VBench 维度；
- 推理时间：timing JSON 的 `pipeline_generate_wall_seconds`，范围是模型加载完成后的
  文本编码、去噪和 VAE 解码，不含模型加载和 MP4 写盘；
- 汇总加速比：所有 prompt baseline 推理时间之和除以同一 condition 推理时间之和，
  不平均逐 prompt 加速比；最终同时保存 T5/DiT/VAE 分项时间、DiT/T5/VAE
  TFLOPs、统一 PSNR/SSIM/LPIPS 与 Vbench200 subset score。

11 条 prompt 按顺序静态分配给 `--gpus` 给出的 worker；baseline 与所有 threshold
使用相同 shard 映射，以降低 GPU 间差异。粗扫后根据聚合曲线选取细扫点，最后在独立
verification 目录重跑 baseline 和三个候选阈值。

## 运行

```bash
env_python=/path/to/wan2.2/bin/python
result_root=/all/yiran07-disk3/huteng_data/exp/teacache_wan21_threshold_calibration

"$env_python" run_calibration.py \
  --output-root "$result_root/coarse" \
  --wan21-root /path/to/locked/Wan2.1 \
  --ckpt-dir /path/to/models/Wan2.1-T2V-1.3B \
  --thresholds 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.24

"$env_python" analyze_calibration.py \
  --result-root "$result_root/coarse" \
  --calflops-profile /all/yiran07-disk3/huteng_data/exp/<profile>/calflops_profile.json \
  --model-cache /path/to/models/torch-cache \
  --vbench-python /path/to/vbench-python \
  --vbench-cache /path/to/models/VBench
```

细扫时再次调用 `run_calibration.py --resume --thresholds ...`，随后重跑分析脚本。
脚本拒绝覆盖已有视频；中断后必须用 `--resume` 继续。

## 文件结构

- `run_calibration.py`：按给定 GPU 列表运行 baseline 和任意 threshold 列表；
- `analyze_calibration.py`：验证 timing/Calflops 完整性，调用统一 VideoMetrics 与
  VbenchEvaluation，并输出 `summary.csv`、`summary.json` 和 `REPORT.md`；
- 外部结果的 `baseline/`、`threshold_*/`：视频、逐样本 timing、日志和生成配置；
- 外部结果的 `analysis/`：加速比曲线、目标最近点和插值建议。
