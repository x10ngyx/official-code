# TeaCache4Wan21 Vbench200 四卡测试

本实验在锁定的原始 Wan2.1 上对比 no-cache baseline 与官方 TeaCache Wan2.1。
两侧使用相同的 200 条 prompt、seed 和生成设置；仅 candidate 启用 TeaCache，
`threshold` 是运行时必传参数。PSNR、SSIM、LPIPS 和 VBench score 只调用本仓库
`VideoMetrics/` 与 `VbenchEvaluation/`，不调用 TeaCache 自带评测代码。

## 固定推理协议

| 项目 | 固定值 |
| --- | --- |
| 模型 | Wan2.1-T2V-1.3B |
| 画面 | 832×480、81 帧、16 fps |
| 采样 | 50 步 UniPC、shift=5、CFG=5、seed=42 |
| 精度 | DiT BF16（锁定 Wan2.1 config） |
| 显存 | model CPU offload 开启、T5 CPU offload 开启 |
| 数据集 | Vbench200，英文 prompt，每个 prompt 一个样本 |

四张 GPU 按 job ordinal 模 4 静态分片。先用四卡完成 baseline，再用相同四卡完成
TeaCache，避免两个 condition 争抢同一 GPU。每条视频都通过独立进程运行统一入口，
因此不会把 TeaCache 状态带到下一条 prompt；模型加载和 MP4 保存不计入推理 latency。

## 一条命令运行

```bash
python_bin=/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python
result_root=/mnt/hdd/xiongyuxiang/tmp/exp/teacache_wan21_vbench200_threshold_0p08

"$python_bin" run_vbench200_4gpu.py \
  --output-dir "$result_root" \
  --teacache-thresh 0.08 \
  --gpu-ids 0 1 2 3 \
  --video-metrics-python /path/to/python-with-torch-and-lpips \
  --vbench-python /path/to/python-with-vbench
```

`--threshold` 是 `--teacache-thresh` 的等价接口。阈值必须是非负有限数；
T2V-1.3B 官方无 retention steps 的参考值为 slow `0.05`、fast `0.08`。
若需要复现官方 retention 版本，可额外传 `--use-ret-steps`（官方 fast 参考阈值
为 `0.10`）。正式结果目录应把阈值写进名称，避免不同配置互相覆盖。

生成与 Calflops profile 默认使用已解包的 Wan2.2 Python。质量评测 Python 必须能
导入 `torch`、`lpips`，VBench Python 必须能导入 `torch`、`vbench`；启动器在正式
评测前会主动检查。VBench 权重和 LPIPS cache 默认读取本工作区的
`models/VBench/` 与 `models/torch-cache/`，也可用对应命令行参数覆盖。

建议先做无写入 dry-run：

```bash
"$python_bin" run_vbench200_4gpu.py \
  --output-dir "$result_root" \
  --threshold 0.08 \
  --dry-run
```

中断后使用完全相同的参数加 `--resume`。恢复逻辑只跳过同时具有非空 MP4 和成功
timing JSON 的样本；配置或计时不完整时会失败关闭，不把不完整结果纳入汇总。
用于小规模校准时，可向底层 `generate_vbench200.py` 传
`--sample-ids vbench200_001 vbench200_016 ...` 显式选择固定子集；该参数与
`--limit` 互斥，未知或重复 ID 会直接报错。

## 指标与性能口径

- latency：每视频 `pipeline_generate_wall_seconds`，包含文本编码、denoising 和 VAE
  decode；不包含 pipeline/model 加载、MP4 保存及指标计算。汇报 200 条视频的 mean、
  p50、p90、min/max 和总体分布。
- TFLOPs：用 Calflops 0.3.2 对真实 832×480、81 帧 Wan DiT forward profile；对
  Calflops 看不到的 FlashAttention core 使用本仓库 `CalflopsEvaluation` 的 dense
  attention 公式补偿，再按每条视频实际 transformer block trace 累加。TFLOPs 是
  `10^12` 浮点运算量；另以运算量除 DiT CUDA-event 时间报告估算 TFLOP/s。
- PSNR/SSIM/LPIPS：baseline 视频作为 paired full-compute reference，使用仓库
  `VideoMetrics` 的 `rgb_full_reference_v1` 固定协议。
- VBench score：reference 和 candidate 都计算 16 个维度；维度拆成四个 GPU shard，
  最后按仓库锁定的官方公式聚合。必须称为 Vbench200 subset score，不能称完整
  VBench leaderboard score。

## 结果结构

```text
result_root/
├── baseline/{videos,timings,logs}/
├── teacache/{videos,timings,logs}/
├── performance/
│   ├── calflops_profile.json
│   ├── per_video.jsonl
│   └── summary.json
├── evaluation/
│   ├── video_metrics/summary.json
│   ├── vbench_reference/vbench200_aggregate_scores.json
│   └── vbench_candidate/vbench200_aggregate_scores.json
├── orchestration_logs/
├── benchmark_report.json
├── benchmark_report.md
├── run_config.json
└── status.json
```

启动器会在本项目 `experiment_results/` 下建立到外部结果目录的同名软链接。

## 文件说明

- `run_vbench200_4gpu.py`：正式一键编排入口。
- `generate_vbench200.py`：单 condition、可静态分片的生成与逐样本性能 trace。
- `profile_calflops.py`：固定 full/reuse forward 成本 profile。
- `aggregate_performance.py`：按真实 cache trace 汇总 latency、TFLOPs、TFLOP/s。
- `evaluate_results_4gpu.py`：仓库 VideoMetrics 与四卡 VBench 编排。
- `build_final_report.py`：把全部指标合并为一份 JSON 和 Markdown headline 报告。
- `evaluate_results.sh`：保留的单卡/conda 手动评测入口；正式四卡流程不使用它。
