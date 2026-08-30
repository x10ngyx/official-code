# SeaCache4Wan21 Vbench200 实验

该实验比较锁定 Wan2.1-T2V-1.3B no-cache baseline 与 corrected
filtered-boundary SeaCache 实现。后者保留官方分支独立 gate/filter/residual
结构，但强制步也保存 SEA-filtered feature，避免下一步混合 filtered/raw
表示计算 relative-L1；因此不应标作官方 raw-boundary 实现的逐行为复现。
两侧使用 Vbench200 的同一组 200 条英文 prompt、seed=42 和完全相同的生成协议；
阈值必须显式给出。脚本不会启用或导入 block cache、CFG cache、ZEUS 或学习策略。

## 固定协议

| 项目 | 值 |
| --- | --- |
| 视频 | 832×480，81 帧，16 fps |
| 采样 | 50-step UniPC，shift=5，CFG=5 |
| 模型 | Wan2.1-T2V-1.3B，DiT BF16 |
| 显存 | 单张 48GB GPU；model、T5 与其他模型组件全部常驻 GPU，禁用任何 offload |
| 样本 | 每条 prompt 一个视频，seed=42 |

## 运行

使用 Wan2.1 环境，并保证其中安装 `calflops==0.3.2`；也可传
`--calflops-source /path/to/calculate-flops.pytorch@027e89a` 使用锁定源码而不修改
共享环境。VBench 可以使用单独环境，通过 `--vbench-python` 指定其 Python。结果目录必须位于
`/all/yiran07-disk3/huteng_data/exp` 下。

```bash
conda activate Wan2.1

python run_vbench200.py \
  --output-dir /all/yiran07-disk3/huteng_data/exp/seacache4wan21_vbench200_thr_0p2 \
  --threshold 0.2 \
  --wan21-root /path/to/Wan2.1-at-65386b2 \
  --checkpoint-dir /path/to/models/Wan2.1-T2V-1.3B \
  --gpu-ids 0 1 2 3 \
  --vbench-python /path/to/vbench-env/bin/python
```

脚本先并行生成 baseline，再生成 SeaCache，避免两组任务争抢 GPU。中断后以完全相同
参数增加 `--resume`。正式运行前可加 `--dry-run` 检查命令；小样本生成检查可使用
`--limit N --skip-evaluation`，该结果不能作为正式 Vbench200 分数。

运行器会在本项目 `experiment_results/` 下创建到外部结果目录的软链接。

## 指标口径

- 推理时间：每视频 `pipeline_generate_wall_seconds`，覆盖文本编码、denoising、
  SeaCache/CFG/scheduler、generate 内传输和 VAE decode；不含 pipeline/model 构造、
  MP4 导出和指标计算；另保存 T5、DiT、VAE decode 的 CUDA 时间和 host span。
- TFLOPs：Calflops 0.3.2 对真实 DiT shape profile，并补偿 Calflops 看不到的 dense
  FlashAttention core；再按照每条视频的 100 个真实 branch decision trace 累加。
  headline 是 estimated DiT TFLOPs（`10^12` 次操作），不是 TFLOP/s；另保存两次
  UMT5 encoder forward 与一次 VAE decode 的 TFLOPs。SeaCache FFT/gate、residual
  add、scheduler 和导出不在这些组件计数内。
- PSNR/SSIM/LPIPS：只调用 `VideoMetrics/run_evaluation.sh` 的
  `rgb_full_reference_v1` 协议，以同 seed baseline 为 reference。
- VBench：只调用 `VbenchEvaluation/run_vbench200.sh`，baseline 和 SeaCache 均计算；
  报告名称是 **Vbench200 subset score**，不是完整 VBench leaderboard score。

## 结果

最终目录包含 `baseline/`、`seacache/`、`performance/`、`evaluation/`、逐进程日志、
`benchmark_report.json`、`benchmark_report.csv` 和 `benchmark_report.md`。JSON 中保留
原始结果文件路径；`performance/per_video.jsonl` 保留逐视频时间与 TFLOPs。

## 文件

- `run_vbench200.py`：一键编排、静态多卡分片、恢复和结果软链接。
- `generate_vbench200.py`：单 condition 生成和逐样本 timing/trace。
- `profile_calflops.py`：固定真实 shape 的 DiT Calflops profile。
- `aggregate_performance.py`：按真实 SeaCache trace 汇总时间与 TFLOPs。
- `evaluate_results.py`：统一 VideoMetrics 和 VbenchEvaluation 入口。
- `build_final_report.py`：合并全部指标为 JSON/CSV/Markdown。
