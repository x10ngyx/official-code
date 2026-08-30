# SeaCache4Wan22 Vbench200 实验

该实验比较锁定 Wan2.2-T2V-A14B no-cache baseline 与本包的干净 SeaCache 实现。
两侧使用 Vbench200 相同的 200 条英文 prompt、seed=42 和固定生成协议；阈值必须
显式给出。实验入口仅使用 `timestep_cache none/seacache`，不会带入 block cache、
CFG cache、ZEUS 或学习策略。

## 固定协议

| 项目 | 值 |
| --- | --- |
| 视频 | 832×480，45 帧，16 fps |
| 采样 | 50-step DPM++，shift=12 |
| CFG | low/high=(3,4)，boundary=0.875 |
| 模型 | Wan2.2-T2V-A14B，DiT BF16 |
| 内存 | 单卡 model offload，T5 GPU |
| 样本 | 每条 prompt 一个视频，seed=42 |

## 准备与运行

先按照项目根 README 用 `scripts/prepare_wan22.sh` 得到带
`.seacache4wan22_prepared.json` 的锁定源码。激活 `wan2.2` 环境并安装
`calflops==0.3.2`，或传 `--calflops-source` 指向锁定 commit `027e89a` 的源码；
VBench 可通过独立 Python 运行。结果必须写到
`/all/yiran07-disk3/huteng_data/exp`。

```bash
conda activate wan2.2

python run_vbench200.py \
  --output-dir /all/yiran07-disk3/huteng_data/exp/seacache4wan22_vbench200_thr_0p2 \
  --threshold 0.2 \
  --wan22-root /path/to/prepared-wan22 \
  --checkpoint-dir /path/to/models/Wan2.2-T2V-A14B \
  --gpu-ids 0 1 2 3 \
  --vbench-python /path/to/vbench-env/bin/python
```

脚本依次运行 baseline 与 SeaCache，多卡仅做 Vbench200 prompt 的静态分片。使用同一
命令加 `--resume` 可恢复；`--dry-run` 检查命令；`--limit N --skip-evaluation`
仅用于生成/汇总链路检查，不能汇报正式 Vbench200 分数。运行器会在本项目
`experiment_results/` 下建立到外部结果目录的软链接。

## 指标口径

- 推理时间：每视频 `pipeline_generate_wall_seconds`，包含文本编码、denoising、
  SeaCache/CFG/scheduler、generate 内权重传输/offload 和 VAE decode；不含 pipeline/
  model 构造、MP4 导出与评测；另保存 T5、DiT、VAE decode 的 CUDA 时间和 host span。
- TFLOPs：分别 profile A14B high/low DiT（固定 32/18 step），补偿 Calflops 看不到的
  dense FlashAttention core，再对每条视频的 50 个真实 SeaCache step trace 按两个
  CFG 分支累计。headline 是 estimated DiT TFLOPs，不是 TFLOP/s；另保存两次 UMT5
  encoder forward 与一次 VAE decode 的 TFLOPs。SeaCache FFT/gate、residual add、
  scheduler 和导出不在这些组件计数内。
- PSNR/SSIM/LPIPS：只调用 `VideoMetrics/run_evaluation.sh` 的
  `rgb_full_reference_v1`，baseline 是相同 seed 的 paired reference。
- VBench：baseline 与 SeaCache 都只调用 `VbenchEvaluation/run_vbench200.sh`；结果
  必须称 **Vbench200 subset score**。

最终生成 `benchmark_report.json`、`benchmark_report.csv` 和
`benchmark_report.md`，同时保留逐视频 timing、gate trace、TFLOPs 和三套评测的原始
JSON。各脚本职责与 Wan2.1 目录同名文件一致。
