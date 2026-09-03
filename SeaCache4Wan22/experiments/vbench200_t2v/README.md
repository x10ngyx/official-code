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

如果推理启动时尚未就绪独立 VBench 环境，可使用 `--defer-evaluation`
先完成视频生成与性能汇总；此时 `status.json` 保持评测 pending。环境
就绪后用原命令加 `--resume`、去掉 `--defer-evaluation`，已完成推理会被
严格校验并跳过，然后继续 VideoMetrics、VBench 和最终报告。

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

`launch_threshold_suite_024_038_055.sh` 是用户选定 `.24/.38/.55` 后的四卡队列
入口：三个 threshold 顺序使用 GPU0--3。每档先并发启动 GPU0/1；只有两者都写出
`pipeline_initialization_count=1` 的 ready 状态并达到显存门槛，确认越过一次性权重读取/
主存峰值后，才并发启动 GPU2/3。每个 GPU worker 只构造一次 `WanT2V`，随后以
batch size 1 顺序处理其静态 shard；每个样本仍重新安装并恢复独立计时 profiler，
SeaCache controller 也逐样本新建，状态不会跨视频。
后两个任务只读复用 `.24` 任务的 matched baseline，不重复生成 `400` 个
baseline 视频。该队列先使用
`--defer-evaluation` 完成推理与性能汇总，评测环境就绪后再按上述恢复流程
生成 VideoMetrics、VBench 和正式报告。

修复仅影响 candidate 路径、且需要复用已完成 baseline 时，可设置
`WAN22_BASELINE_SOURCE=/path/to/completed/baseline`。运行器会逐一校验 200 个视频与
timing、四个 shard 配置、prompt/协议/线程环境和 prepared manifest；仅当旧、新
prepared 文件唯一差异为 `wan/modules/model.py`，且旧内联 norm1 表达式与新 helper
在 AST 上等价时才允许只读软链接复用，并把全部来源哈希写入 `run_config.json`。
旧 baseline 可以保留原逐样本进程生命周期，因为其
`pipeline_generate_wall_seconds` 本来就排除管线构造/权重读取和 MP4 导出；新的
SeaCache candidate 使用持久 worker。两者计时区间定义不变，但进程级 cold/warm 状态
可能带来轻微偏差，因此该生命周期差异会同时写入 `run_config.json` 和最终报告，不能
把一次性初始化节省计入或宣称为推理加速。

若某档生成已完成、只在后处理失败，可先验证并完成该档后处理，再设置
`WAN22_START_THRESHOLD=0.38` 或 `0.55` 从后续档继续。跳过 `.24` 时必须显式提供
`WAN22_BASELINE_SOURCE`，防止意外生成或选错 baseline。
