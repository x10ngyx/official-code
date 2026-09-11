# Ours4Wan22 — CNN+G1

正式 Wan21 CNN+G1 的 Wan2.2-T2V-A14B 方法实现。入口为 `main.py`；依赖同仓库 Ours4Wan21、SeaCache4Wan22、ComponentMetrics、VideoMetrics、VbenchEvaluation，引用文件的 SHA256 固定在 `source_lock.json`。

**VBench整套测试入口：** [experiments/vbench_speed_targets_v1/run.sh](experiments/vbench_speed_targets_v1/README.md)。参考Wan21流程，串联固定五prompt或完整200集、多个目标加速比、一次共享baseline、VideoMetrics及VBench评分和报告。支持新标定、`--dry-run`、已完成阶段`--resume`和严格同卡baseline复用；无需手工依次调用generate/evaluate。

## 方法

- Raw latent 为 `[16,12,60,104]`。当前输入、前一步输入、最近实际 recompute 输入各先 FP16→FP32；各直接池化为 `16×4×8×8`，另对完整时间轴求 mean/population variance 后各池化为 `16×8×8`。三个角色的 3D 特征先拼接，再拼接各角色 mean/variance，最后接 SEA7，共 18439 维。G1 latent 不做 SEA 滤波。
- SEA7 顺序：相邻 filtered relative L1、包含当前距离的累积量、cached_valid、step/49、K/50、used/50、consecutive/50。SEA 来自首个 block 的 modulated input，沿用干净 Wan22 SeaCache 的滤波和 shared gate。边界距离置零，实际重算后 accumulator 归零。
- 第 0/32 步重置 latent 历史，新 latent 特征为零；第 0/32/49 步强制重算。K 为全视频 skip 步数，范围 0–47；跨专家保留全局 used，两个 CFG 分支共享动作、分别存残差。每个自由步查询 Actor 一次；预算强制动作及原生强制动作的 actor_mask=0。
- 网络直接执行 Wan21 正式 `model.py::CNN('G1', ...)`：3D 与 mean/variance 合并 2D 两条卷积路径（32/32/64），768→128 fusion，拼接 SEA7，再接两层 256 Linear–LayerNorm–SiLU。Actor 316034 参数。Actor/Q1/Q2/V 各有独立编码器，两个 target-Q 为独立 EMA 副本。
- 仅 train 拟合逐坐标 population mean/std，std floor=1e-6；归一化后 FP16，再 FP32 入网络。Wan21 checkpoint、旧 Wan22 MLP/CNN checkpoint 不能直接作为新方法权重，严格 loader 会拒绝错误合同。
- Exact-K 和 IQL 更新复用本地实现的冻结版本 `Ours4Wan21/ours4wan21/local_iql.py`。训练超参数按用户要求与正式 Wan21 CNN+G1 对齐：200epoch、batch256、AdamW lr1e-4/wd .01、tau .7、beta1.5、cap30、gamma1、target rho .999、clip1、seed42，终局 absolute RGB PSNR。数据使用本地纯随机2323，保留1851/472训练/验证划分；不使用Wan21 mixed3500或其选点epoch。

## 运行

```bash
conda activate wan2.2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
cd work/offical-code/Ours4Wan22
python main.py --help
```

固定模型协议为 832×480/45帧/16fps、50-step DPM++、shift12、low/high CFG=(3,4)、boundary=.875、seed42、DiT BF16、batch1/单卡、offload_model=True、t5_cpu=False，关闭 FSDP/SP 与 prompt extension。所有权重必须位于 workspace/models；所有输出位于 `/all/yiran07-disk3/huteng_data/exp` 并自动建立 experiment_results symlink。

先准备经过锁验证的共享 Wan22 树（本机已有 `../SeaCache4Wan22/build/Wan2.2-42bf4cf-prepared-modnorm-fix`）：

```bash
bash ../SeaCache4Wan22/scripts/prepare_wan22.sh ../SeaCache4Wan22/build/ours22-prepared
python main.py profile --wan22-root ../SeaCache4Wan22/build/ours22-prepared   --checkpoint-dir ../../../models/Wan2.2-T2V-A14B   --output /all/yiran07-disk3/huteng_data/exp/ours22_profile/profile.json
```

Profile 直接调用共享 Calflops，含 DiT high/low cond/uncond 及 T5/VAE；也可复用同模型、同 source lock、同 shape 的既有完整 v2 profile。为避免重复昂贵 profile，生成入口要求显式指定其路径。

`--jobs` 接收 JSON 列表：`[{"sample_id":"001","prompt":"A cat walks through a garden."}]`。同一调用的所有 prompt 在一个 pipeline 进程内顺序运行。正式 Vbench200 使用该数据集的原始 sample_id/prompt。每个 baseline/candidate 条件都先执行一个完整同协议 warmup，warmup 不进入结果。

```bash
python main.py generate --wan22-root ../SeaCache4Wan22/build/ours22-prepared   --checkpoint-dir ../../../models/Wan2.2-T2V-A14B --jobs /path/to/jobs.json   --profile /all/yiran07-disk3/huteng_data/exp/ours22_profile/profile.json   --output /all/yiran07-disk3/huteng_data/exp/ours22_baseline
python main.py generate --wan22-root ../SeaCache4Wan22/build/ours22-prepared   --checkpoint-dir ../../../models/Wan2.2-T2V-A14B --jobs /path/to/jobs.json   --profile /all/yiran07-disk3/huteng_data/exp/ours22_profile/profile.json   --policy ../../../models/ours22_training/epoch_200.pt --target-speedup 2.0   --output /all/yiran07-disk3/huteng_data/exp/ours22_target2p0
```

`--target-speedup` 默认使用 [新完整generate标定](configs/speed_to_k_full_generate_20260912.json)，支持1.5–3.5×；示例2.0×选择K27，epoch200仅示例，不声明其为最佳checkpoint。该表合并GPU0修正后的600条SeaCache结果与GPU1/2/3三个prompt的17条补测，保留实测K点并对中间缺失K按归一化latency（1/speedup）线性插值；按最近速度选择K，平局选较小K。超出默认支持范围会报错。

| 目标speedup | 1.5 | 1.8 | 2.0 | 2.2 | 2.4 | 2.5 | 2.6 | 2.8 | 3.0 | 3.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| K | 18 | 24 | 27 | 29 | 31 | 32 | 33 | 35 | 36 | 38 |

其中K32为插值估计。标定源是SeaCache，未包含Ours的CNN/latent-feature开销，实际Ours speedup以测试计时为准。`run.json`/`manifest.json`保存所用标定完整内容、路径和SHA。可用`--target-speedup X --calibration FILE`显式覆盖；文件须满足`ours4wan22_speed_to_k_v1`、status=calibrated、protocol与configs/protocol.json一致、forced_steps=[0,32,49]及单调entries合同。也可用`--skip-budget K`直接指定预算，此时不读取标定。

## 计量与质量

完整 generate wall 为主 latency，包含 scheduler、feature/actor 和 GPU/CPU 权重搬运；不包含模型加载、MP4 导出、trace 序列化或质量评测。T5/DiT/VAE 使用共享 CUDA-event 计时，各调用数和实际 block 执行必须通过校验。DiT TFLOPs 由实际 high/low、CFG、full/reuse path 与 Calflops profile 汇总。Actor 独立记录 CUDA/decision wall 和 Conv/Linear TFLOPs（24580608 FLOPs/次）；这些时间已包含在 generate 中，不能再相加。feature/SEA FFT、pooling、normalization、activation、residual add 和 scheduler FLOPs 未估算，不将其伪报为零或完整方法 FLOPs。

```bash
python main.py evaluate   --baseline /all/yiran07-disk3/huteng_data/exp/ours22_baseline   --candidate /all/yiran07-disk3/huteng_data/exp/ours22_target2p0   --output /all/yiran07-disk3/huteng_data/exp/ours22_target2p0_metrics   --metric-models ../../../models/evaluation --vbench-mode vbench200
```

此入口核验配对 prompt/protocol/source/profile/GPU UUID 和视频/trace/计时 SHA，调用仓库 VideoMetrics 的 RGB PSNR/SSIM/LPIPS，并分别计算 baseline/candidate 的 VBench，最终 summary.json 保留全部分项及 ratio-of-sums speedup。默认 vbench200 要求完整固定200集；任意 prompt 子集显式使用 `--vbench-mode custom`（10维 custom-input 均分，不能与16维 Vbench200 aggregate 混称）。不会静默跳过 VBench。

## 离线训练数据接口

训练文件为 `torch.save` 的字典，schema=`ours4wan22_cnn_G1_dataset_v1`，protocol/feature_contract 分别为 configs 下两个 JSON 的内容，另含：

- `states`: FP32 raw `[N,50,18439]`，必须是候选实际轨迹的 pre-action 特征；不要传旧 MLP state 或已归一化特征。
- `actions`: `[N,50]` 二元动作，0=recompute、1=reuse；K 从真实动作和计算。
- `terminal_psnr`: `[N]`，同 prompt/seed/protocol no-cache baseline 的 VideoMetrics absolute RGB PSNR。
- `prompt_ids`、`split`: N 个 prompt 标识与 train/val/test 标签，同一 prompt 的轨迹不能跨 split。train 和 val 必须非空。

构建 states 时可将生成保存的 `features/<id>.pt` 与 `traces/<id>.json` 中 `decisions[].state` 逐步拼接；其他采集轨迹应逐步调用同一个 features.History.observe/commit，避免离在线特征漂移。训练器从实际动作重新计算 Exact-K mask，只在终局置 reward，并拒绝跨 split prompt 泄漏及旧 shape/合同。

```bash
python main.py train --dataset /all/yiran07-disk3/huteng_data/exp/ours22_dataset/dataset.pt   --weights ../../../models/ours22_training   --output /all/yiran07-disk3/huteng_data/exp/ours22_training
python main.py verify --policy ../../../models/ours22_training/epoch_200.pt
```

训练入口创建新 run，不自动选择最佳 epoch，不自动启动在线采集。`--smoke-only` 将测试训练权重标记为禁止正式推理。已提供 `experiments/random2323_cnn_g1_v1/` 正式任务：从既有纯随机2323无损packed latent构建G1 cache、重算既有视频RGB PSNR后启动上述200epoch训练；运行状态见本项目PROGRESS.md。该流程不收集新数据。

## 验证与目录

`tests/` 覆盖所有 K/极端 actor、历史与边界、正式网络逐值一致和 FLOPs、checkpoint 合同、真实 IQL 梯度更新、组件计量拒绝规则、scoped hook 恢复，以及 prepared WanModel 原始 forward 的 CPU 小模块 K0 数值等价。CPU fixtures 不等于完整 14B GPU 验证；真实 Wan 推理、latency/TFLOPs profile 与视频质量结果尚未运行。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1   python -m unittest discover -s tests -v
```

`ours4wan22/` 为方法/训练/生成/评测模块；`configs/` 保存协议与特征合同；`experiments/fixed_protocol/` 为单线程 launcher；`experiment_results/` 保存外部结果 symlink；`PROGRESS.md` 与 `logs/` 保存本地交接状态。
