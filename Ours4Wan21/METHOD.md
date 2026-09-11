# 正式方法：CNN+G1

用户确定 Ours4Wan21 正式方法为 **CNN+G1**，状态标识 `cnn_G1`。
顶层正式入口为 `main.py`，直接调用已完成训练和视频评测的版本化实现。
`train.py`、`generate.py`、`online.py` 保留旧 MLP 实验接口；它们不会自动加载 CNN checkpoint。
G2/G3/G4、SEA7、Dynamics128、16×20 CNN 和 G1特征+MLP 均为对照或消融。

## 方法定义与代码对应

| 项目 | 正式配置 / 实现 |
|---|---|
| 三个状态 | 当前输入 latent、前一步输入 latent、最近一次实际 recompute 的输入 latent；`experiments/cnn_mixed3500_v1/features.py` |
| G1特征 | 原始 latent 三状态；每个状态直接池化到 `16×4×8×8`，另对完整时间轴求均值和总体方差后分别空间池化到 `16×8×8` |
| SEA | G1的三个 latent 不做 SEA 滤波；仍拼接 SEA7 标量状态，不能描述为“完全不使用 SEA” |
| 输入 | 18432维池化特征 + 7维 SEA7，共18439维；step0新特征为零，历史按实际动作更新 |
| 归一化 | 仅 train 拟合逐坐标 mean/std，std下限1e-6；归一化后 FP16，再以 FP32送入网络；latent先 FP16→FP32对齐归档 |
| CNN | `experiments/cnn_mixed3500_v1/model.py::CNN`；3D路径与均值/方差共用的2D路径，通道32/32/64，拼接768维后融合到128维，再拼接SEA7 |
| 输出头 | 两层256维 LayerNorm+SiLU MLP；actor输出两动作，独立Q1/Q2/V各有自己的CNN编码器；actor参数量316034 |
| 训练 | mixed3500（random3000+Increase500），2800/350/350轨迹划分；200epoch、batch256、seed42、AdamW lr1e-4、weight decay .01 |
| IQL | `aggressive_a1_v1`：τ=.7、β=1.5、actor权重cap30、γ=1、target_rho=.999；终局absolute RGB PSNR奖励 |
| 选点 | 验证集e170–200普查，e171–199双侧稳定性选点；当前G1为e182，未用视频测试质量选点 |
| 推理 | `experiments/cnn_vbench20_v1/adapter.py::Policy/SelectedHistory/Controller/apply_policy`；复用 `ours4wan21/runtime.py` 的Exact-K和SeaCache接入 |
| 动作 | step0/49强制重算，K范围0–48；cond自由步查询actor一次，两个CFG分支共用动作并保留独立残差历史 |

“三分支CNN”沿用实验名称，实际网络实现为一个3D卷积路径和一个处理均值/方差拼接输入的2D卷积路径，不能写成三个独立CNN。

## 正式入口与权重

在 `Ours4Wan21/` 下使用既有 `wan2.2` 环境：

```bash
python main.py --help
python main.py verify
```

`verify` 在CPU核对选点SHA、CNN结构、G1身份、特征合同、固定推理协议、normalizer与epoch，加载真实actor并执行一次前向。
当前选点记录：`experiment_results/ours21_cnn_mixed3500_v1/analysis/G1/checkpoint_selection.json`。
当前权重：workspace的 `models/ours21_cnn_mixed3500_v1/G1/checkpoints/epoch_182.pt`。
完整权重与优化器保留原位置，未复制或重新训练。

```bash
# 仅在需要继续既有冻结mixed3500的G1训练时执行；需要已完成的初始化与cache。
CUDA_VISIBLE_DEVICES=0 python main.py train

# 仅接收既有CNN评测合同内、同一GPU的G1 candidate job列表。
CUDA_VISIBLE_DEVICES=0 python main.py generate --jobs /path/to/g1_jobs.json
```

`train` 固定调用原训练管线的 `train_group --index 0`，仅G1；它不是新数据集的通用训练CLI。
`generate` 拒绝空列表及任何非 `group=G1/state_mode=cnn_G1` 任务，随后使用原常驻GPU worker与冻结来源/SHA/同卡baseline检查。
job字段和身份必须来自既有 `ours21_cnn_vbench20_v1` 的配置；可从其 `jobs/gpuN.json` 按G1筛选，并保持其余字段原样。
该入口不自动创建新评测协议、标定、baseline或VBench评分；四组完整复现仍使用两个CNN实验目录的launcher。
远端根目录通过 `OURS4WAN21_WORKSPACE`、`OURS4WAN21_EXP_BASE` 配置，冻结记录内的绝对路径也需要符合远端实际布局。

## 推理与评测口径

Wan2.1-T2V-1.3B，batch1，832×480、81帧、16fps，50-step UniPC、shift5、CFG5、seed42；DiT BF16，单张48GB GPU，T5/DiT/VAE常驻GPU，关闭offload、FSDP/SP与prompt扩展。
主耗时为完整generate，主计算量为DiT TFLOPs；保留T5/VAE及predictor分项，predictor耗时已包含在generate中。
K23/29/35是预算，约1.77/2.23/3.00×是已有评测实测值。
当前G1同20prompt三档RGB PSNR为28.5239/25.3117/21.7181 dB；历史本批未计算VBench score，不能称为已完成VBench评分。
正式质量评测遵循项目约定的VideoMetrics PSNR/SSIM/LPIPS与vbench_score要求；用户明确缩小范围的历史实验保留其原口径。

结果与比较见 `experiment_results/ours21_cnn_vbench20_v1/analysis/rgb_comparison20/`。
版本化CNN源码位于 `experiments/` 是为保持已有checkpoint、来源哈希与在途消融可复现；主入口直接复用这些文件，未另造一份网络实现。
