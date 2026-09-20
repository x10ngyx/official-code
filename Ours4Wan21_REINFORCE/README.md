# Ours4Wan21_REINFORCE

**资源边界：本项目并非自包含。** 除模型/环境外仍依赖同仓库其他子项目及外部Wan源码、profile和可选baseline。发布与本机依赖版本差异见[DEPENDENCIES.md](DEPENDENCIES.md)，逐文件哈希见[dependency_inventory.json](dependency_inventory.json)。

独立的 **CNN+G1、随机初始化、on-policy REINFORCE** 消融。复用同级
`Ours4Wan21` 的冻结 CNN、G1 池化公式、Exact-K 与 SeaCache 残差缓存实现，
不修改原 IQL 项目，不接受 IQL 权重，不加载离线归一化统计。

## 已确认的实验设置

- 一次正式训练，8个epoch；每epoch将800个train prompt独立打乱并各生成一条新轨迹。batch=32，每epoch25次更新，共6400条、200次更新。每条轨迹独立、等概率采样整数 `K∈{20,…,40}`。
- Wan2.1-T2V-1.3B，832×480、81帧、16fps、UniPC50、shift5、CFG5、生成seed42、DiT BF16。
- 单卡offload：`offload_model=True, t5_cpu=False`；T5在GPU编码后移回CPU，DiT去噪后移回CPU。
- G1原始三状态8×8池化 + SEA7，共18439维；CNN actor参数316034，原网络LayerNorm保留。
- 批内冻结actor与输入mean/std，采集完整轨迹；每批只做一次optimizer更新，然后合并该批原始池化特征的累计统计，供下一批使用。
- 第一批mean=0、std=1。累计population mean/variance使用float64合并；std下限1e-6。标准化后FP16→FP32，与原CNN输入编码一致，无额外裁剪。
- 训练按actor分布采样，评测argmax；首尾及Exact-K强制动作不计策略梯度；cond查询一次，CFG共享动作、独立残差。
- 终局奖励为同级VideoMetrics定义的**解码RGB逐帧PSNR的均值**。不使用未编码RGB MSE代替奖励。
- 只保存训练相关池化特征、trace、PSNR、组件计量、checkpoint等；**不保存原始latent、逐步latent、原始RGB数组或训练视频**。

## 训练更新

每批B条轨迹的损失：

`L = -mean_i[(R_i - b[K_i]) * sum_{t: free} log π(a_it | s_it)]`

`b[K]`只来自此前批次：首次无历史为0，首次观测后设为该K的批内平均奖励，之后以
`0.9 * old + 0.1 * current_mean`更新。无Q/V、replay、离线预训练、熵奖励、优势标准化、PPO比率或多epoch复用。
AdamW lr1e-4、weight decay .01、clip norm1；γ=1。一个batch分16个决策行的小块累计梯度，
按轨迹数归一化，整批结束只step一次。batch size与epoch数（或旧模式批数）必须在prepare显式提供；训练seed默认42。

每条轨迹记录采集策略版本、冻结标准化后的输入与行为log-prob。更新前校验输入一致、
Exact-K mask、train split，并重放log-prob（GPU采集/CPU更新绝对容差3e-4）。旧策略、
旧批次、重复轨迹、评测argmax轨迹均不能进入更新。历史状态只用于累计输入统计，不进入策略梯度replay。

## 目录

| 路径 | 内容 |
|---|---|
| `main.py` | prepare / train / evaluate / test入口 |
| `experiments/reinforce_v1/` | 本次消融完整实现、测试及运行说明 |
| `experiment_results/` | 指向disk1结果目录的软链接 |
| `PROGRESS.md`, `logs/` | 当前状态和交接记录 |

使用现有 `/home/wangyue/miniconda3/envs/wan2.2/bin/python`，不另建环境。
结果只写 `/all/yiran06-disk1/wangyue_home/exp`，权重写对应`models`目录。

## 使用

```bash
cd /home/wangyue/xiongyuxiang/tmp/work/official-code/Ours4Wan21_REINFORCE
/home/wangyue/miniconda3/envs/wan2.2/bin/python main.py test
/home/wangyue/miniconda3/envs/wan2.2/bin/python main.py prepare --help
```

用户已将正式训练改为800训练prompt×8个epoch。原有放回采样先导运行已停止并保留；以下运行从seed42随机策略重新开始，四卡采集同一个策略：

```bash
/home/wangyue/miniconda3/envs/wan2.2/bin/python main.py prepare \
  --run /all/yiran06-disk1/wangyue_home/exp/ours21_reinforce_g1_seed42_epochs8_v1 \
  --epochs 8 --batch-size 32 --seed 42 --gpus 0,1,2,3
/home/wangyue/miniconda3/envs/wan2.2/bin/python main.py train \
  --run /all/yiran06-disk1/wangyue_home/exp/ours21_reinforce_g1_seed42_epochs8_v1
```

默认使用原Ours固定1000-prompt名单及800/100/100划分；仅train池参与采样/统计/梯度。
每epoch无放回遍历train池，下一epoch用最新策略重新生成轨迹。每批32条只更新一次，输入统计在epoch边界不重置。
STATUS.json报告已完成epoch、批数和轨迹数；epochs/保存每轮完成的checkpoint收据。自定义训练池不足整批时保留最后小批。
旧`--batches`模式继续支持有放回采样，与`--epochs`互斥。
默认读取现有compact1000同协议、同模型SHA、同物理GPU的baseline **MP4**，不打开其latent。
baseline不匹配可用GPU时，临时生成native参考视频，指标计算后删除。
可用`--prompts`提供带`sample_id,prompt,split`的JSON/JSONL；按ID和规范化文本检查跨split泄漏。
`--no-baseline-reuse`完全使用临时新参考。准备阶段不运行Wan。

GPU独立worker常驻进程，模型按offload合同在CPU/GPU间移动；每批全局收齐才更新。
单次运行只训练一个actor；多GPU只加速轨迹采集，非多seed、非FSDP。
prepare冻结配置/源码/权重/来源SHA；train同命令可恢复，已封存轨迹复核后复用。
checkpoint原子落盘是更新提交点，恢复不会重复应用梯度或重复合并统计量。
每条轨迹开始前保留30GiB及每worker1GiB预算；空间不足停止并保留完整产物。

## 评测与存储

训练结束仅生成`TRAINING_COMPLETE.json`，不表示质量评测已完成。最终checkpoint是主结果，
不沿用IQL验证稳定性选点。独立评测入口：

```bash
/home/wangyue/miniconda3/envs/wan2.2/bin/python main.py evaluate \
  --run /all/yiran06-disk1/wangyue_home/exp/ours21_reinforce_g1_seed42_epochs8_v1 \
  --name final_test20 --split test --count 20 --budgets 20 30 40
```

每档使用相同固定held-out prompt，按当前卡重新生成native参考并实测完整generate。
默认是原1000-prompt名单中的test子集，不是历史CNN报告的VBench20；比较IQL与REINFORCE时需要在同一名单、同一offload协议下评测双方。
计算PSNR/SSIM/LPIPS及VBench custom-input十维分数和本地raw-mean `vbench_score`，
**不是官方完整VBench或16维Vbench200分数**。VBench阶段需要既有依赖与本地权重。
完整质量指标落盘后删除临时评测视频；失败时临时视频保留以便恢复，可手动删除后用新的评测名重跑。
评测期间不更新输入统计；baseline与candidate的T5/DiT/VAE和predictor计量分别保存。
predictor/特征耗时已包含在DiT/generate，不能再次相加。特征提取FLOPs未计，明确为null。
训练复用的旧baseline包含诊断开销，仅用于PSNR，不据其耗时推导加速比。

每条训练轨迹仅约5.3MiB特征（50×18439的FP32池化值与FP16网络输入）加小型记录。
6400条特征约33GiB，另有trace/checkpoint等；训练临时MP4按每worker一条候选及必要参考保存，计算PSNR后立即删除。
源归档不修改、不删除、不复制大文件。原始latent仅在当前生成的内存中存在，用于构造G1，绝不落盘。

真实GPU验收可独立prepare `--smoke --batches 1 --batch-size 1`，再train。
它额外验证native与Exact-K K=0生成bitwise一致，并跑一条随机Actor轨迹及一次梯度更新。
smoke checkpoint带标记，禁止冒充正式结果。该验收与正式预算分开。

已完成本机验收：10项CPU测试通过；4090上native/K=0 bitwise一致，K33轨迹及一次更新成功，
行为log-prob重放最大差3.58e-7；重复启动不重新采集、不重复更新。产物检查无视频或原始latent残留。
证据见`experiment_results/ours21_reinforce_g1_gpu_smoke_v1/VALIDATION.json`。正式训练已在tmux `ours21_reinforce_g1_seed42_epochs8_v1` 启动；完整质量评测尚未运行。
