# Ours4Wan22 与本机实验实现对照

结论：排除用户明确更新的网络架构和 latent feature 设置后，核心离线 IQL、Exact-K 和 SeaCache 残差执行语义一致；当前 Ours4Wan22 尚不是本机完整实验流程的等价迁移。主要差异是标定生成与绑定、历史 reward/速度口径，以及选点和在线续训流程。另发现标准 VBench200 prompt 校验缺口。此次只审核，未修改方法实现。

## 对照范围与证据

- 新版：`Ours4Wan22/ours4wan22/{runtime,policy,training,generation,metrics,evaluation}.py`。
- 本机训练：`work/Wan2.2/predicotr-rl/{train_iql_hard_budget,hard_budget,hard_budget_data,policy}.py`。
- 本机缓存：`work/Wan2.2/adaptive_seacache_wan22/{rl_cache,rl_rollout_state}.py`、`wan/{timestep_cache,text2video}.py`、`wan/modules/model.py`。
- 最近本机正式离线流程：`work/data-collection-random/experiments/latent_feature_groups_v1/{train,select_checkpoint,vbench}.py`，random2323、400epoch；同时参考已有 e385 Exact-K 与 `work/online-finetune/online_finetune/trainer.py`。
- 不把 CNN/MLP、latent 维数、池化方式、feature 历史、已确定的新输入量化设置列为错误；不要求加载旧架构权重。
- 可复跑脚本：[audit.py](../experiments/local_consistency_audit/audit.py)。最终证据：[validation.json](../experiment_results/ours22_local_consistency_audit_20260911_verified/validation.json)，包含源码 SHA 和历史数据来源。

## 已确认一致

| 项目 | 核对结果 |
|---|---|
| 模型生成协议 | A14B、832×480、45帧、16fps、batch1、50-step DPM++、shift12、low/high CFG=3/4、boundary=.875、seed42、BF16、offload=True、t5_cpu=False、单卡，无FSDP/SP或prompt extension |
| 预算 | K为全视频skip步数，0–47；0/32/49强制重算；used跨high/low连续；预算耗尽强制重算、剩余eligible等于剩余K时强制skip |
| 动作与缓存 | 一个step共享cond/uncond动作，两分支及两专家残差隔离；缓存的是transformer blocks残差，reuse仍执行head/unpatchify；recompute后更新残差 |
| SEA状态 | 首block modulated norm输入、实际scheduler sigma、mean归一化SEA、相邻relative L1；当前距离先计入accumulator，实际重算后清零；边界距离置零 |
| 训练transition | 50步，真实动作和作为K；仅最后一步给绝对PSNR，gamma=1；所有transition进入Q/V，native/预算强制动作actor_mask=0 |
| IQL更新 | V使用target min-Q expectile；Q使用更新后的V bootstrap；Q更新后EMA；actor使用当前min-Q−V，仅在actor有效样本上拟合advantage均值/方差，exp(beta·标准化advantage)封顶20 |
| 超参数 | 400epoch、batch256、AdamW lr=1e-4、wd=.01、tau=.6、beta=1、gamma=1、target rho=.999、grad clip=1、reward scale=1、seed42 |
| 标准化原则 | 只在train拟合逐坐标population mean/std，std floor=1e-6，推理复用checkpoint统计；新CNN量化属于已确定feature合同 |
| 计量与新质量入口 | 完整generate wall，T5/DiT/VAE分项及实际full/reuse路径TFLOPs；VideoMetrics RGB PSNR/SSIM/LPIPS；custom VBench均调用同一10维入口 |

AST逐函数比较：`_move_batch`、`_actor_terms`、`_run_epoch`、`normalize_forced_steps`、`max_skip_budget`、`target_skip_budget_from_calibration`、`required_hard_budget_action`共7项完全相同。不是仅凭README判断。

SEA滤波对照：FP32/FP16/BF16各50步，使用相同小张量和sigma，150组输出逐值一致，最大绝对误差0；相邻relative L1也相同。此测试验证滤波实现，不等于完整14B数值等价。

## 差异与缺口

### 1. target speedup→K：选择公式一致，标定流程未迁移

本机 `hard_budget_data.py:78` 用train轨迹的每K速度中位数做按样本数加权的isotonic拟合，`hard_budget_data.py:401` 将标定表、fit_split及速度来源写入dataset manifest，checkpoint携带该表，推理从checkpoint读取。

新版 `policy.py:11` 仅接受外部 `ours4wan22_speed_to_k_v1` JSON，复用相同的“最近speedup，平局选较小K”函数；`training.py:89` 不生成、保存或绑定标定表。虽然检查protocol和forced_steps，但不核验fit_split、源轨迹或与policy的绑定。同一policy替换外部表即可改变目标到K的映射。

因此直接指定同一K时预算逻辑一致；仅指定相同target speedup不保证得到同一K。当前也没有CLI从训练数据生成所要求的新版calibration文件。

### 2. reward名称都是absolute PSNR，历史数值口径却不同

新版 `training.py:25`/README要求同protocol no-cache baseline的VideoMetrics **RGB** PSNR。本机random2323从active数据继承旧 `mean_psnr`，`hard_budget_data.py:284`直接作为终局reward，不自动重算。

已核实一条源样本 `openvidhd_balanced_1000_0002__rt1520`：训练表值 `21.171111111111127` 与原始PSNR JSON完全相同，原始method明确为 `ffmpeg_psnr_filter_psnr_avg_yuv_weighted`。其源JSON及SHA已记录在validation中。另有旧runner `run_batch.py:127` 调用 `compute_psnr.py` 未指定protocol，默认仍为legacy YUV；最近的latent-group评测另外运行VideoMetrics RGB，因此两种读数可能同时存在。

RGB符合当前用户规范，不应为追求旧实验一致而改回YUV。但迁移旧数据时不能直接把旧mean_psnr改名为terminal_psnr并宣称RGB；需要对应视频的RGB重测或经验证的RGB标签。当前新版dataset只核对有限非负数与合同，无法自行证明标签实际来自RGB。

### 3. speedup：新版完整generate，与旧标定的compute-only不同

本机 `wan/text2video.py` 的历史compute_elapsed将T5/逐步DiT/VAE计算计时相加，权重搬运在这些窗口外；本机hard-budget标定继承该数据列。抽样源表还明确存在历史GPU计时归一化修正。

新版 `generation.py`/`metrics.py` 使用共享profiler的完整generate wall，含offload搬运、feature/actor/scheduler；每条件一次完整warmup、同进程处理所有jobs。最近本机latent-group流程已补充同类组件及完整generate计时，但旧训练标定仍未随之重拟合。

新计量符合当前规范。不能把旧compute-only标定表直接换schema用于新版完整generate目标，也不能将旧speedup与新版headline当作同口径。DiT/T5/VAE FLOPs主体的计算路径一致；新版另外报告actor Conv/Linear FLOPs，未计SEA/feature/scheduler FLOPs均有披露。

### 4. 训练外层流程尚不完整

本机 `latent_feature_groups_v1/train.py` 在400epoch结束后自动绘loss曲线并调用 `select_checkpoint.py`；后者在e300–400验证集状态上采用双侧actor agreement与归一化Q漂移，选择e301–399中的checkpoint。

新版 `training.py:104` 明确写 `checkpoint_selection=not_selected`，不执行上述选点，也没有resume CLI。仅离线tensor dataset→训练→每epoch checkpoint已实现；既有采集数据到新版dataset的完整构建入口也尚未交付。

本机在线流程具备探索采集、跨轮replay与IQL续训。Ours4Wan22没有online入口。即使排除网络本身差异，checkpoint还存在接口差异：新字段为 `optimizers` 列表，现有online trainer读取 `optimizer_states` 字典及 `model_config`/`dataset_manifest`，不能直接连接。此项是流程适配缺口，不将旧权重不可加载当作架构错误。

### 5. 小的策略决策差异：概率恰好0.5时方向相反

本机 `policy.py:220` 使用 `p_skip >= .5`，平局选reuse；新版 `policy.py:80` 使用二分类logits.argmax，完全相等时选索引0=recompute。非平局且远离浮点阈值时两者等价；极接近阈值也可能受softmax舍入影响。没有证据说明该差异影响过现有真实轨迹，属于严格一致性的小项。

### 6. VBench200实际prompt缺少校验（应修复的独立问题）

`evaluation.py:19`只检查baseline与candidate的jobs彼此一致，未与固定 `Vbench200/prompts.jsonl` 对照。`evaluation.py:44`虽然保存实际prompt_map，但标准模式在第59行调用 `run_vbench200.sh` 时不传它。

下游 `VbenchEvaluation/prepare_videos.py:110` 仅根据固定sample_id查找视频，并用固定 `prompt_en` 命名评测文件。结果是：只要两边都用相同错误prompt生成、文件名却保留标准200 IDs，就可以通过当前配对/覆盖检查，随后按另一个prompt评语义质量。正常使用正确prompt不会触发，但README中“核验完整固定200集”的保证尚不完整。

建议在任何昂贵评测前核对完整ID集合及 `jobs[id].prompt == canonical[id].prompt_en`；这不涉及方法架构或feature变更。本次仅记录，未修改实现或运行实际VBench。

## 验证范围

- conda `wan2.2`，显式四项BLAS线程=1、`CUDA_VISIBLE_DEVICES=''`。
- 原有16项CPU测试全部通过，含全部K×两种极端actor、CFG约束、真实小模块prepared forward K0等价、checkpoint拒绝及真实IQL更新。
- 本次7函数AST一致、150组SEA数值对照及历史PSNR来源核验通过。
- 没有加载14B权重、启动GPU生成/训练/质量重测，也没有改变现有实验或方法代码。CPU通过不能替代真实14B同K、同动作序列的完整生成对照。
