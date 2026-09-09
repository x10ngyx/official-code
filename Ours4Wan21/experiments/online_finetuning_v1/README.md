# Ours4Wan21 在线微调：远端操作手册

本入口从已有 **Ours4Wan21离线checkpoint及其原训练cache** 继续在线IQL。沿用e385的“当前策略采样→累计单池replay→critic预热→joint更新→actor一致率选点→定期评测”框架，全部实现位于Ours4Wan21内，不依赖本机 `work/online-finetune` 或Wan2.2训练项目。

支持scalar5、SEA7及已注册的十个SEA7+latent模式。模式从checkpoint读取，在线逐步特征直接复用该模式的因果提取器；不为未选模式读取raw latent，不需要重新构建离线特征cache。

## 固定协议与实现范围

| 项目 | 正式默认值 |
|---|---|
| 在线轮数 | 8轮 |
| 每轮采集 | 从冻结3000个合格prompt有放回抽100条；每条50步 |
| 目标speedup | `[1.3,3.5]`分100层，每层均匀抽一个目标；用Wan21实测标定映射到整数K |
| 行为 | 直接按当前actor分布采样；native/Exact-K强制动作绕过采样 |
| Replay | 原offline **train** + 所有已完成online轮次，单池transition均匀有放回；没有固定50/50比例 |
| 预热 | 5个critic-only replay epochs，更新V、Q1/Q2及target，actor冻结 |
| Joint | 每轮20个joint replay epochs；每epoch `ceil(replay_rows/256)` 次更新 |
| 选点 | joint e11–e20；预热不计入joint编号；每epoch保存，完整计算到e20后选点 |
| Actor一致率 | 固定actor自由状态上，相邻epoch真实argmax的加权一致率；最大化最近两对的一致率最小值 |
| 平局 | 最近两对概率漂移的最大值更小优先，再平局选更晚epoch；无新增阈值 |
| 学习率 | Actor `4e-5`，Q/V `1e-4`，AdamW，无scheduler |
| IQL | τ=.7，β=1，cap20，γ1，target rho=.999，clip norm1 |
| AdamW | weight decay .01，betas(.9,.999)，eps1e-8 |
| 归一化 | 原checkpoint的train-only mean/std保持冻结 |
| 恢复 | R1显式新建optimizer；R2+继承**入选epoch**的optimizer及RNG |
| 评测 | VBench20，每4轮一次，即R4/R8；各轮五档1.5/2/2.5/3/3.5× |
| 评测prompt | 20个prompt由远端确定，prepare时冻结ID、文本、顺序与SHA |
| 生成 | Wan2.1-T2V-1.3B，832×480、81帧、16fps，50-step UniPC、shift5、CFG5、seed42、BF16 |
| 驻留 | 单卡48GB级GPU，所有模型组件常驻GPU；无CPU/model offload，无FSDP/SP或prompt改写 |
| 强制步 | 0/49；Wan21没有32步专家切换边界 |

20个joint epochs、e10之后按actor一致率选点、每4轮VBench20及远端指定prompt来自用户明确配置；5epoch预热、τ=.7、β1/cap20采用随后“按讨论配置实现”任务中的讨论默认值。完整导出见 `configs/online.json`，与 `online.py show-config` 一致。JSON是参考导出，不是可编辑的CLI覆盖文件；运行前修改算法需更新源码、导出和测试，并使用新run，不能改变已冻结run。

固定状态在prepare时从本Wan21 offline train中的actor自由行按step/K分层抽取，最多4096行，按原分层人口加权；同一run所有轮次固定，不使用validation/test，不搬用Wan22的1933个状态。e11主分数比较e9→e10与e10→e11；e9/e10只作历史，不能被选中。

## 1. 准备环境与输入路径

传输整个 `work/offical-code/`，包括Ours4Wan21、SeaCache4Wan21、Wan21Benchmark、ComponentMetrics、VideoMetrics、VbenchEvaluation及其引用的Vbench200 metadata、CalflopsEvaluation和源码lock。不能只传Ours4Wan21。沿用[主README](../../README.md)的安装/模型准备，环境统一为 **wan2.2**。

需要：已完成的random3000训练cache、该cache训练的离线checkpoint、Wan21源码lock兼容版本、模型权重、81帧分组件FLOPs profile、VideoMetrics/VBench依赖和本地权重、FFmpeg/FFprobe。VBench20使用任意远端prompt，因此走已有 `VbenchEvaluation/run_custom_vbench.sh`，不是要求200条视频的标准入口。

```bash
export OURS4WAN21_WORKSPACE=/srv/ours21
export OFFICIAL_CODE="$OURS4WAN21_WORKSPACE/work/offical-code"
export OURS_PROJECT="$OFFICIAL_CODE/Ours4Wan21"
export EXP_BASE=/data/exp
export OURS4WAN21_EXP_BASE="$EXP_BASE"
export WAN21_ROOT="$OURS4WAN21_WORKSPACE/work/Wan2.1-locked-65386b2"
export CHECKPOINT_DIR="$OURS4WAN21_WORKSPACE/models/Wan2.1-T2V-1.3B"
export TORCH_HOME="$OURS4WAN21_WORKSPACE/models/torch-cache"
export VBENCH_CACHE_DIR="$OURS4WAN21_WORKSPACE/models/VBench"
export HF_HOME="$VBENCH_CACHE_DIR/huggingface"
export XDG_CACHE_HOME="$VBENCH_CACHE_DIR/xdg"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# 按实际离线训练产物填写；支持训练结果中的最终选点symlink。
export OFFLINE_CACHE="$EXP_BASE/ours21_random3000_sea7_cache"
export START_CHECKPOINT="$EXP_BASE/ours21_random3000_sea7_analysis/selected_model.pt"
export FLOPS_PROFILE="$EXP_BASE/wan21_component_profile/profile.json"
# 完整源prompt注册表，例如已有OpenVid balanced-5000；必须覆盖offline所有split的ID及原文本。
export PROMPT_REGISTRY=/data/prompts/openvid_registry.jsonl
# 在远端人工/既定规则选出20条并保存，本地代码不代选。
export EVAL_PROMPTS=/data/prompts/vbench20.jsonl
export SUITE_NAME=ours21_sea7_online_v1
export RUN_DIR="$EXP_BASE/$SUITE_NAME"
export ONLINE_WEIGHTS="$OURS4WAN21_WORKSPACE/models/$SUITE_NAME"
cd "$OURS_PROJECT"

conda run -n wan2.2 python online.py show-config
bash experiments/online_finetuning_v1/validate.sh
```

远端 `/data/exp` 是示例挂载点；本机默认结果根为 `/all/yiran07-disk3/huteng_data/exp`。模型和训练checkpoint始终位于 `$OURS4WAN21_WORKSPACE/models`。每个新结果目录自动在本项目 `experiment_results/` 建立symlink，文件夹内含README。

VBench模型与LPIPS/AlexNet权重应按同级VideoMetrics/VbenchEvaluation文档提前放到上述models目录。prepare检查依赖可导入、FFmpeg存在、源码/模型路径和profile合同；它不加载全部VBench权重，也不验证GPU运行环境，不能替代首次远端真实运行验证。

## 2. 冻结20条评测prompt与3000条训练prompt池

注册表、训练池和评测文件均为JSONL，推荐每行：

```json
{"sample_id":"remote_0001","prompt":"A person walking along a quiet beach."}
```

兼容 `id/text` 与 `sample_id/prompt_en`。ID只能用安全的ASCII字母、数字、下划线、点和短横线，最长120字符；ID和规范化文本均须唯一。VBench20必须恰好20条，与offline所有split及online训练池隔离。文本检查可识别“换ID但文本相同”，但不会判断语义近似。

```bash
bash experiments/online_finetuning_v1/run.sh build-pool \
  --dataset "$OFFLINE_CACHE" \
  --prompt-registry "$PROMPT_REGISTRY" \
  --eval-prompts "$EVAL_PROMPTS" \
  --output-dir "$EXP_BASE/${SUITE_NAME}_prompt_pool"
export TRAIN_PROMPTS="$EXP_BASE/${SUITE_NAME}_prompt_pool/train_prompts.jsonl"
```

build-pool剔除offline validation/test及VBench20，随后以固定seed从合格注册表抽3000个prompt。合格数量不足3000时明确失败，应提供更完整的源注册表；不会把held-out补回池中。已有合规3000-prompt文件可直接指定 `TRAIN_PROMPTS` 跳过构建。

## 3. 使用实测Wan21 speedup→K标定

有匹配当前协议/硬件、覆盖完整 `[1.3,3.5]` 的标定文件时直接设置：

```bash
export CALIBRATION="$EXP_BASE/wan21_measured_k_calibration/calibration.json"
```

本仓库 `configs/speed_to_k.pending.json` 不可用于正式采集；不能把Wan22映射或理论 `1/(1-K/50)` 当作实测标定。

没有标定时，先从训练池另选一个小的固定prompt集合（不使用VBench20），以现有离线策略在同一空闲GPU上生成baseline和多个明确K的候选。以下示例需先设置真实的 `CALIBRATION_PROMPTS`：

```bash
export CALIBRATION_PROMPTS=/data/prompts/wan21_calibration.jsonl
export CUDA_VISIBLE_DEVICES=0
export CAL_PREFIX="$EXP_BASE/${SUITE_NAME}_calibration"
conda run --no-capture-output -n wan2.2 python generate.py \
  --wan21-root "$WAN21_ROOT" --checkpoint-dir "$CHECKPOINT_DIR" \
  --baseline --prompts "$CALIBRATION_PROMPTS" \
  --flops-profile "$FLOPS_PROFILE" --output-dir "${CAL_PREFIX}_baseline"

calibration_candidates=()
for k in 0 8 16 24 32 40 44 46 48; do
  candidate_dir="${CAL_PREFIX}_K${k}"
  conda run --no-capture-output -n wan2.2 python generate.py \
    --wan21-root "$WAN21_ROOT" --checkpoint-dir "$CHECKPOINT_DIR" \
    --policy-checkpoint "$START_CHECKPOINT" --skip-budget "$k" \
    --prompts "$CALIBRATION_PROMPTS" --flops-profile "$FLOPS_PROFILE" \
    --output-dir "$candidate_dir"
  calibration_candidates+=("$candidate_dir")
done
bash experiments/online_finetuning_v1/run.sh build-calibration \
  --baseline-dir "${CAL_PREFIX}_baseline" \
  --candidate-dirs "${calibration_candidates[@]}" \
  --output-dir "${CAL_PREFIX}_mapping"
export CALIBRATION="${CAL_PREFIX}_mapping/calibration.json"
```

导出器检查相同prompt、协议、GPU UUID，按完整generate时间之和的比值建立离散映射，目标取最近实测K，平局取较小K。实测不单调时拒绝导出，不静默平滑；请检查测量或扩大固定prompt集合重测。prepare要求映射覆盖1.3–3.5；目标speedup是请求值，最终报告仍用实际完整generate时间，不假设不同策略/机器必然命中目标。生成配置、checkpoint和权重都应在同一远端机器核验。

## 4. CPU预检并冻结正式run

```bash
bash experiments/online_finetuning_v1/run.sh prepare \
  --dataset "$OFFLINE_CACHE" --start-checkpoint "$START_CHECKPOINT" \
  --train-prompts "$TRAIN_PROMPTS" --eval-prompts "$EVAL_PROMPTS" \
  --prompt-registry "$PROMPT_REGISTRY" \
  --calibration "$CALIBRATION" --flops-profile "$FLOPS_PROFILE" \
  --wan21-root "$WAN21_ROOT" --checkpoint-dir "$CHECKPOINT_DIR" \
  --gpus 0,1,2,3 --output-dir "$RUN_DIR" --weights-dir "$ONLINE_WEIGHTS"
```

prepare不生成视频。它检查checkpoint与cache的manifest完全一致、只使用原train split，冻结全部输入SHA、代码SHA、20条评测prompt和训练固定状态。新run及models目录必须使用新名字。准备完成后同样参数可重复执行；若准备阶段被中断且没有完整inputs标记，使用新的run/weights名称重新prepare。

`--gpus`可填物理索引或GPU UUID。各prompt按ID稳定分配GPU，所以重复抽到的同一prompt、它的baseline及后续评测候选始终在同一GPU上；不照搬Wan22的22/26/26/26计数。各GPU独立单卡运行，不是FSDP。加载与原生预热串行，正式生成可并行；生成worker要求至少44GiB总显存、40GiB空闲显存。恢复时保持映射与设备不变，否则同GPU配对检查会失败。

## 5. 启动与恢复

在前台运行：

```bash
bash experiments/online_finetuning_v1/run.sh run --run-dir "$RUN_DIR"
```

需要后台运行时先开交互tmux，在tmux内重新设置第1节环境变量，运行同一命令，再按 `Ctrl-b d` 脱离：

```bash
tmux new -s ours21_online
```

主要阶段：每轮100条新轨迹→PSNR/SSIM/LPIPS→冻结5000个transition→累积replay→5预热epoch→20joint epoch→e11–20选点。R4/R8额外执行VBench20。baseline首次按需生成并缓存；VBench20离线起点五档只生成一次，后续复用。

查看状态与日志：

```bash
cat "$RUN_DIR/STATUS.json"
tail -n 30 "$RUN_DIR/logs/train_r001.log"
```

中断或失败后，修复环境问题，再运行**同一个run命令**。不要删除完成标记或手动覆盖数据：

- 每个视频的 `COMPLETE.json` 保存身份和文件SHA；已完成视频复用，损坏的已完成产物会拒绝继续。
- 未完成视频/指标目录移动到相邻 `incomplete/` 后重做，保留排查证据。
- 训练从最近完整epoch恢复；中断epoch从上一epoch重做，恢复torch/采样器RNG与optimizer。
- 每轮必须完整生成100条、构建5000行后才进入训练；失败不会跳到下一轮。
- 后续轮继承入选epoch，而非已计算的e20尾部；R1显式不继承离线optimizer moments。
- 一个run同时只允许一个编排进程。`LAST_ERROR.json`保留历史失败信息，当前状态以 `STATUS.json` 为准。
- 输入、配置、代码变化会阻止恢复；更改实验协议请建立新run。

## 6. 输出与指标口径

```text
RUN_DIR/
  manifest.json, STATUS.json, RESULT.json
  inputs/                    # frozen pool, VBench20 list, fixed actor states
  artifacts/baselines/        # per-prompt immutable native videos/time
  rounds/round_001/
    plan.json                # prompt, target, integer K, sampling seed
    collection/              # per-trajectory video/trace/component timing
    quality/                 # VideoMetrics: PSNR/SSIM/LPIPS, all81frames
    replay/transitions.pt    # raw pre-action states, masks, terminal PSNR
    training/                # epoch metrics, latest/selected checkpoint SHA
  rounds/round_004/evaluation/
    metrics.json, READOUT.md  # 20 prompts × 5 targets and offline comparison
  evaluation_reference/      # reused baseline VBench20 and offline five targets
  model_weights -> models/SUITE_NAME/
  commands/, logs/
```

模型目录每轮包含 `warmup_01..05.pt`、`epoch_01..20.pt` 和 `selected.pt`。checkpoint继续兼容现有 `generate.py --policy-checkpoint`；在线参数与恢复信息在checkpoint的 `online` 字段，原 `train_config` 和 `dataset_manifest` 保留离线来源；`offline_epoch`记录起点离线epoch，顶层`epoch`为当前joint epoch（预热为0）。

完整推理时间为 `pipeline.generate` wall time，排除模型加载、独立预热、MP4/trace写盘；记录T5、DiT、VAE各组件时间与TFLOPs，最终表报告DiT TFLOPs。predictor MLP单独按实际调用计时/计FLOPs，其耗时已包含在generate/DiT内，不能重复相加；特征FFT/归约等不计入predictor MLP FLOPs。

VBench20的20表示prompt数量。沿用任意prompt的 **custom-input十维** 协议，`vbench_score`为十个raw dimension score的等权均值，不是官方16维完整VBench总分。每档20视频，离线/在线分别计算，基线20条的VBench只算一次。PSNR/SSIM/LPIPS来自同级VideoMetrics的完整81帧RGB评测。

`RESULT.json`写出8轮完成及最终checkpoint；`rounds/round_004/evaluation/READOUT.md` 与R8同路径提供可读表格，详细分组件读数和逐视频证据在相邻JSON/CSV。

## 7. 验证范围

本机CPU测试覆盖：策略采样RNG、强制行mask、latent状态保存、单池抽样、e11–20选择、预热冻结actor、R1optimizer重置、R2继承、模拟中断后逐位一致、完成文件防篡改、R4/R8评测时机及20×5矩阵。

本轮没有启动真实Wan21视频生成、GPU在线训练或VBench权重推理；远端实际数据、显存与模型依赖须在上述环境准备后验证。不要将CPU测试结果写成正式在线质量结果。

本目录：`run.sh`为固定wan2.2/单线程启动器，`validate.sh`为CPU测试入口，README为远端手册。
