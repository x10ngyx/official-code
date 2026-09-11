# Ours4Wan21 在线微调：远端操作手册

本入口从已有 **Ours4Wan21离线checkpoint及其原训练cache** 继续在线IQL。沿用e385的“当前策略采样→累计单池replay→critic预热→joint更新→actor一致率选点→定期评测”框架，全部实现位于Ours4Wan21内，不依赖本机 `work/online-finetune` 或Wan2.2训练项目。

支持scalar5、SEA7及已注册的十个SEA7+latent模式。模式从checkpoint读取，在线逐步特征直接复用该模式的因果提取器；不为未选模式读取raw latent，不需要重新构建离线特征cache。

## 固定协议与实现范围

当前Dynamics128 e391四卡8轮任务使用 `--iql-profile aggressive_a2_a3_v1`：τ=.85、β=2.5、cap75（原A2/A3算术中点），入口为 `../dynamics128_online_e391_8rounds_a25_v1/`。表中的τ=.7、β=1、cap20仍是通用默认值；当前run明确覆盖这三项，其余算法不变。

| 项目 | 正式默认值 |
|---|---|
| 在线轮数 | 默认8轮；prepare --rounds冻结本次轮数；Dynamics128 e391当前按用户指定为8轮，四卡已启动 |
| 每轮采集 | 本次从离线train的800个prompt有放回抽100条，复用原baseline；每条50步 |
| 目标speedup | `[1.5,3.5]`分100层，每层均匀抽一个目标；用Wan21实测标定映射到整数K |
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
| 评测 | 每4轮及末轮；当前8轮为R4/R8；三档1.8/2.4/3.0×，沿用原VBench50的K23/K29/K35；仅PSNR/SSIM/LPIPS和组件计量，关闭VBench评分 |
| 评测prompt | 原Dynamics128随机VBench50中以seed42无放回选20条，冻结ID/文本/顺序/SHA，复用原同卡native baseline |
| 生成 | Wan2.1-T2V-1.3B，832×480、81帧、16fps，50-step UniPC、shift5、CFG5、seed42、BF16 |
| 驻留 | 单卡48GB级GPU，所有模型组件常驻GPU；无CPU/model offload，无FSDP/SP或prompt改写 |
| 强制步 | 0/49；Wan21没有32步专家切换边界 |

用户最新配置为训练目标1.5–3.5×、每4轮三档20prompt评测、复用原VBench50 baseline且不计算VBench；20个joint epochs和e10之后按actor一致率选点保持原配置；5epoch预热、τ=.7、β1/cap20采用随后“按讨论配置实现”任务中的讨论默认值。完整导出见 `configs/online.json`，与 `online.py show-config` 一致。JSON是参考导出，不是可编辑的CLI覆盖文件；运行前修改算法需更新源码、导出和测试，并使用新run，不能改变已冻结run。

固定状态在prepare时从本Wan21 offline train中的actor自由行按step/K分层抽取，最多4096行，按原分层人口加权；同一run所有轮次固定，不使用validation/test，不搬用Wan22的1933个状态。e11主分数比较e9→e10与e10→e11；e9/e10只作历史，不能被选中。

## 1. 准备环境与输入路径

传输整个 `work/official-code/`，包括Ours4Wan21、SeaCache4Wan21、Wan21Benchmark、ComponentMetrics、VideoMetrics、CalflopsEvaluation和源码lock。不能只传Ours4Wan21。沿用[主README](../../README.md)的安装/模型准备，环境统一为 **wan2.2**。

需要：已完成的random3000训练cache、该cache训练的离线checkpoint、Wan21源码lock兼容版本、模型权重、81帧分组件FLOPs profile、VideoMetrics依赖和本地LPIPS权重、FFmpeg/FFprobe。评测不导入或调用VBench，prepare也不检查VBench依赖。VBench50仅表示评测prompt和baseline的来源。

```bash
export OURS4WAN21_WORKSPACE=/srv/ours21
export OFFICIAL_CODE="$OURS4WAN21_WORKSPACE/work/official-code"
export OURS_PROJECT="$OFFICIAL_CODE/Ours4Wan21"
export EXP_BASE=/data/exp
export OURS4WAN21_EXP_BASE="$EXP_BASE"
export WAN21_ROOT="$OURS4WAN21_WORKSPACE/work/Wan2.1-locked-65386b2"
export CHECKPOINT_DIR="$OURS4WAN21_WORKSPACE/models/Wan2.1-T2V-1.3B"
export TORCH_HOME="$OURS4WAN21_WORKSPACE/models/torch-cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# 当前机器也可直接指定已展开的 wan2.2 环境解释器；run.sh/validate.sh 会优先使用它。
# export WAN22_PYTHON=/absolute/path/to/wan2.2/bin/python

# 按实际离线训练产物填写；支持训练结果中的最终选点symlink。
export OFFLINE_CACHE="$EXP_BASE/ours21_random3000_sea7_cache"
export START_CHECKPOINT="$EXP_BASE/ours21_random3000_sea7_analysis/selected_model.pt"
export FLOPS_PROFILE="$EXP_BASE/wan21_component_profile/profile.json"
# 完整源prompt注册表，例如已有OpenVid balanced-5000；必须覆盖offline所有split的ID及原文本。
export PROMPT_REGISTRY=/data/prompts/openvid_registry.jsonl
# 原50条归档必须可读；build-evaluation只选prompt和建立baseline引用，不生成视频。
export VBENCH50_RUN="$EXP_BASE/ours21_dynamics128_e391_vbench50_random42_4gpu_v1"
export EVALUATION_BUNDLE="$EXP_BASE/ours21_online_eval20_from_vbench50_random42_v1"
export EVAL_PROMPTS="$EVALUATION_BUNDLE/eval_prompts.jsonl"
export SUITE_NAME=ours21_sea7_online_v1
export RUN_DIR="$EXP_BASE/$SUITE_NAME"
export ONLINE_WEIGHTS="$OURS4WAN21_WORKSPACE/models/$SUITE_NAME"
cd "$OURS_PROJECT"

bash experiments/online_finetuning_v1/run.sh show-config
bash experiments/online_finetuning_v1/validate.sh
```

远端 `/data/exp` 是示例挂载点；本机默认结果根为 `/mnt/hdd/xiongyuxiang/tmp/exp`。模型和训练checkpoint始终位于 `$OURS4WAN21_WORKSPACE/models`。每个新结果目录自动在本项目 `experiment_results/` 建立symlink，文件夹内含README。

LPIPS/AlexNet权重按同级VideoMetrics文档放到models目录。prepare检查依赖、FFmpeg、源码/模型/profile、复用baseline来源SHA及物理GPU匹配；以nvidia-smi解析GPU索引，不初始化CUDA或生成视频。实际GPU推理仍需正式运行验证。显式WAN22_PYTHON可指向已解包的wan2.2解释器，不要求其目录名恰好为wan2.2。

## 2. 从原VBench50冻结20条评测prompt，并复用baseline

本机已准备参考包`/mnt/hdd/xiongyuxiang/tmp/exp/ours21_online_eval20_from_vbench50_random42_v1/`，通过`experiment_results/`同名链接访问；`eval_prompts.jsonl`为固定清单，`VALIDATION.json`记录61项CPU测试及20条原baseline验证。

```bash
bash experiments/online_finetuning_v1/run.sh build-evaluation \
  --source-run "$VBENCH50_RUN" --output-dir "$EVALUATION_BUNDLE"
```

固定算法为：按sample_id排序原50条，用`random.Random(42).sample(rows,20)`无放回抽取，再按ID排序。抽样不读取质量指标、不反复重抽。新目录保存完整prompt、原视频/计时/trace的symlink、原组件测量和来源SHA，不复制视频。原50条归档保留。

原baseline每条绑定自己的物理GPU UUID；在线和离线起点评测候选都路由到该GPU，即使`--gpus`参数顺序改变也不能换卡。prepare要求配置包含全部引用GPU，并匹配Wan模型路径和FLOPs profile。若原baseline或来源被修改，恢复会报错，不自动重算或替换。

### 冻结3000条训练prompt池

注册表、训练池和评测文件均为JSONL，推荐每行：

```json
{"sample_id":"remote_0001","prompt":"A person walking along a quiet beach."}
```

兼容 `id/text` 与 `sample_id/prompt_en`。ID只能用安全的ASCII字母、数字、下划线、点和短横线，最长120字符；ID和规范化文本均须唯一。评测清单必须与上述bundle恰好一致，共20条，与offline所有split及online训练池隔离。文本检查可识别“换ID但文本相同”，但不会判断语义近似。

```bash
bash experiments/online_finetuning_v1/run.sh build-pool \
  --dataset "$OFFLINE_CACHE" \
  --prompt-registry "$PROMPT_REGISTRY" \
  --eval-prompts "$EVAL_PROMPTS" \
  --output-dir "$EXP_BASE/${SUITE_NAME}_prompt_pool"
export TRAIN_PROMPTS="$EXP_BASE/${SUITE_NAME}_prompt_pool/train_prompts.jsonl"
```

build-pool剔除offline validation/test及20条评测prompt，随后以固定seed从合格注册表抽3000个prompt。合格数量不足3000时明确失败，应提供更完整的源注册表；不会把held-out补回池中。已有合规3000-prompt文件可直接指定 `TRAIN_PROMPTS` 跳过构建。

## 3. 使用实测Wan21 speedup→K标定

有匹配当前协议/硬件、覆盖完整 `[1.5,3.5]` 的标定文件时直接设置：

```bash
export CALIBRATION="$EXP_BASE/wan21_measured_k_calibration/calibration.json"
```

本仓库 `configs/speed_to_k.pending.json` 不可用于正式采集；不能把Wan22映射或理论 `1/(1-K/50)` 当作实测标定。

没有标定时，先从训练池另选一个小的固定prompt集合（不使用20条评测prompt），以现有离线策略在同一空闲GPU上生成baseline和多个明确K的候选。以下示例需先设置真实的 `CALIBRATION_PROMPTS`：

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

导出器检查相同prompt、协议、GPU UUID，按完整generate时间之和的比值建立离散映射，目标取最近实测K，平局取较小K。实测不单调时拒绝导出，不静默平滑；请检查测量或扩大固定prompt集合重测。prepare要求映射覆盖1.5–3.5；目标speedup是请求值，最终报告仍用实际完整generate时间，不假设不同策略/机器必然命中目标。生成配置、checkpoint和权重都应在同一远端机器核验。

## 4. CPU预检并冻结正式run

```bash
bash experiments/online_finetuning_v1/run.sh prepare \
  --dataset "$OFFLINE_CACHE" --start-checkpoint "$START_CHECKPOINT" \
  --train-prompts "$TRAIN_PROMPTS" --eval-prompts "$EVAL_PROMPTS" \
  --evaluation-bundle "$EVALUATION_BUNDLE" \
  --prompt-registry "$PROMPT_REGISTRY" \
  --calibration "$CALIBRATION" --flops-profile "$FLOPS_PROFILE" \
  --wan21-root "$WAN21_ROOT" --checkpoint-dir "$CHECKPOINT_DIR" \
  --gpus 0,1,2,3 --rounds 8 --output-dir "$RUN_DIR" --weights-dir "$ONLINE_WEIGHTS"
```

prepare不生成视频。它检查checkpoint与cache的manifest完全一致、只使用原train split，冻结全部输入SHA、代码SHA、20条评测prompt和训练固定状态。`--rounds`在prepare时冻结；其余算法配置保持默认。当前Dynamics128 e391任务使用`--rounds 8 --iql-profile aggressive_a2_a3_v1 --training-bundle <offline_train_reference>`，详见`../dynamics128_online_e391_8rounds_a25_v1/`；用户已授权四卡启动。新run及models目录必须使用新名字。准备完成后同样参数可重复执行；若准备阶段被中断且没有完整inputs标记，使用新的run/weights名称重新prepare。

`--gpus`可填物理索引或GPU UUID。使用`--training-bundle`时，prepare要求prompt池精确等于离线train集合，并按原baseline卡号冻结每个ID的GPU；旧模式按新baseline和候选工作量均衡分配；重复训练prompt与训练baseline同卡；20条评测prompt按原baseline UUID分配GPU；不照搬Wan22的22/26/26/26计数。各GPU独立单卡运行，不是FSDP。加载与原生预热串行，正式生成可并行；生成worker要求至少44GiB总显存、40GiB空闲显存。恢复时保持映射与设备不变，否则同GPU配对检查会失败。

## 5. 启动与恢复

在前台运行：

```bash
bash experiments/online_finetuning_v1/run.sh run --run-dir "$RUN_DIR"
```

需要后台运行时先开交互tmux，在tmux内重新设置第1节环境变量，运行同一命令，再按 `Ctrl-b d` 脱离：

```bash
tmux new -s ours21_online
```

主要阶段：每轮100条新轨迹→PSNR/SSIM/LPIPS→冻结5000个transition→累积replay→5预热epoch→20joint epoch→e11–20选点。每4轮及末轮额外执行20prompt三档评测，无VBench调用。本次通过`--training-bundle`复用离线train的800个baseline；未传此参数的旧模式才首次按需生成并缓存；20条评测native baseline全部复用bundle；当前Dynamics128 e391离线起点三档60候选和既有质量结果全部从原VBench50归档接入缓存，R4/R8均不重复评测离线起点。在线每个评测轮生成60候选。三档和离线/在线checkpoint合并为一次四卡生成调度，每卡保持模型常驻；VideoMetrics按四卡分片，汇合原始逐帧/逐视频CSV。

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
  inputs/                    # frozen pool, evaluation20 list, fixed actor states
  artifacts/baselines/        # per-prompt immutable native videos/time
  rounds/round_001/
    plan.json                # prompt, target, integer K, sampling seed
    collection/              # per-trajectory video/trace/component timing
    quality/                 # VideoMetrics: PSNR/SSIM/LPIPS, all81frames
    replay/transitions.pt    # raw pre-action states, masks, terminal PSNR
    training/                # epoch metrics, latest/selected checkpoint SHA
  rounds/round_004/evaluation/
    metrics.json, READOUT.md  # 20 prompts × 3 targets and offline comparison
  evaluation_reference/      # offline three-target candidates, reused at R4/R8
  model_weights -> models/SUITE_NAME/
  commands/, logs/
```

模型目录每轮包含 `warmup_01..05.pt`、`epoch_01..20.pt` 和 `selected.pt`。checkpoint继续兼容现有 `generate.py --policy-checkpoint`；在线参数与恢复信息在checkpoint的 `online` 字段，原 `train_config` 和 `dataset_manifest` 保留离线来源；`offline_epoch`记录起点离线epoch，顶层`epoch`为当前joint epoch（预热为0）。

完整推理时间为 `pipeline.generate` wall time，排除模型加载、独立预热、MP4/trace写盘；记录T5、DiT、VAE各组件时间与TFLOPs，最终表报告DiT TFLOPs。predictor MLP单独按实际调用计时/计FLOPs，其耗时已包含在generate/DiT内，不能重复相加；特征FFT/归约等不计入predictor MLP FLOPs。

20表示prompt数量，源自原随机VBench50；**不计算任何VBench分数**，结果标记`vbench_status=skipped_by_user`。三档名义目标为1.8/2.4/3.0×，评测固定原K23/K29/K35，不受在线训练speedup→K标定重新映射影响。每档20视频，离线起点与在线入选checkpoint均与同一原baseline配对。PSNR/SSIM/LPIPS来自同级VideoMetrics的完整81帧RGB评测；最终报告实际完整推理加速，不保证命中名义目标。

`RESULT.json`写出本次冻结轮数完成及最终checkpoint；`rounds/round_004/evaluation/READOUT.md` 与R8同路径提供可读表格，详细分组件读数和逐视频证据在相邻JSON/CSV。

## 7. 验证范围

本机CPU测试覆盖：策略采样RNG、强制行mask、latent状态保存、单池抽样、e11–20选择、预热冻结actor、R1optimizer重置、R2继承、模拟中断后逐位一致、完成文件防篡改、每4轮及末轮评测时机、20×3矩阵、原baseline软链接复用/来源损坏拒绝、GPU顺序变化时的同卡路由、无baseline生成和无VBench调用。

配置修改与参考bundle构建不启动真实Wan21视频生成或GPU在线训练；远端实际数据、显存与模型依赖须在上述环境准备后验证。不要将CPU测试结果写成正式在线质量结果。

本目录：`run.sh`为固定wan2.2/单线程启动器，`validate.sh`为CPU测试入口，README为远端手册。
