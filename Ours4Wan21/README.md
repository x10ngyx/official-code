# Ours4Wan21：远端端到端运行手册

本项目实现 Wan2.1-T2V-1.3B 的随机行为数据采集、训练 cache、离线 IQL、checkpoint 选择/诊断、在线微调，以及 Vbench200/在线VBench20推理和评测。推理基础方法直接使用同级 `SeaCache4Wan21` 的 forward、采样器、SEA 滤波和分支残差缓存。

**当前远端进度（用户已确认）：random管线已经生成3000条随机候选轨迹。当前就使用这3000条，不需要继续补到9000条，也不重新抽样。** 一条轨迹是一条完整的50步候选视频；实际prompt数量和train/val/test数量从原始manifest读取，不假定是3000个不同prompt。

**现在从第1节确认环境变量，再直接执行第3节冻结已有数据并整理cache，随后按第4–7节训练、选ckpt和测试。第2节仅保留从零造数据的历史入口，不是这批已有数据的前置步骤。** 本助手尚未读取远端产物，数据状态依据用户报告；cache入口仍会核验completion、真实PSNR/trace及原split，不把“已生成视频”自动当作所有训练材料齐全。

在线微调从已有离线checkpoint继续：**8轮、每轮100条、5个critic预热epoch＋20个joint epoch、e11–e20按actor一致率选点；每4轮VBench20，20个prompt由远端确定**。完整远端步骤见 [在线微调手册](experiments/online_finetuning_v1/README.md)，入口为 `online.py`。该流程与下方离线400epoch流程独立。

## 目录与交付内容

| 路径 | 作用 |
|---|---|
| `data_collection/` | 不可变采集manifest、原始视频/latent/trace/质量/计量、发布和审计 |
| `select_data.py` | 默认原样冻结已有3000条完整random轨迹，不抽样；保存身份/split/SHA |
| `prepare_features.py` | 一次遍历已有候选raw latent，提取全部10组因果特征，支持中断续提取 |
| `prepare_data.py` | 读取完成标记和trace，整理训练cache，不加载raw latent |
| `train.py` | 固定本机配置的400epoch离线IQL |
| `online.py` | 在线准备、采集、累计replay、续训、恢复与每4轮VBench20 |
| `analyze_training.py` | 全训练指标表/曲线，以及本机post300规则的checkpoint选择 |
| `generate.py` | 匹配baseline或指定策略/K的固定协议推理 |
| `evaluate.py` | 性能及predictor overhead汇总、PSNR/SSIM/LPIPS和Vbench200评测 |
| `ours4wan21/` | 共用状态、数据、网络、损失、控制器、计量和分析实现 |
| `configs/` | 固定训练参数参考和未标定的speed→K示例合同 |
| `local_training_lock.json` | 本机训练源码/抽取函数SHA；远端运行不需要Wan2.2训练项目 |
| `experiments/rl_{cache,training,analysis,inference,validation}_v1/` | 各阶段launcher/CPU验收 |
| `experiment_results/` | 外部结果目录的symlink；模型实际文件只放workspace的`models/` |
| `PROGRESS.md`, `logs/` | 当前状态与交接记录 |

**当前实验矩阵：scalar5、SEA7两个对照 + SEA7分别加10个候选特征，共12组。** 完整分阶段命令见 [12组远端入口](experiments/latent_groups_v1/README.md)，精确公式见 [CANDIDATES.md](experiments/latent_groups_v1/CANDIDATES.md)。先完成第3节selection，随后按12组入口一次提取特征，再整理各组cache、训练、选择ckpt及VBench。

## 1. 远端环境与目录

传输单位是整个 `work/offical-code/`，不能只拷贝Ours4Wan21。需要同级 `SeaCache4Wan21`、`Wan21Benchmark`、`ComponentMetrics`、`CalflopsEvaluation`、`VideoMetrics`、`Vbench200`、`VbenchEvaluation`，以及采集preflight检查的notice/lock依赖。不要对已有实验结果symlink使用`rsync -L`复制大文件。

示例传输（请替换登录与路径；这是操作说明，不会自动发送文件）：

```bash
rsync -a --exclude='experiment_results/*' --exclude='__pycache__/' \
  work/offical-code/ USER@REMOTE:/srv/ours21/work/offical-code/
```

另外准备：

- 全项目统一的conda环境 **`wan2.2`**，含可用PyTorch/CUDA、Wan依赖、NumPy、Matplotlib。`Wan2.1`仅指模型/源码版本，不能另用同名conda环境。
- Wan2.1源码固定commit `65386b2e03c490796eede31b0325a6a595cc684e`；入口会检查兼容文件SHA。
- `$OURS4WAN21_WORKSPACE/models/Wan2.1-T2V-1.3B` 权重，以及同一models根下的LPIPS/AlexNet和VBench依赖权重；不能依赖本机软链接目标在远端仍存在。
- 单卡推理要求48GB级GPU；采集launcher使用4张独立GPU，每个worker单卡，无FSDP/SP。
- 外部大容量结果盘、FFmpeg/FFprobe。先确认视频、逐步latent和模型checkpoint所需空间。

在远端同一个shell里设置一次，后续阶段继续使用：

```bash
export OURS4WAN21_WORKSPACE=/srv/ours21
export OFFICIAL_CODE="$OURS4WAN21_WORKSPACE/work/offical-code"
export OURS_PROJECT="$OFFICIAL_CODE/Ours4Wan21"
export EXP_BASE=/data/exp
export OURS4WAN21_EXP_BASE="$EXP_BASE"
export WAN21_ROOT=/srv/ours21/work/Wan2.1-locked-65386b2
export CHECKPOINT_DIR="$OURS4WAN21_WORKSPACE/models/Wan2.1-T2V-1.3B"
export METRICS_MODEL_CACHE="$OURS4WAN21_WORKSPACE/models/torch-cache"
export TORCH_HOME="$METRICS_MODEL_CACHE"
export VBENCH_CACHE_DIR="$OURS4WAN21_WORKSPACE/models/VBench"
export HF_HOME="$VBENCH_CACHE_DIR/huggingface"
export XDG_CACHE_HOME="$VBENCH_CACHE_DIR/xdg"
export VBENCH_PYTHON="$(conda run -n wan2.2 python -c 'import sys; print(sys.executable)')"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

conda run --no-capture-output -n wan2.2 python -m pip install \
  -r "$OFFICIAL_CODE/VideoMetrics/requirements.txt" \
  -r "$OFFICIAL_CODE/CalflopsEvaluation/requirements.txt"
cd "$OURS_PROJECT"
conda run -n wan2.2 python -c 'import torch, numpy, matplotlib, calflops; print(torch.__version__)'
conda run --no-capture-output -n wan2.2 python -m unittest discover -s tests -v
```

VBench依赖和本地权重准备详见同级 `VbenchEvaluation/README.md`；其评测默认本地加载，不应到正式评测时才发现缺权重。所有阶段仍使用wan2.2。

本机不设置环境变量时，结果根仍为 `/all/yiran07-disk3/huteng_data/exp`、模型根为本workspace的`models/`。远端必须在**启动Python之前**设置上述root变量；训练/推理/共享评测桥接都会使用它们。输出须是root下的新目录，不能覆盖旧结果。

## 2. 从零造数据的历史入口（当前已有3000条，跳过本节）

本节描述仓库原始3000 prompts×3轨迹的大规模采集计划，仅在另行要求从零重建时使用。不要为了整理当前已有3000条而重跑plan、baseline、candidate或要求全9000 archive的finalize。

```bash
export RUN_ID=wan21_random_collection_v1
export COLLECTION_ROOT="$EXP_BASE/$RUN_ID"
cd "$OURS_PROJECT/data_collection"
bash experiments/random_threshold_collection_v1/launch_4gpu.sh preflight
bash experiments/random_threshold_collection_v1/launch_4gpu.sh plan
bash experiments/random_threshold_collection_v1/launch_4gpu.sh profile
bash experiments/random_threshold_collection_v1/launch_4gpu.sh baselines
```

- `plan`：冻结3000 prompts、每prompt3候选、80/10/10 prompt split和随机路径，不执行GPU推理。
- `profile`：生成 `$COLLECTION_ROOT/calflops_profile.json`，含81帧DiT/T5/VAE计数。
- `baselines`：生成同prompt、seed42、固定协议的3000个无缓存baseline，保存视频、timing、50步latent。
- `RUN_ID`必须在各次命令之间保持相同；否则会进入不同archive。

**以下是从零新增随机候选的前置条件；整理已有3000条不需要重新标定。仓库随附配置仍为pending，这不代表远端当时使用的标定状态。** `data_collection/configs/speed_threshold_mapping.pending.json`为空，不能改一个status便假装标定完成，也不能直接套用Wan22参数。需先在独立标定prompt上按相同协议实测SeaCache阈值/完整generate时间，形成实际数据支持的映射。

标定文件必须满足：`schema=ours4wan21_speed_threshold_mapping_v1`、`calibration_status=calibrated`、固定Wan21 protocol、有效 `fit_source` 文件及其SHA256、正的 `threshold_bounds=[min,max]`、`mapping.kind=monotone_piecewise_linear`，以及严格递增的 `mapping.speedups` 和非递减的 `mapping.mean_thresholds`。实测节点必须覆盖 `[1.5,3.5]`，禁止外推。完整合同由 `data_collection/src/ours4wan21_data/manifest.py::_load_calibration`检查。如果实测覆盖不足，候选阶段保持阻塞；本文没有虚构测量节点或声称标定已完成。将标定原始记录及配置一起放在远端结果盘，避免`fit_source`仍指向本机路径。

取得有效标定后：

```bash
export CALIBRATION_CONFIG="$EXP_BASE/wan21_calibration/speed_threshold_mapping.calibrated.json"
bash experiments/random_threshold_collection_v1/launch_4gpu.sh materialize
bash experiments/random_threshold_collection_v1/launch_4gpu.sh candidates
bash experiments/random_threshold_collection_v1/launch_4gpu.sh finalize
```

`materialize`将9000条随机阈值路径写入 `manifests/random_runnable.jsonl`。`candidates`会保留实际执行动作、两个CFG分支各自的SEA d/D、视频、50步latent、latency/TFLOPs，以及相对baseline的RGB PSNR/SSIM/LPIPS。

`finalize`完成发布、采集archive的custom-input VBench和完整审计；查看 `audits/archive_audit.json`、`published/`及各candidate的 `CANDIDATE_COMPLETE.json`，所有缺失/失败应先在采集阶段解决。采集VBench采用支持任意视频的10维原始分数算术均值，这与后文固定Vbench200的16维加权分数是不同口径。

本训练计划不调用 `seacache_threshold_collection_v1`：其固定阈值数据属于另一采集支路，不混入3000条随机训练子集。细节见 `data_collection/README.md` 和 `data_collection/REMOTE_DEPLOYMENT.md`。

## 3. 原样冻结已经完成的3000条，再准备训练cache

先冻结一次selection，再为12组分别构建cache；下例为两个标量对照，特征组完整命令见12组入口。这里selection是**现有数据清单**，默认不做随机抽取。只需指出远端已经完成数据所在的archive：

```bash
cd "$OURS_PROJECT"
export COLLECTION_ROOT=/data/exp/EXISTING_RANDOM_3000_ARCHIVE
export SELECTION_DIR="$EXP_BASE/ours21_existing_random3000_selection_v1"
conda run --no-capture-output -n wan2.2 python select_data.py \
  --collection-root "$COLLECTION_ROOT" \
  --strategy all-completed --output-dir "$SELECTION_DIR"
```

`all-completed`是默认策略，也可以省略`--strategy`。入口读取原 `manifests/random_runnable.jsonl` 和现有 `shards/shard_*/candidates/*/CANDIDATE_COMPLETE.json`：

- 原manifest可以只规划3000条，也可以仍规划9000条但其中只完成3000条；不要求其余候选补齐，不要求每prompt有3个完成样本。
- 当前恰有3000个完成的random轨迹时，原样全部保留，仅按trajectory_id排序，不按质量、速度、prompt或完成先后再次选择。
- 少于或多于3000个完成项都明确报错，不静默截断、补抽；有损坏或身份不匹配的completion也报错。
- 只接受 `random_continuous_seacache_threshold`，不混入固定SeaCache anchors或baseline。
- 输出 `selection.json`，保存全部3000 IDs、原split、相对completion路径、源SHA、计划数及实际完成数。不同state mode共用这份清单。

以后若另有更大的完整采集，`uniform-trajectories`/`one-per-prompt`仍可显式调用；它们需要完整源采集并做seed42抽样，**不用于当前已有3000条的流程**。保留原prompt split，不重新按行划分；实际各split数量以输出为准。若已有数据缺少验证split或必要trace/PSNR，入口会指出缺项，不擅自移动测试样本或虚构标签。

```bash
export STATE_MODE=sea7  # 对照组；10个特征组使用sea7_<候选名>，见12组入口
export CACHE_DIR="$EXP_BASE/ours21_random3000_${STATE_MODE}_cache_v1"
bash experiments/rl_cache_v1/run.sh \
  --collection-root "$COLLECTION_ROOT" \
  --selection "$SELECTION_DIR/selection.json" \
  --state-mode "$STATE_MODE" --output-dir "$CACHE_DIR"
```

这个cache是供MLP/IQL读取的**小型标量transition cache**，不是重新保存一份视频/模型残差缓存。它从原采集completion指向的trace与VideoMetrics JSON读取，不重新推理、不解码视频、不打开raw latent、不复制大latent文件。原archive仍须保留，不能因当前不用latent就删除原始数据。

| 产物 | 内容 |
|---|---|
| `transitions.pt` | 3000×50=150000个state/next_state/action/reward/done/actor_mask；train/val/test索引及来源manifest |
| `manifest.json` | state字段顺序、协议、3000条来源SHA、selection及prompt split |
| `COMPLETE.json` | cache构建完成及transitions.pt SHA |

原 `val`映射到训练器的`evaluation`标签；原`test`单独保留在 `test_indices`，不进训练梯度、normalizer、逐epoch验证或checkpoint选择。模型梯度使用的是这3000条子集中原train部分，不把验证/测试样本一起喂给optimizer。

每条轨迹的K由真实reuse总数重标记。状态全部是动作前状态；next_state只在本轨迹内前移，终局自指且done=1；奖励只有step49的匹配RGB PSNR。cond/uncond动作或SEA信号不一致的轨迹会报错，不能悄悄当成一次完整step复用。程序会检查source身份/协议、SEA累积与重置、有限值、forced mask和split。

## 4. 状态模式与固定训练设置

| 模式 | 网络输入顺序 | 状态 |
|---|---|---|
| `scalar5` | cache_valid, step/49, K/50, used_skips/50, consecutive_skips/50 | 可训练/推理 |
| `sea7` | SEA相邻relative-L1、包含当前项的累计SEA距离，再接scalar5 | 可训练/推理 |
| `sea7_<候选名>` | 所选latent feature在前、SEA7在后，共10组 | 可提取cache/训练/推理，版本及公式随checkpoint保存 |
| `sea7_latent` | 未指定候选的旧占位名称 | 明确报错；必须使用具体组名 |

cache_valid只有step0为0；第0/49步强制recompute，SEA d/D用零sentinel；第32步是普通Wan21步。一个RL动作对应一个去噪step、同时应用于CFG两分支；残差和滤波历史仍是独立分支状态。训练和推理调用同一个state builder，actor仅在自由cond步查询一次。

参考本机 `random_only_state_revision_v1/train_pooled_control.py` 与 `Wan2.2/predicotr-rl/train_iql_hard_budget.py`，MLP/normalizer/IQL核逐段保留并锁定SHA。

| 参数 | 固定值 |
|---|---|
| epochs / batch / seed / workers | 400 / 256 / 42 / 0 |
| Actor、双Q、V | 独立3×256 MLP，LayerNorm+SiLU，dropout0 |
| optimizer | 3个AdamW；lr=1e-4，weight_decay=.01 |
| IQL | tau=.6，beta=1，gamma=1，target_rho=.999 |
| actor权重 | 自由状态batch内advantage标准化后exp，cap20 |
| gradient clip | V、joint Q1/Q2、actor分别clip norm1 |
| reward | 终局absolute RGB PSNR，scale1，无speed/latent-MSE/frontier项 |
| actor mask | 排除native/Exact-K强制动作；critic保留全部训练行 |
| normalizer | 仅原train状态拟合mean/std，std下限1e-6 |
| 验证/checkpoint | 每epoch完整验证并保存，另有best actor loss和final |

`configs/training.json`是参数参考，不是覆盖文件。正式入口不允许换epoch/网络/奖励，也拒绝不是3000条selection的dataset。只有CPU验收可使用标记为smoke的2epoch配置。

```bash
export TRAIN_DIR="$EXP_BASE/ours21_random3000_${STATE_MODE}_train_v1"
export POLICY_WEIGHTS="$OURS4WAN21_WORKSPACE/models/ours21_random3000_${STATE_MODE}_train_v1"
CUDA_VISIBLE_DEVICES=0 bash experiments/rl_training_v1/run.sh \
  --dataset "$CACHE_DIR" --state-mode "$STATE_MODE" \
  --output-dir "$TRAIN_DIR" --checkpoint-dir "$POLICY_WEIGHTS"
```

权重写到 `models/.../checkpoints/epoch_001.pt` 至 `epoch_400.pt`。每个checkpoint有actor、Q1/Q2、V、target-Q1/Q2、normalizer、optimizer/RNG状态、配置和数据manifest。训练结果有 `config.json`、`dataset_manifest.json`、`split.json`、`epoch_metrics.jsonl`、`model_weights` symlink，成功结束才有 `TRAINING_COMPLETE.json`。

`best_model.pt`是全400epoch的最低validation actor loss，`final_model.pt`是e400；二者不自动等于后面的正式post300选择。入口是fresh training，尚无resume CLI；中断不能拿半程目录当完整训练结果，重开必须使用新run。每个训练模式使用独立权重/结果目录。

## 5. 全训练指标分析与checkpoint选择

沿用本机SEA7后300轮的完整验证自由状态普查规则，而不是在VBench结果中挑checkpoint。

```bash
export ANALYSIS_DIR="$EXP_BASE/ours21_random3000_${STATE_MODE}_analysis_v1"
# 不占GPU可用--device cpu；较快的单卡普查用CUDA_VISIBLE_DEVICES及--device cuda。
CUDA_VISIBLE_DEVICES=0 bash experiments/rl_analysis_v1/run.sh \
  --training-result "$TRAIN_DIR" --dataset "$CACHE_DIR" \
  --checkpoint-dir "$POLICY_WEIGHTS" \
  --output-dir "$ANALYSIS_DIR" --device cuda
```

入口要求真实完整400epoch、正确cache及checkpoint归属，拒绝smoke结果。输出：

- `training_metrics.csv/png/svg`：全部400epoch train/val的Q loss、V loss、actor loss、advantage mean/std、actor weight mean/max、logged/greedy skip率、自由/强制行数、奖励；CSV另有累计训练时间。
- `post300_validation_values.npz`：e300…400在全部validation自由状态上的min(Q1,Q2)两动作值、actor动作和原始行索引。
- `post300_adjacent_census.csv`：相邻epoch actor agreement、平均绝对ΔQ、Q分布IQR和归一化ΔQ。
- `post300_checkpoint_ranking.csv`：e301…399每个候选的左右两侧稳定性。
- `checkpoint_selection.json`、`selected_model.pt`（指向models下原checkpoint）、`REPORT.md`和 `COMPLETE.json`。

选择规则：两侧最小actor agreement先要求≥.96；无候选则退到.95，再无候选则取观测最大agreement。通过门槛后最小化两侧平均ΔQ/IQR，再依次按raw ΔQ更小、最小agreement更高、epoch更晚破同分。IQR用本机midpoint经验分位数，零IQR加1e-12分母保护。e300/e400只提供邻居，不是两侧候选。

看指标时注意：

1. train V-loss使用target Q、validation V-loss使用online Q，这是保留的本机口径，不能把两条曲线当同目标的直接差值。
2. Q/V loss变大可能包含值尺度漂移；结合actor agreement、归一化ΔQ与advantage/weight看，不能仅凭loss断言策略失效或收敛。
3. actor loss只统计自由状态。logged/greedy skip率来自记录状态上的离线前向，不是实际闭环reuse率、speedup或视频质量。
4. 相对稳定的checkpoint不保证更高PSNR；test和VBench从未参与这一步选择。

使用 `checkpoint_selection.json`记录的checkpoint和SHA开始测试；不要将选中的checkpoint复制到结果盘变成第二份权重。

## 6. 固定协议Vbench200推理

协议：单张48GB GPU、batch1、832×480、81帧、16fps、UniPC50、shift5、CFG5、seed42、BF16 DiT；T5/DiT/VAE驻留GPU，`offload_model=False,t5_cpu=False`，关闭FSDP/SP和prompt rewrite/extension。每次run先做一次不计入结果的native full warmup，然后持久pipeline处理全部prompt。

```bash
export FLOPS_PROFILE="$COLLECTION_ROOT/calflops_profile.json"
export VBENCH_BASELINE="$EXP_BASE/ours21_vbench200_baseline_v1"
CUDA_VISIBLE_DEVICES=0 bash experiments/rl_inference_v1/run.sh \
  --wan21-root "$WAN21_ROOT" --checkpoint-dir "$CHECKPOINT_DIR" --baseline \
  --prompts "$OFFICIAL_CODE/Vbench200/prompts.jsonl" \
  --flops-profile "$FLOPS_PROFILE" --output-dir "$VBENCH_BASELINE"

export SKIP_BUDGET=25  # 示例K；不是已标定的某个speedup
export VBENCH_CANDIDATE="$EXP_BASE/ours21_${STATE_MODE}_vbench200_K${SKIP_BUDGET}_v1"
CUDA_VISIBLE_DEVICES=0 bash experiments/rl_inference_v1/run.sh \
  --wan21-root "$WAN21_ROOT" --checkpoint-dir "$CHECKPOINT_DIR" \
  --policy-checkpoint "$ANALYSIS_DIR/selected_model.pt" --state-mode "$STATE_MODE" \
  --skip-budget "$SKIP_BUDGET" \
  --prompts "$OFFICIAL_CODE/Vbench200/prompts.jsonl" \
  --flops-profile "$FLOPS_PROFILE" --output-dir "$VBENCH_CANDIDATE"
```

K范围0…48，保证每条候选恰好K个完整step复用。各预算应分别放新目录；可复用同协议、prompt、seed、同GPU的baseline，不能使用不同offload/24GB协议或不同shape的速度参照。

若使用目标speedup而非显式K，替换为 `--target-speedup S --calibration FILE`。这是另一个**speed→K映射**，与造数据所用speed→mean-threshold不同。合同见 `configs/speed_to_k.pending.json`；必须来自Wan21真实测量，条目为 `skip_budget` 和 `calibrated_speedup`，`status=calibrated`，固定protocol/forced_steps正确；选择最近实测点，同距取较小K。没有标定就使用显式K并报告实测speedup，不把K直接命名为1.8×/2.4×等。

成功生成后应有200个 `videos/vbench200_XXX.mp4`、对应 `timings/` 与 `traces/`、`run.json`、`components.json`和 `COMPLETE.json`。该marker只表示生成完成，不代表质量评测完成。单prompt调试可用 `--prompt`，但单视频不能送进下面完整Vbench200入口。

## 7. Predictor overhead、性能与质量报告

每次实际小网络查询记录forward耗时和运算量；强制动作不调用actor，uncond不重复调用actor。

| 字段 | 口径 |
|---|---|
| `timings/<sample>.json → predictor.call_count` | 本视频真实actor查询次数，与trace的policy_queried/actor_queries核对 |
| `predictor.tflops` | 实际次数×每次网络forward FLOPs÷1e12，不是TFLOP/s |
| `predictor.network_cuda_seconds` | CUDA events仅包住policy_net.forward；全部调用之和 |
| `predictor.network_host_span_seconds` | CPU上是forward执行时间；CUDA上只是host dispatch span，不能冒充GPU执行时间 |
| `predictor.decision_wall_seconds` | state搬运、归一化、forward、finite检查、softmax/argmax及结果取回的wall总时间 |
| `predictor.calls` | 逐次调用索引和上述耗时，逐视频reset |
| `components.json → predictor_overhead` | 总/平均调用数、TFLOPs、耗时，及ratio-of-sums占generate百分比 |

MLP forward估计与本机Calflops0.3.2一致：Linear为2FLOPs/MAC且不计bias，affine LayerNorm为5/element，SiLU为1/element；scalar5每次270336 FLOPs，SEA7每次271360 FLOPs。估计范围不包括输入归一化、softmax/argmax、数据传输、SEA滤波。10个latent实验组按实际网络输入维数自动计数，不能套用SEA7数字；另记录逐视频特征提取wall时间，feature本身FFT/quantile等FLOPs未估计，明确留为null。

小网络在测量前做一次FP32 forward warmup，该次不计实际actor查询。CUDA event池在测量前初始化复用；不额外对每个预测调用synchronize，只在generate结束后解析事件。网络时间和decision wall是包含关系，**均已计入DiT/generate latency，不得重复相加**。decision wall是predictor决策路径的观测耗时，不是整个cache方法相对baseline的净增耗时；SEA滤波和其他控制逻辑不在该预测器字段内。K=0/48时actor查询数和相应TFLOPs/耗时为0。

```bash
conda run --no-capture-output -n wan2.2 python evaluate.py summarize \
  --baseline-dir "$VBENCH_BASELINE" --candidate-dir "$VBENCH_CANDIDATE" \
  --profile "$FLOPS_PROFILE"

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n wan2.2 python evaluate.py evaluate \
  --baseline-dir "$VBENCH_BASELINE" --candidate-dir "$VBENCH_CANDIDATE"
```

`summarize`生成candidate的 `performance.json`：完整generate速度、DiT TFLOPs速度，以及逐视频/全量predictor overhead；T5/DiT/VAE时间及TFLOPs分别保留。主latency是完整generate（不含模型加载、warmup、MP4写盘），不是只报DiT。缺少新predictor计时的旧结果会报错，不能补写0冒充测量。

`evaluate`调用同级VideoMetrics和VbenchEvaluation，输出：

- `evaluation/video_metrics/`：匹配baseline的81帧RGB PSNR、SSIM、LPIPS表/汇总。
- `evaluation/vbench_reference/vbench200_aggregate_scores.json`与`evaluation/vbench_candidate/vbench200_aggregate_scores.json`：16维分数及聚合。
- `evaluation/COMPLETE.json`：全部质量阶段成功后才写入。

报告应附checkpoint SHA、state mode、selection SHA、K或已标定目标、实际reuse/actor调用数、固定协议、完整generate latency/speedup、DiT TFLOPs、小网络TFLOPs/forward时间/decision wall及占比、T5/VAE分项，以及PSNR/SSIM/LPIPS/VBench各维与聚合分数。

这里是固定200 prompts的 **Vbench200子集**，不能写成官方完整VBench套件结果。16维加权聚合口径由同级评测脚本锁定；与采集archive的10维custom-input均值分开报告。原始component timing有嵌套关系，不能把DiT和其中的predictor再求和当总时间。

## 8. CPU验收与交接

```bash
cd "$OURS_PROJECT"
conda run --no-capture-output -n wan2.2 python experiments/rl_validation_v1/validate.py \
  --output-dir "$EXP_BASE/ours21_cpu_validation_UNIQUE"
```

验收覆盖核心IQL源码一致性、全部K可达性、两模式真实CPU反向更新、checkpoint保存/重载和控制器回放、predictor FLOPs对Calflops计数、耗时汇总、reset/forced-zero、抽样可复现、split隔离与远端root传播。合成2epoch smoke权重放models，production loader明确拒绝；这不是正式Wan推理GPU时序验收，也不是3000条正式训练。CUDA事件单测使用mock时只验证计量逻辑，不构造GPU实测耗时结论。

远端交接时保存selection、cache manifest/SHA、完整训练config/metrics、model checkpoints、post300选择及分析、生成timing/trace/videos、全部质量输出。目录级README和source hash用于核验。发生错误以FAILED/缺少完成marker判定，不能用部分目录宣称完整结果；不覆盖既有实验。
