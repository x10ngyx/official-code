# Ours4Wan21 在线微调配置复核

状态：下文保留历史参数讨论。当前用户配置已更新：训练目标1.5–3.5×；R4/R8从原VBench50固定抽20条评测1.8/2.4/3.0×，复用原同卡baseline，不计算VBench。当前执行配置以online.json、online.py show-config及在线手册为准；未启动正式在线实验。

结论：e385 的采集→累计 replay→IQL 续训→checkpoint 选择→评测结构可以复用；原参数是可运行的历史设置，尚无证据证明它能稳定改善离线策略。用户已确定保持框架、每轮20个joint replay epochs、在e11–e20按actor一致率选择checkpoint；VBench20每4轮一次，prompt名单由远端确定。实现默认采用5个critic-only replay epochs、tau=.7与beta1/cap20。离线约200epoch稳定不足以推导在线最优预算，所选20epoch也不表示已经证明最优。

## 已核实的 e385 协议

来源为 `work/online-finetune/experiments/e385_uniform_replay8_policy_sampling_v1/{common.py,training_worker.py,collector_worker.py}`、`online_finetune/trainer.py` 及已运行结果中的冻结 `protocol.json`；R9–R16 续跑保持参数，仅延长轮数。

| 项目 | 原 e385 设置 | 判断与迁移建议 |
|---|---|---|
| 轮数 | 原8轮，后扩到16轮；偶数轮评测 | 首版沿用8轮组织方式；评测已改为每4轮一次（8轮时为R4/R8）。轮数本身不保证收益，16轮尚无完整结果支持 |
| 评测 | VBench10，原8轮中的R2/R4/R6/R8 | 用户确定改为VBench20，每4轮一次；20个prompt由远端机器确定，本地不预选 |
| 每轮采集 | 3000-prompt池有放回抽100条，每条50步 | 100条可作起点；Wan21只从训练侧prompt池采集，必须排除validation/test及评测prompt，不能照搬池文件 |
| 目标 | `[1.3,3.5]`分100个等宽区间，每区间随机抽一点 | 可保留目标范围和分层规则；必须使用Wan21实测speedup→K标定并检查实际K覆盖；speedup均匀不等于K均匀 |
| 行为策略 | 当前actor categorical直接采样；无greedy/uniform mixture、温度改动、cap/cooldown/Q过滤 | 保留；采样是否提供有效探索取决于actor熵和自由动作覆盖，不能仅凭“随机采样”断言充分探索 |
| Replay | offline train + 所有已完成online轮次，transition均匀有放回 | 保留，不改固定50/50或recency；每轮依据真实train数据量计算比例，旧3043轨迹/152150行含anchors，不能搬到Wan21 random3000 |
| Batch | 256 | 保留 |
| Critic warm-up | 每轮50个batch，只更新V/Q及target | 仅0.081个旧R1 replay epoch；建议讨论5个critic-only replay epochs，尚未证明充分或最优 |
| 联合更新 | 每轮50 replay epochs，每epoch `ceil(N/256)` 个有放回batch | 用户已确定改为20个joint replay epochs；epoch仍为 `ceil(N/256)` 个有放回batch，预热单独计数 |
| Actor LR | `4e-5` | 可保留；较离线`1e-4`低，但不能抵消更强权重及大量重复更新 |
| Q/V LR | `1e-4` | 可保留起点；已有结果不能单独归因于此LR，不建议同时再降LR以增加混杂 |
| Expectile τ | `.7`，现行Wan21离线`.6` | `.6`是温和上expectile，`.7`是合理候选；没有证据证明`.6`过度保守或`.7`导致旧实验退化。切换tau需要让critic适应新目标 |
| 优势 β / cap | `1.5 / 10`，现行Wan21离线`1 / 20` | 建议继承`1 / 20`；旧组合强化中等正优势、抑制负优势、截断极端正优势，不能统称更保守 |
| γ / reward | `1 / terminal absolute RGB PSNR`，scale1 | 保留，与有限50步、终局质量目标一致 |
| Target EMA | 旧参数权重 `rho=.999`，新参数权重`.001` | 保留以减少改动；半衰期约693次Q更新，随更新预算一起解释 |
| AdamW | WD`.01`，betas`(.9,.999)`，eps`1e-8`；固定LR，无scheduler | 可保留；不是从原论文推出的最优值 |
| Gradient clip / KL | grad norm1；KL系数0 | 保留；KL=0无显式策略距离约束，漂移需记录 |
| Normalizer | 冻结起点train-only统计 | 必须保留；新增数据不能直接重新拟合后套用旧网络 |
| Actor mask | native及Exact-K强制行排除actor，保留critic | 必须保留，Wan21 native仅0/49，不能继承Wan22的32步边界 |
| 优化器/RNG | R1新optimizer；R2+继承所选checkpoint的optimizer与RNG | 可保留该实验语义，必须显式实现；Wan21离线checkpoint本来有optimizer，不能靠“字段不存在”隐式重置 |
| Epoch选择 | 固定1933状态；e30–50按连续两对argmax一致率的较小值最大化，再最小化概率漂移，平局选更晚；必选 | 用户确定从e10之后按actor一致率选点，候选e11–e20；沿用原连续两对一致率排名及平局处理。固定状态在Wan21重建，不继承1933硬编码 |

## 更新量复算与局限

e385每轮新增5000 transitions，旧offline train为152150行。第r轮总池 `N_r=152150+5000r`，每轮计算的joint updates为 `U_r=E*ceil(N_r/256)`。

这里UTD定义为本轮joint batch更新数/本轮新增transition数，**不是**每条新数据的重复使用次数；每条池内数据的期望抽样次数为 `256*U_r/N_r`。有放回“epoch”只是抽样预算，不保证每条数据恰好遍历一次。

| 轮次 | 累计online占比 | 最新一轮占比 | 50epoch joint updates | joint UTD | 每条数据期望抽样次数 |
|---|---:|---:|---:|---:|---:|
| R1 | 3.18% | 3.18% | 30,700 | 6.14 | 50.01 |
| R8 | 20.82% | 2.60% | 37,550 | 7.51 | 50.03 |
| R16（协议推算，非完成结果） | 34.46% | 2.15% | 45,350 | 9.07 | 50.01 |

以上不含50次warm-up，且为计算到e50的预算。下一轮继承所选epoch，未选中的后续更新不进入lineage。例如R8选e43，继承32,293次joint + 50次warm-up，而非全部37,600次更新。

早期online占比低和单条online重复使用多可以同时成立：R1中96.82%的抽样来自offline，但新数据仍每条期望约50次。旧online还会在后续轮被反复重放；这些比例不足以证明过拟合，但说明50epoch并非轻量更新。

按用户确认的E=20，旧e385 R1规模对应12,280次joint更新，UTD2.456，每条期望约20次，相比50epoch减少60%的joint更新。Wan21必须代入其真实train split行数，不可直接沿用这组更新数。

50次warm-up在R1仅期望抽到约407个online行（允许重复）；target EMA经过50次更新后，初始target仍有约95.12%的线性系数。因此“50次”只能称短预热，不能断言critic已适应或收敛，也不能把693次EMA半衰期机械当作最优warm-up长度。

## 优势权重为何不是简单变保守

本地对actor-eligible行做batch内优势标准化：`z=(A-mean(A))/max(std(A),1e-6)`，权重为 `min(exp(beta*z),cap)`，actor loss按有效行数平均，未按权重和归一化。

| z | 离线 β1/cap20 | e385在线 β1.5/cap10 |
|---:|---:|---:|
| -1 | .368 | .223 |
| +1 | 2.718 | 4.482 |
| +2 | 7.389 | 10 |

截断起点从 `ln(20)=2.996` 降到 `ln(10)/1.5=1.535`。cap变小只约束极端权重，β提高则增加中等正优势的相对偏重；有效actor梯度强度还与batch优势分布、平均权重和梯度裁剪有关。

官方IQL使用原始优势的指数加权而非本地的batch标准化优势，因此不能直接拿官方temperature数值当本地β的推荐值。[官方actor实现](https://github.com/ikostrikov/implicit_q_learning/blob/master/actor.py)。官方在线脚本将offline与新增数据放入同一replay，每次环境step后做一次batch更新，可支持单池结构的合理性，但不能证明本任务的最佳UTD为1。[官方在线脚本](https://github.com/ikostrikov/implicit_q_learning/blob/master/train_finetune.py)。官方AntMaze在线配置保持固定LR；其任务和本项目不同。[官方配置](https://github.com/ikostrikov/implicit_q_learning/blob/master/configs/antmaze_finetune_config.py)。

## 已完成结果提供的约束

直接读取原结果 `run/round_{002,004,006,008}/evaluation/metrics.json`：共同10 prompts×5档的mean PSNR为离线e385 22.276804 dB；R2/R4/R6/R8分别22.087058/22.195551/22.125787/22.121040 dB，差值−.189747/−.081253/−.151018/−.155764 dB。四个已评测在线轮次均未超过起点。

独立30-prompt扩展的 `strict_summary/checkpoint_selection_report.json` 为R8 22.078327、离线22.283258 dB，差值−.204930 dB。以上为相同prompt/目标预算的描述性质量比较；没有本轮重跑生成、bootstrap或逐视频指标。不能归因为某一超参数，也不能据此推断Wan21必然退化。

稳定性选择不看生成质量，且在e30前改善、e30后稳定退化时仍必须选择一个后段模型。“相邻epoch稳定”不等于“相对parent未退化”。按用户沿用框架的要求，不在本轮擅自引入质量gate或KL惩罚；实际生成评测仍应报告与离线起点的差异。若评测集用于参数/轮次选择，它应视为开发集，另保留最终测试集。

## 当前确认项与待讨论参数

保持8轮、每轮100条、batch256、直接policy采样、累计单池uniform replay、actor LR4e-5、Q/V LR1e-4、γ1、rho.999、WD.01、clip1、KL0、固定normalizer、原生/Exact-K actor mask，以及R1重置/R2+恢复optimizer/RNG的结构。

**已确认：每轮20个joint replay epochs，在e11–e20按actor一致率选择checkpoint。** 预热单独计数，joint epoch从e1开始，不因预热长度而偏移。

按原e385框架落实排名：在本实验冻结的同一组actor-eligible状态及同一组归一化权重上，令 `A_e = weighted_mean[argmax(pi_e) == argmax(pi_(e-1))]`，候选epoch e的主分数为 `min(A_(e-1), A_e)`；主分数越大越优。这是原框架连续两对相邻epoch一致率的定义。e11使用e9→e10及e10→e11两对，e9/e10只提供比较历史，不进入候选。主分数相同时沿用原框架：两对概率漂移的较大值越小越优，再平局选更晚epoch。仅使用真实actor输出，排除native/Exact-K强制行；不设新一致率阈值，不按actor loss选点。每epoch保存模型，完整计算到e20，再将入选checkpoint及其optimizer/RNG传到下一轮。

**已确认评测：VBench20，每4轮执行一次。** 即R4、R8、R12……；若首版为8轮，则只在R4/R8做轮次评测。具体20个prompt由远端机器确定，本地不选择名单、不假设它是现有VBench10或VBench30的某个子集。远端首次评测前冻结20个唯一prompt的文本、ID、顺序与文件SHA；各轮及对应离线起点比较使用同一名单，且与训练/在线采集池隔离。沿用原五档时，每个评测checkpoint为20×5=100个candidate cells。复用baseline与离线起点结果须匹配同一prompt、协议和checkpoint，不能将旧VBench10的50-cell完成标记当作VBench20完成。未额外增加不在4轮间隔上的末轮评测。

**实现默认：5个critic-only replay epochs、tau=.7、beta=1/cap20。** 用户随后要求按讨论配置实现，现将这组讨论值落实为默认；它们不是消融验证的最优值。

1. **预热单位与作用。** 原R1每epoch614次更新，50次预热仅0.0814epoch，是后续joint更新数的0.163%。每条数据预热期望曝光0.0815次。若改5个replay epochs，为3070次更新、每条期望约5次；EMA初始target的线性系数从50次后的95.12%降为4.63%。这些是预算/滤波时间尺度说明，不能作为critic收敛证明，不能声称预热必须占joint固定百分比。Wan21按实际train+online池重新计算更新数。
2. **IQL结构边界。** 本实现的Q/V目标依赖固定replay、target Q及V，不使用当前actor输出。每轮固定数据阶段的warm-up主要让actor延后学习更新中的优势，并给Q/V额外更新；不是传统on-policy actor-critic的同步需求。为研究warm-up本身，须区分“actor延后开启”和“critic总更新数增加”，不能把两者收益混为一谈。继承的critic并非随机初始化，所以也没有理论要求必须先充分收敛才可联合训练。
3. **离线200epoch与在线预算。** 用户报告离线约200epoch稳定，当前没有指定曲线和稳定性指标，不能把它自动解释为全部Q/V/actor共同收敛。在线继承已有网络、target、normalizer及后续轮optimizer，新数据规模、支持和tau变化决定适应难度；epoch长度还随replay池变化，因此没有200乘某个固定比例的可靠换算。此前优先推荐10epoch主要出于降低曝光，是弱工程启发，现保留为低预算候选，不作默认最优判断。
4. **20epoch预算的解释边界。** 20epoch是用户已确定的每轮joint预算，替代之前的50epoch建议；选点窗口同步改为e11–e20。旧R1–R6的后5对actor agreement均值约.961–.968，但normalized Q漂移仅1/6轮低于离线局部参考，且未记录独立validation loss。因此仍需区分actor一致率与critic/生成质量，不能把选中checkpoint称为已经收敛或质量最优。
5. **tau的精确含义。** 固定同等绝对残差时，正/负平方残差的系数比为tau/(1-tau)：.6→1.5，.7→2.333，.8→4，.9→9。tau=.6确实偏温和，但不是60%分位数，也不能凭此认定过度保守。较大tau提高对支持内高Q动作的重视，同时可能放大函数逼近误差；本地标准化优势下也不能简单推导tau增大必然使actor更激进。
6. **当前tau建议。** 保留原e385的.7作为Wan21在线候选有合理依据；此前建议.6是减少离在线目标变化的对照原则，并非证明.6更好。官方MuJoCo配置使用.7，AntMaze使用.9，原论文指出更大tau有助于其轨迹拼接任务；不等于本视频任务应直接取.9。[MuJoCo配置](https://github.com/ikostrikov/implicit_q_learning/blob/master/configs/mujoco_config.py)、[原论文§5.2](https://arxiv.org/html/2110.06169)。若tau从离线.6改为.7，先冻结actor适应Q/V新目标比同时提高beta更易解释。beta1/cap20仍是与原e385不同的待确认建议。

迁移的必要适配：Wan21 train-only prompt池和offline缓存；对应feature-mode的起点checkpoint、normalizer和固定状态；Wan21实测speedup→K（本地当前仍pending）；81帧/UniPC/shift5/CFG5/48GB单卡全GPU常驻；0/49强制步；checkpoint optimizer列表与旧框架value/q/policy字典、RNG字段的转换。GPU资源门限、绝对路径和22/26/26/26分配也不能当算法超参数照搬到远端机器。

计量沿用Ours4Wan21的完整generate latency及T5/DiT/VAE分项时间/TFLOPs，另报predictor overhead；质量使用VideoMetrics PSNR/SSIM/LPIPS及vbench_score。建议训练记录自由动作覆盖、熵、优势std、权重cap率/ESS及策略/Q漂移，避免只依据loss或固定状态一致率解释收益；这些诊断建议尚未实施。

## 本地证据入口

- 旧冻结协议和轮次结果：`work/online-finetune/experiment_results/wan22_e385_uniform_replay8_policy_sampling_gpu0123_20260826_013312/`。
- 30-prompt比较：`work/online-finetune/experiment_results/wan22_e385_online_r8_vbench30_seed42_gpu0123_then_resume_r9_r16_20260829_002513/strict_summary/checkpoint_selection_report.json`。
- Wan21离线参数/语义：`configs/training.json`、`ours4wan21/{train.py,local_iql.py,policy.py}`。
- 实现边界：新增独立在线执行代码及采样接口；离线IQL损失内核和400epoch流程保留。验证仅CPU，未启动正式Wan/VBench实验。
