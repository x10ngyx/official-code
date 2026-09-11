# BLOC 特征设计与评价协议（设计草案）

目标是在相同精确 reuse 预算 K 下提高最终配对 PSNR，同时保留实际推理加速。
底层完整状态的 MDP reduction、动作定义、预算计数与 action mask 不变。
仅比较观测表示；不证明严格 Markov 性，不要求直接重建 PSNR。

## 1. 已核实的当前实现

源码路径相对 Ours4Wan21/；SeaCache 路径相对父仓库。

- `ours4wan21/runtime.py:apply_policy` 在 cond forward 的 patch embedding 之前调用
  `observe_latent`，每步一次；uncond 共用该向量。
- 真正决策在 `SeaCache4Wan21/wan21_integration.py:seacache_forward` 中，已完成 patch
  embedding、时间嵌入、文本投影和首 block 的 norm/modulation，但未执行 Transformer blocks。
  `Controller.plan_step` 调用 SEA filter 后计算观测、检查预算 mask，再查询 actor。
- 当前 raw latent x_t 是本步 scheduler 更新之前的 `[16,21,60,104]` 输入，模型外部为
  FP32；既有 latent 特征路径显式 FP16→FP32，匹配离线存储精度，不修改模型输入。
- latent 特征的 cached latent c_t 是最近一次实际 recompute 的 raw 输入 x_r，r<t。
  在 observe(t) 之后 commit(action_t) 时，若 recompute 则缓存 x_t；reuse 保留旧缓存。
  它不是上一步 latent，不是预测的干净视频 latent，也不是 scheduler 更新后的 latent。
- SeaCache 的 cached residual 按 cond/uncond 独立保存，为最近一次重算的
  `TransformerBlocks(token_input, conditioning)-token_input`。只有重算 blocks 完成后
  `record_recompute` 更新；reuse 加到当前 patch token，随后仍执行 head/unpatchify。
- SEA distance 的缓存是每一步的 filtered first-block modulated input，包含 reuse 步；
  与仅重算时更新的 cached raw latent 是不同对象。相邻相对 L1 比较的是两步各自 sigma
  下的 filtered feature；累计值在决策前加入本步距离，重算后清零。
- SEA7 为 `[adjacent_distance, accumulated_with_current, cached_valid, t/49,
  K/50, used/50, consecutive/50]`；Scalar5 去掉两个 SEA 距离，不能替代本任务主基线。
- Dynamics128 是 16 通道×8 项 raw/SEA-filtered latent 短期变化统计，加 SEA7 共135维。
  除已有 token SEA filter 外，它额外对 raw latent 历史作 FFT filter；不能预设其开销可忽略。
- Exact-K 是共享 cond/uncond 的去噪 step 预算，每条50步恰有K次 reuse，双方缓存独立。
  forced=(0,49)，可行 K∈[0,48]；预算耗尽、剩余可用步数恰等于剩余预算时均为强制动作。
  actor_mask 只在两个动作均可行时为1，训练和动作区分分析均应用此 mask。
- 每次 generate 重置 residual、previous feature、accumulator、预算和特征历史。
  Wan2.1 是 single expert，无第32步切换；第49步强制重算但其决策前历史仍保留。
  当前 observation 合同要求 cached_valid==(step>0)，不能直接支持中途清缓存。
  扩展 Wan2.2 前需另行实现 expert/stage 转换与预算可达性，不能借用本实验单阶段假设。
- `local_iql.compute_normalizer` 仅用 train_indices 拟合 mean/std（std下限1e-6），
  policy 和 online_training 沿用 checkpoint 统计，不在线重新拟合。
- 当前 train.py 正式入口要求 TrainingConfig 完全等于默认值（400 epoch、seed42），
  支持多训练seed必须增加显式的独立实验合同，不能用 smoke 绕过正式校验。
- 当前离线数据按实际轨迹 reuse 总数回填 K，是已有固定预算离线学习构造；所有组保持一致。
  这不是可输入未来 latent/最终 PSNR 的许可，线上K仍是生成前指定。

## 2. 离线数据与缺失信息

现有 frozen selection 含3000轨迹、1000 prompt：train/val/test=2400/300/300轨迹。
每prompt三条随机threshold轨迹，生成seed42；保留原prompt split，不重新跨集合抽样。

采集 `data_collection/src/ours4wan21_data/runtime.py` 在 cond/uncond 之前记录 x_t，
在生成计时外保存 `step_NNN_input.pt`，FP16 `[16,21,60,104]`，每轨迹50份。
trace保存 timestep、sigma、model_stage、实际双分支动作、已有 SEA 距离及累计值。
因此可按历史动作重建 c_t，A/B/C 方向不需要重新生成训练视频。
已读取实际完成条目和 step_records 核对字段；全量重新提取仍需逐条验证路径/哈希/形状。

缺失：cached token residual 张量、token input/filtered feature 张量、原FP32 latent完整精度，
以及任意中途状态的 scheduler 多步求解器内部状态和 RNG 快照。
涉及 residual 的候选需要在前一次重算完成时补存 residual 或其预先定义的摘要，
下一决策步才能读取；不可用当前步动作完成后的 residual 代替。
分支动作价值诊断需补采完整决策前状态（scheduler history、两支缓存、历史、预算、RNG），
只有输入latent不足以精确分叉 UniPC；当前优先实验不依赖此诊断。

## 3. 紧凑候选（待原型验证，非效果结论）

设 x=x_t，c=x_r，d=x-c，所有归约在每通道的T/H/W轴进行。
ε=1e-6，r(z)=sqrt(mean(z²))，s=(r(x)+r(c))/2+ε。
所有张量先FP16→FP32；不读baseline、奖励、未来状态，不额外执行DiT或VAE。

A（64维，16通道×4）：
1. r(d)/s：对称归一化差异；比仅除以接近零的缓存范数更稳定。
2. mean(x*c)/max(r(x)*r(c),ε²)：余弦，clip[-1,1]，任一范数≤ε时定义为0。
3. mean(d)/s：带符号相对漂移。
4. (r(x)-r(c))/s：带符号相对幅度变化。
A无需额外FFT；缓存c及其每通道统计，重算时更新。

B（64维）：对d/s做轻量adaptive average pooling至(1,2,2)，每通道4个粗空间格。
使用16×4个带符号格均值；保留相对缓存的低频结构。
此版本沿时间轴平均，不声称测量真实运动；分帧/时序粗格可作为后续替代，首轮不扩大组合数。
A+B共128维，便于与Dynamics128比较同输入维数、同网络规模。

C（16维）：从A的4类通道统计各取跨通道mean和std，得到q_t∈R^8；
输出 `[q_t-q_{t-1}, q_t-mean(q_{t-1},q_{t-2},q_{t-3})]`。
均值只含过去可用值，不含当前值；无历史则相应差分为0。
缓存更新引起的变化是真实已执行动作的后果，可由SEA7的缓存年龄/连续reuse解释。
只保存至多3个8维历史向量；不新增过去完整latent，不用Δsigma除法放大末期噪声。
C依赖A的提取，但可单独移除A输出做输入消融，报告其实际提取成本。

候选：SEA7+A64、SEA7+A64+B64、SEA7+A64+C16、SEA7+A64+B64+C16。
主基线SEA7；现有版本SEA7+Dynamics128。最佳组合再做删除单一特征组的消融，
已有单组结果可复用，但不能把不同训练预算或不同seed的结果当消融。

无缓存时全部新增特征为0，由原cached_valid表示；首个有效q不与无缓存零哨兵做差。
每视频重置；任何将来的expert/reset事件必须同时清缓存、摘要和趋势历史，并强制重算。
当前单expert实现应拒绝未声明的stage改变，而不是静默继续。
噪声阶段由同一固定scheduler的t/49提供；按实际sigma预登记高/中/低三段分层报告，
不按测试效果重新调分界；首轮不为不同组增加不同时间输入或不同normalizer。

## 4. 分阶段比较

以下为建议实验规模，用户尚未确认规模时不视为已承诺的运行配置。

- 先原型正确性/因果性检查、实际latent微基准，确认输出维数、无缓存/零范数、历史更新和
  离线/在线一致。记录wall/CUDA时间及额外allocated/reserved显存，含CPU传输；
  微基准只是提取开销，不替代完整rollout。
- 单训练seed筛选：所有组同3000轨迹、同400 epoch/更新次数和IQL配置；网络同3×256。
  两基线+4候选。可复用严格同合同的已有seed42基线训练；checkpoint选择规则所有组一致。
  特征候选排名用验证prompt上的实际闭环PSNR/耗时，不能只按actor loss或稳定性排名。
- 验证集预固定少量prompt与K23/K29/K35，先做闭环筛选，再只对最多2个候选扩测；
  不在现有已多次查看的VBench50上选特征，避免把测试集用作验证集。
- 入选候选、SEA7、Dynamics128及必要组消融，建议训练seed=42/43/44；
  生成seed始终42（遵守Wan2.1固定协议），固定未参与选择的test prompt和全部三个K。
  划分审核同时使用sample_id和规范化prompt文本，重复文本不得跨集合。
- 所有组保持相同隐藏层/宽度并报告输入、actor和全部可训练网络参数量；
  A+B与Dynamics128同维。其他组的第一层参数差异明确报告，不宣称完全等参数。
- 等预算在线微调采用相同prompt/K/采样seed计划、round数、每round交互数、更新次数、
  replay混合及checkpoint规则；normalizer冻结。采集不含val/test prompt。
  当前现成8round×100轨迹、每轮5critic+20joint epoch配置可作为统一预算，
  但多方法运行前须冻结统一比较合同与独立结果目录。

## 5. 验收和交付

主结果逐prompt/生成seed/K配对，报告PSNR差均值、median、胜率与prompt聚类区间；
跨训练seed分别展示均值和波动，不能把同prompt多个K/seed当独立样本夸大样本量。
同时报告原VideoMetrics PSNR/SSIM/LPIPS、VBench评分（固定测试集的适用维度与聚合
需明确，子集不得冒称完整官方VBench200；本轮未新增取消评分指令）。

速度报告完整generate latency和同卡native baseline加速比，分别记录T5/DiT/VAE、
feature、policy时间，后两者已嵌套于generate，不重复求和。
主计算量报告实际DiT TFLOPs，保留T5/VAE和policy TFLOPs；未计量的feature运算明确未知。
每轨迹核验50步、100双CFG调用、原首尾强制重算、两支各K次reuse且实际blocks一致。
动作区分/噪声分层统计只计actor_mask=1的步骤，强制步只用于开销和完整性统计。
最终交付公式/维数/来源/计算时机/成本、闭环结果与消融、最小有效组合，以及无效/不稳定证据。
允许结论为所有候选均未优于SEA7或旧特征，不为增加特征而增加特征。

## 6. 可选经验动作价值诊断

仅在解释效果需要时，保存同一决策前完整状态并选择两个动作均可行的步。
reuse/recompute分支分别调整剩余预算，用同一后续策略规则闭环生成，各自满足相同总K；
不强行用同一后续动作序列。差值仅称给定后续策略下的经验动作价值差，不称最优Q。
