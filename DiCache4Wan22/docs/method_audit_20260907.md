# DiCache4Wan22 方法一致性审查

审查时间：2026-09-07 13:52 CST。范围：官方缓存算法、Wan2.2 接口、双专家状态、计量接入和现有验证的证据边界；未修改方法代码或启动完整 A14B 推理。

结论：缓存计算段与锁定的官方 Wan2.1 实现一致，未发现额外修改 probe、gate、DCTA 或残差数值语义的情况。接口与状态隔离适配必要且合理；“全局 retention + 新专家首个 CFG 对初始化”是合理的最小迁移选择，但不是官方定义的双专家策略。因此可以称“保留官方缓存算法的 Wan2.2 迁移”，不能无条件称为整个方法执行过程、缓存轨迹或质量完全相同。

## 来源与验证

- 官方源：[Bujiazi/DiCache@fdbe20b / WAN2.1/run_wan_dicache.py](https://github.com/Bujiazi/DiCache/blob/fdbe20b669c9174bbed5ec994de073fd881c8010/WAN2.1/run_wan_dicache.py)。本次从固定 commit 下载到内存，与 vendor 比较，23738 bytes 逐字节一致，SHA256=`49edfa4033d1af2983a6db07354d87b142cacdf232566bb37a2c2b7b7853ed04`。
- Wan2.2 固定为 `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`；本次源码及 36 项包/依赖文件锁验证通过。
- conda `wan2.2`、四个 BLAS 线程变量显式为 1，重跑 `tests/run_tests.sh`，17/17 通过。含 12 组、1200 次小模型调用的官方输出/决策对照及单专家未改生命周期对照。
- 既有 CUDA 小模型记录为 all-full 100/100、重复 cache 100/100 精确一致；本次只查阅记录，没有重跑 GPU。完整 A14B 视频、真实形状组件 profile 和质量/速度仍未实测。

## 没有改变的官方语义

`runtime/official.py:34–53` 直接编译官方文件 `148–207` 行的 AST，只附加返回隐藏状态和观测信息的语句。

| 项目 | 核对结果 |
| --- | --- |
| Probe | 固定首 1 个完整 block；gate 拒绝 reuse 时从 probe 输出继续剩余 blocks，不重复执行首层。保留期直接跑全部 blocks。 |
| 判断信号 | 浅层输出与该 CFG 分支前一次浅层输出的 relative-L1；独立累计，严格 `< threshold` 才 reuse，触发 full 时累计量清零。 |
| CFG | cond/uncond 独立缓存和 gate；可出现一个分支 full、另一个分支 reuse。没有 CFG 同步决策。 |
| 残差 | 全部 blocks 输出减输入；不是最终噪声输出缓存，也不是仅剩余 39 层的残差。 |
| DCTA | 当前 probe 残差与倒数第二份历史的 L1 变化，除以最近两份 probe 历史的 L1 变化，gamma clip 到 `[1,2]`；使用原来的两份 full 残差外推。 |
| 更新时机 | full 残差及 DCTA 窗口只在 full 时更新；`previous_input`、`previous_internal_states` 在 reuse 时也更新。因此 gate 的“前一次”不同于 DCTA 的“上一次 full”。 |
| 历史及精度 | 保留 `len <= 2` 实际形成三份窗口、未用于 gate 的 `delta_x`、`previous_output`、无 epsilon 除法和原地 `x += ...`。单残差 fallback 的非原地加法也保持原样。 |
| 额外约束 | 未增加最终步 full、最大连续 skip、Exact-K、SEA filter、阈值映射或其他方法的保护规则。 |

## 迁移项及合理性

| 适配 | 代码位置 | 判断 |
| --- | --- | --- |
| 使用 Wan2.2 原生 forward 前后处理，只替换 blocks 循环 | `runtime/official.py:86–103` | 必要且正确。Wan2.1 的 timestep embedding 为 `[B,D]`、projection 为 `[B,6,D]`；Wan2.2 为 `[B,L,D]` 和 `[B,L,6,D]`。保留新模型的 embedding、block kwargs、head、unpatchify 避免错用旧接口。 |
| 状态从类共享属性改为每视频、每专家的实例状态，每专家仍保留两份 CFG 状态 | `runtime/official.py:68–83,133–138,166–168` | 必要且正确。high/low 权重不同，不能跨专家复用 full/probe 残差或误差累计；原官方类属性直接套两个实例会共享可变列表。 |
| 全局计数与新专家首对初始化分开 | `runtime/official.py:176–190` | 初始化必要；保留全局 retention 是合理选择，并非唯一迁移方案。新专家首次 cond/uncond 各设置官方 cnt=0/1 走原始初始化，其余调用回到全局计数。避免空历史、避免把高专家历史当低专家历史。 |
| 专家切换前释放 high 历史，最后 uncond 后释放全部历史，finally 恢复接口 | `runtime/official.py:155–165,205–226` | 单卡 offload 下合理。普通 Python 容器缓存不会自动随权重 `.to('cpu')` 迁移；高专家退出后不再使用其历史，提前释放不改变后续数值。当前实现明确只支持 high→low、每步 cond→uncond 的顺序，与锁定 sampler 一致。 |
| 模型层数/宽度采用 A14B 原生配置，probe 仍为 1 | `runtime/official.py:130–132,198` | 正确。full 用 `len(model.blocks)`；每专家 40 层时 reuse 算 1 层、full 算 40 层，未按层数比例扩大 probe。 |
| 使用 45 帧、DPM++、shift12、CFG(3,4)、boundary.875 等项目冻结协议 | `runtime/generation.py:10–25` | 符合用户要求和目标模型协议；帧数/solver/CFG 是实验协议变化，不应全部称为接口强制要求，也不能据此期待 Wan2.1 相同缓存决策/质量。 |
| 参数检查、baseline 原生路径、trace 和 full/probe 计量 | `runtime/official.py:56–65,141–153`；`runtime/generation.py:60–85`；`experiments/performance_t2v_a14b/profile_calflops.py:149–174` | 合理的工程接入，非缓存公式适配。拒绝无法初始化两分支的 retention。原生 baseline 无缓存记账，DiCache threshold=0 仍有记账开销。trace 的 scalar 转换可能同步 GPU，完整 generate latency 含该开销；FLOPs 排除 gate/DCTA/残差加法，应保留已有披露。 |

## 需要准确披露的差异

1. 默认 50 步对应 100 次 CFG 调用，retention=.2 表示前 20 次调用即 step0–9 full。冻结协议在 step32 首次进入 low，其 cond/uncond 额外 full；step33 起立即回到全局 gate，而非重新运行 low 阶段 20% 保留期；step49 没有额外强制 full。这里的 step 从 0 开始。
2. low 初始化后每个 CFG 分支只有一份完整残差。即使 gate 决定 reuse，也只执行官方 `x = x + residual_x` 回退。必须等该分支第二次 full 后，后续 reuse 才使用两历史 DCTA；如果一直 reuse，DCTA 可在整个剩余 low 阶段都不启用。这与官方默认单专家 10 步 warmup 后通常已有充分历史的状态不同，但未改变原公式。主动增加第二次 full 是另一种策略，不是现有公式正确性的必要修复。
3. `tests/test_official_parity.py:209–214` 的双专家 oracle 主动向官方参考注入相同的专家切换及 cnt 规则。通过测试证明“给定该迁移策略，计算结果等价”；不能证明这个双专家策略出自官方，或证明它优于阶段独立 warmup。单专家测试则确实对照未改的官方生命周期。
4. 无 epsilon 的分母为零可产生 NaN，已由现有测试明确覆盖。这是忠实保留的上游行为，不能当成 Wan2.2 迁移引入的问题；若以后加 epsilon、强制 full 或改 dtype，应单独标注为数值稳定性变体。

本次没有发现要求立即修改缓存算法的实现偏差。结论中的“合理”限于结构、数值语义和工程生命周期；A14B 上的性能与质量有效性尚无实测证据。
