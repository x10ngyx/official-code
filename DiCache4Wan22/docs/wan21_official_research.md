# DiCache 官方 Wan2.1 实现调研

核查日期：2026-09-05。范围：官方源码静态审计、论文对照、Wan2.2 移植分析。没有运行模型、生成视频或测量性能。

## 结论

DiCache 是 training-free 的动态缓存方法，包含两个不可缺少的部分：浅层 online probe 决定何时复用，以及 Dynamic Cache Trajectory Alignment（DCTA）决定如何修正历史残差。复用步仍运行前 m 个完整 DiT blocks；重算步从 probe 输出继续执行剩余 blocks。

Wan2.2 可以沿用这两个机制，但必须重做双专家的状态隔离、阶段切换及视频生命周期管理。应称为本项目移植版本。本次核对的官方仓库没有 Wan2.2 实现。

## 来源及版本

- [DiCache 官方仓库](https://github.com/Bujiazi/DiCache)，`git ls-remote` 核对 HEAD/main=`fdbe20b669c9174bbed5ec994de073fd881c8010`；完整 git tree 已检查。
- [官方核心脚本](https://github.com/Bujiazi/DiCache/blob/fdbe20b669c9174bbed5ec994de073fd881c8010/WAN2.1/run_wan_dicache.py)：`dicache_forward` 位于69–223行，参数默认值在261–425行，patch/state初始化在543–557行。
- [官方采样链路](https://github.com/Bujiazi/DiCache/blob/fdbe20b669c9174bbed5ec994de073fd881c8010/WAN2.1/wan/text2video.py)：每个去噪步先cond、后uncond。
- [官方 blocks 实现](https://github.com/Bujiazi/DiCache/blob/fdbe20b669c9174bbed5ec994de073fd881c8010/WAN2.1/wan/modules/model.py)：核对缓存张量位置和dtype传播。
- [论文 arXiv v2](https://arxiv.org/html/2508.17356v2)：方法§3.2、实验§4.1/Table 1、配置Appendix B。本记录明确引用v2，未将其参数冒称为所有后续版本的统一设置。
- Wan2.2 基底采用项目已锁定的 [Wan-Video/Wan2.2@42bf4cf](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc)，与现有SeaCache/MagCache包的上游锁一致。

六个DiCache源码文件已通过固定commit URL重新下载，先前main下载的五个文件逐字节相同；SHA256见 `source_manifest.json`。本次未直接执行上游脚本。

## 缓存位置与前向流程

这里用 `h0` 表示patch embedding后的token序列、`hm`表示前m个blocks输出、`hL`表示全部L个blocks输出。它们不是VAE latent或最终noise prediction。

- 全网残差 `R = hL - h0`。
- 浅层残差 `q = hm - h0`。
- 复用时构造 `hL_hat = h0_current + R_hat`，随后仍执行当前step的head和unpatchify。
- patch/time/text embedding、浅层blocks、head仍然每步执行；T5、VAE与scheduler没有被DiCache跳过。

默认Wan2.1-1.3B有30个blocks，`probe_depth=1`表示运行第一个完整block，包括self-attention、cross-attention和FFN。默认复用步跳过余下29个blocks。

```text
latent → patch/time/text embedding → 前m个blocks → 得到hm
  ├─ 保留期或误差达到阈值：继续剩余blocks → 刷新R、q历史
  └─ 误差低于阈值：用浅层轨迹外推R → h0 + R_hat
两条路径随后均执行head → unpatchify → CFG → scheduler
```

## Online Probe：何时复用

每个CFG分支单独计算相邻去噪步的浅层输出变化：

`d_s = mean(abs(hm_s - hm_prev)) / mean(abs(hm_prev))`

`E_s = E_prev + d_s`

代码严格使用 `E_s < rel_l1_thresh` 复用；等于或超过阈值则执行剩余blocks并把E清零。论文伪代码写成`<=`，复刻代码时应以严格小于为准。

无论复用还是重算，`previous_internal_states`均更新为本步真实probe输出。因此gate比较的是相邻步的浅层输出；DCTA所用历史则来自最近两次完整重算。这两种历史不能混淆。

`delta_x`虽然计算了输入变化，但没有参与decision、累计量或DCTA。这里没有TeaCache的离线多项式、MagCache预计算ratio，也没有训练出来的策略网络。

## DCTA：如何使用历史

设a为较早、b为最近的两个完整重算时刻，保存对应的`R_a/R_b`、`q_a/q_b`。当前probe残差为`q_s=hm_s-h0_s`。

`gamma_s = clip(mean(abs(q_s-q_a)) / mean(abs(q_b-q_a)), 1, 2)`

`R_hat_s = R_a + gamma_s * (R_b-R_a)`

`hL_hat_s = h0_s + R_hat_s`

gamma是对整个tensor归约得到的单个标量，不是逐token权重。`gamma=1`退化为最近一次残差，`gamma=2`沿两次重算形成的残差方向再外推一个增量。只有一份重算历史时，直接复用最近残差。

两份历史只在完整重算时更新，复用产生的估计值不会写入重算窗口。这是浅层特征驱动的一阶外推，不按去噪step间隔显式缩放，也没有高阶Taylor状态。

## 官方默认参数与边界行为

| 项目 | 官方Wan2.1代码行为 |
|---|---|
| 任务 | `t2v-1.3B` |
| 视频 | `832×480`、81帧、16fps |
| 采样 | 50-step UniPC、shift=5、CFG=5 |
| 精度 | BF16 autocast；不代表所有中间张量或存储权重都是BF16 |
| seed | Python CLI默认0；官方shell示例显式3 |
| probe_depth | 初始化处硬编码1，不是CLI参数 |
| rel_l1_thresh | CLI默认0.08；论文v2主实验Wan阈值为0.2 |
| ret_ratio | 默认0.2，只保留开头一段完整计算 |
| CFG状态 | cond/uncond的累计误差、缓存、历史与decision独立 |
| 计数 | `num_steps=sample_steps*2`，`cnt`每次模型forward增加1；`cnt%2`标识分支 |
| 首段 | `cnt < int(num_steps*ret_ratio)`全部重算；50步默认对应20次forward、10个完整去噪步 |
| 尾段 | 没有最后一步强制重算、最大连续skip或Exact-K约束 |
| 单卡offload | CLI未指定时自动True；t5_cpu默认False |
| 视频结束 | 最后一次uncond结束后清空全部历史 |

改变ret_ratio时，代码按forward数取整，奇数保留调用数可能使cond/uncond保留步数不同；它不是对称的首尾retention。ret_ratio=0还会让首步在缓存尚空时进入probe比较，因此移植时必须显式处理未初始化缓存。

阈值增大通常允许更多复用，但它不直接表示目标speedup；闭环轨迹变化也使逐prompt严格单调性没有保证。不能把论文或Wan2.1默认阈值直接当作Wan2.2某档速度的已校准参数。

## 源码中需要保留或明确说明的细节

1. **类级patch和可变状态。** `wan_t2v.model.__class__.forward`及缓存list写到类上。单实例示例可以工作，多个pipeline或Wan2.2双专家直接共用会有状态共享风险；实例计数还不能可靠表示整段视频进度。
2. **窗口实际长度是3。** `len(window)<=2`时仍append，因此第三份加入后保留3份；以后只移动末两项，第一项一直留存。计算始终只用`[-2:]`，多出的一份不参与DCTA。
3. **无效状态。** `previous_output`写入后没有读取；`previous_input`只服务于没有被使用的delta_x。可做不改变决策公式的清理，但应记录为工程差异。
4. **除数没有epsilon/finite保护。** probe relative-L1及gamma都直接做除法。gamma clip并不能处理0/0的NaN；原版并未提供异常输入fallback。采用保护逻辑须披露，而不是声称与原版对所有输入逐位相同。
5. **复用路径使用原地加法。** DCTA分支是`x += ...`，x来自patch embedding；BF16 autocast下该操作会将FP32外推结果写回BF16 x。full路径经过FP32 modulation则返回FP32 hidden states，单历史fallback的非原地加法也可能提升dtype。改成`x = x + ...`可能改变数值，移植差分验证必须覆盖此点。
6. **resume不应重复probe。** 已算m个blocks后仅运行后L−m个blocks；若从block0重新跑会人为增加重算成本。原实现记录probe state的循环索引还隐含浅probe假定，开放任意深度时应重新检查。

## 显存与计量影响

静态张量核算：官方实现每个CFG分支有3份deep residual、3份probe residual、previous internal/output两份FP32序列，加一份previous input BF16序列；`residual_cache`和`probe_residual_cache`引用窗口最新项，不重复计数。两分支合计16份FP32与2份BF16。

| 模型与协议 | 序列shape | 单份FP32 | 上述历史的合计常驻量 |
|---|---|---:|---:|
| Wan2.1-1.3B，81帧 | `[1,32760,1536]` | 191.953MiB | 3.187GiB |
| Wan2.2-A14B，45帧，每个专家 | `[1,18720,5120]` | 365.625MiB | 6.070GiB |

第二行是假设按原版数量及dtype逐项保留的静态估算；未计模型权重、临时张量、workspace或allocator。若高专家历史不释放、低专家再独立建满，历史可能接近两倍。普通Python list中的缓存不会自动随model.cpu()迁移；因此必须管理阶段结束时的释放。可去除第三份窗口和无效状态减少占用，具体实现尚未冻结。

若某专家某分支共T次调用，其中K次full、L层、probe深度m，则blocks执行次数为`K*L+(T-K)*m`。以m=1/L=40为例，复用一次仍有一个完整block，不能把它记作零DiT计算量。实际TFLOPs还应加入每次调用的embedding/head，再按专家及分支汇总。

项目正式latency应采用完整generate墙钟，包含probe、DCTA、权重搬运和其余推理工作；分别记录T5/DiT/VAE用时与TFLOPs。若Calflops不统计DCTA归约/逐元素运算，必须披露排除项。PSNR/SSIM/LPIPS沿用VideoMetrics，另计算VBench score。

## 论文结果及其用途

论文v2 §4.1/Appendix B给出Wan2.1-1.3B、832×480、81帧、50步、默认单A800 80GB、300个视频prompts；主实验阈值0.2、probe深度1。Table 1作者报告baseline 192.47s，DiCache 78.42s/2.45×，PSNR26.45、SSIM0.8885、LPIPS0.1734。

这些是作者协议下的公开结果，不是本地复现值。官方Wan目录没有300-prompt批处理、逐视频质量评测、VBench或组件计量流水线，无法仅凭当前quick-start确认其论文latency包含T5/VAE/offload的精确边界。复现实验仍需要本项目的统一计量链路。

## Wan2.2适配建议（尚未实施）

1. 基于与SeaCache/MagCache相同的Wan2.2@42bf4cf前向主干接入DiCache；关闭缓存时直通未修改的原生full路径，以便核对共同baseline。
2. 每个视频创建独立controller，状态按`(expert, CFG branch)`隔离。cond/uncond继续独立gate；不能直接改成现有SeaCache的共享decision而仍声称原版行为。
3. 全局去噪step、专家内step及CFG branch由sampler显式提供；在generate开始/结束与异常清理处reset，不沿用`self.cnt>=100`作为唯一清理条件。
4. 高低专家切换时禁止跨权重复用，清空/释放退出专家的cache；新专家首步必须full。只有一份新专家历史时走官方latest-residual fallback，之后DCTA使用该专家自己的两次full历史。
5. ret_ratio的全局/分阶段口径需要在移植配置中显式定义。较小改动方案是保留全局初始20% full，再加新专家首次调用full；这属于建议，尚未冻结，不能直接宣称官方规定了双阶段warmup。
6. 显式保留probe_depth、threshold、ret_ratio、gamma clip及dtype语义。先验证忠实实现，再把节省无效缓存、除零保护等单独记入适配说明；目标speedup需在Wan2.2同协议下重新标定。
7. 增加逐专家/CFG trace及full/probe block统计，接入现有组件计量、持久batch和质量链路。用户本轮要求先调研汇报，此处没有启动实现或实验。

Wan2.2正式协议继续为根规范的45帧、50-step DPM++、shift12、CFG(low/high)=(3,4)、boundary0.875、seed42、BF16、单卡offload_model=True、t5_cpu=False、无FSDP/SP及prompt extension；不采用上游A14B配置文件中的默认40步覆盖本项目50步约定。
