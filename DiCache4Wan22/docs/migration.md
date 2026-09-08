# 已实现的迁移边界

## 执行来源

`vendor/run_wan_dicache.py` 保留官方完整文件。`load_official()` 在核对其SHA后，逐字提取从 `skip_forward=False` 到head调用之前的原版AST；给函数添加返回值用于返回隐藏状态和观测信息，没有修改原算法statement。

`compose_forward()` 核对Wan2.2@42bf4cf的model.py SHA，从原生WanModel.forward提取AST，仅替换其full-block循环为上述运算段。这样保留Wan2.2自己的token-wise timestep输入与输出接口，不把Wan2.1的embedding形状移植过去。Baseline不进入该替换路径。

## 双专家状态

原版用单模型的100次CFG调用管理一次视频。本适配使用独立实例状态，记录全局调用index、专家内调用index和CFG分支；不共用high/low历史，CFG两分支也不共享gate。

retention采用全局调用数，与原版单模型相同。新专家第一次cond/uncond各必须重算：在这两个调用中向原版分支提供cnt=0/1，选择它本身的初始化路径，其余调用提供真实全局cnt。默认50步下保留step0–9，另在low首次step32重算。只保存一次low历史时，官方代码自动走single-residual fallback。没有添加low阶段百分比warmup，也没有强制最终step49重算。

高专家历史在原生 `_prepare_model_for_timestep` 权重切换之前释放；全部历史在最后一次uncond完成后、VAE解码前清空。作用域finally恢复forward、状态属性和准备接口；原版中全局类状态的作用范围被限制为一段视频。

## 明确保留的行为

严格`<` gate、逐CFG误差累计、probe真实计算、full路径从probe继续、gamma原公式与clip[1,2]、原地加法、full残差与DCTA窗口只在full时刷新、`len<=2`形成3份窗口、delta_x计算及previous_output保存、分母无epsilon，均保留。`previous_input`和`previous_internal_states`在reuse时也会更新，用于下一次gate的相邻调用比较。没有删减无效缓存或加入新的稳定性修正。参数入口只拒绝非有限/越界值及无法初始化两CFG缓存的retention。

trace为观测信息，包含真实block计数、累计量前后值、probe差异、gamma、窗口长度、dtype和专家初始化标志。NaN/Inf诊断以字符串记录以保持JSON有效；模型数值不被改写。最终视频非有限时实验失败，不发布成功manifest。

## 计量

Wan2.2每专家40层；full执行40层，reuse执行1层，不记录为零DiT工作。分别profile原生full、前1block+head、head-only路径并补偿对应层数的dense FlashAttention FLOPs；逐调用按full/probe选择成本。Gate/DCTA和残差加法属于披露的FLOPs排除项，完整generate latency仍包含这些运算和观测开销。

默认阈值.08/retention.2来自官方Wan2.1 CLI，不代表在Wan2.2上已标定速度。模型设置采用项目冻结的45帧、50步DPM++、shift12、CFG(3,4)、boundary.875、seed42和单卡offload；不沿用官方Wan2.1示例的视频、采样或seed默认值。
