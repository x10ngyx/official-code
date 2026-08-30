# Official Code

Evaluation resources and reproducible utilities for diffusion model caching
research. The local workspace directory and public repository are both named
`official-code`.

## 目录结构

- `TeaCache4Wan22/`：面向 Wan2.2-T2V-A14B 的 TeaCache 完整复现包，含锁定上游的集成补丁、runtime、固定协议、70-prompt 系数标定、组件 CUDA 计时、Calflops/真实 cache path 计算量聚合、运行/验证脚本与测试。
- `TeaCache4Wan21/`：以锁定的原始 Wan2.1 为统一入口，显式可选注入官方 TeaCache
  方法，并使用本仓库 VideoMetrics 与 VbenchEvaluation 工具完成质量评测。
- `SeaCache4Wan22/`：面向锁定 Wan2.2-T2V-A14B 的干净 SeaCache 包；仅含 stage-aware timestep residual cache、SEA filter、共享 CFG gate、最小集成 patch、固定协议 runner、trace、推理计时与 CPU tests。
- `SeaCache4Wan21/`：以锁定原始 Wan2.1 为 baseline，显式可选注入 corrected filtered-boundary SeaCache T2V 实现；cond/uncond 独立 gate、独立 previous/accumulator/residual，强制步也保存 SEA-filtered feature，不含其他 cache 方法。该修正版不冒充官方 raw-boundary 的逐行等价实现。
- `Ours4Wan21/`：Wan2.1-T2V-1.3B learned cache controller 项目；当前包含 OpenVid3000×3 随机 threshold 数据采集、TeaCache 风格 inference timing、trace-weighted Calflops TFLOPs、发布与审计子项目。
- `Vbench200/`：从 VBench 944 个唯一 prompt 中固定随机抽取的 200 条开源测试集，含复现脚本与来源校验信息。
- `VbenchEvaluation/`：Vbench200 的官方 VBench 16维评测适配、版本锁定、权重缓存约定与分数聚合工具。
- `VideoMetrics/`：统一视频 RGB PSNR、SSIM、LPIPS 全参考评测包、命令行入口与回归测试；公式来源单独记录在包内上游锁和致谢中。
- `CalflopsEvaluation/`：基于 Calflops 的 forward-only FLOPs/TFLOPs 计数、未覆盖算子补偿与 cache trace 聚合工具；区分每视频总 TFLOPs 和吞吐率 TFLOP/s。
- `ComponentMetrics/`：五条正式方法链路共用的 T5/VAE 计时、T5/VAE Calflops profile、固定协议验证与 strict schema 提取器。
TeaCache Wan2.1 原始 baseline、显式可选的官方方法注入、复现校验和评测编排已加入，
历史上完成过 GPU 一致性、latency 与 Calflops TFLOPs 冒烟实验，但旧产物不含当前
强制的全部组件计量和 VBench，不能作为新 schema 的合规结果；
TeaCache Wan2.2 推理集成、经 70-prompt 标定终验的正式 high/low 系数，以及
Wan2.1 同款 pipeline/DiT timing 与 Calflops 性能计量链路也已加入。
SeaCache Wan2.1/Wan2.2 的干净发布包也已加入；两者都只暴露 no-cache/SeaCache，
明确排除历史实验树中的 block cache、CFG cache、ZEUS、TeaCache 与 learned policy。
`Vbench200/`、`VbenchEvaluation/`、`VideoMetrics/` 与 `CalflopsEvaluation/` 均已加入。

Model weights, generated videos, experiment results, local progress files,
and internal session logs are intentionally excluded from version control.
Each tool documents how to obtain its external resources and choose an output
directory.

No repository-wide license has been selected yet. Component-level upstream
attribution and version locks are recorded in their respective `NOTICE.md` and
`upstream_lock.json` files.

## 固定模型推理协议

跨缓存方法比较统一使用以下模型侧设置；方法自身的 threshold/cache schedule 作为
独立参数记录，不改变这里的 baseline 生成协议。

| 模型 | 画面大小与帧数 | 生成设置 | 精度与显存设置 |
| --- | --- | --- | --- |
| Wan2.2-14B | 832×480、45 帧、16 fps | 50 步 DPM++、shift=12、seed 42、双阶段 CFG=(3,4)、boundary=.875 | DiT BF16；batch 1、单卡、`offload_model=True`、`t5_cpu=False`，关闭 FSDP/SP 与 prompt rewrite/extension |
| Wan2.1-1.3B | 832×480、81 帧、16 fps | 50 步 UniPC、shift=5、CFG=5、seed 42 | DiT BF16；batch 1、单张 48GB GPU，model、T5 与其他组件全部常驻，`offload_model=False`、`t5_cpu=False`，关闭分布式与 prompt rewrite/extension |
| HunyuanVideo-13B | 640×480、65 帧、24 fps | 50 步 Euler、flow-reverse、shift=7、CFG=1、embedded-CFG=6、seed 42 | DiT BF16、文本编码器 FP16；CPU offload、VAE tiling、FlashAttention |
