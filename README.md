# Official Code

Evaluation resources and reproducible utilities for diffusion model caching
research. The local workspace directory and public repository are both named
`official-code`.

## 目录结构

- `TeaCache4Wan22/`：面向 Wan2.2-14B 的 TeaCache 实现项目。
- `TeaCache4Wan21/`：以锁定的原始 Wan2.1 为统一入口，显式可选注入官方 TeaCache
  方法，并使用本仓库 VideoMetrics 与 VbenchEvaluation 工具完成质量评测。
- `Vbench200/`：从 VBench 944 个唯一 prompt 中固定随机抽取的 200 条开源测试集，含复现脚本与来源校验信息。
- `VbenchEvaluation/`：Vbench200 的官方 VBench 16维评测适配、版本锁定、权重缓存约定与分数聚合工具。
- `VideoMetrics/`：统一视频 RGB PSNR、SSIM、LPIPS 全参考评测包、命令行入口与回归测试；公式来源单独记录在包内上游锁和致谢中。
- `CalflopsEvaluation/`：基于 Calflops 的 forward-only FLOPs/TFLOPs 计数、未覆盖算子补偿与 cache trace 聚合工具；区分每视频总 TFLOPs 和吞吐率 TFLOP/s。
TeaCache Wan2.1 原始 baseline、显式可选的官方方法注入、复现校验和评测编排已加入，
并已完成 GPU 一致性、latency 与 Calflops TFLOPs 冒烟实验；
TeaCache Wan2.2 推理实现尚未开始。`Vbench200/`、`VbenchEvaluation/`、
`VideoMetrics/` 与 `CalflopsEvaluation/` 已加入。

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
| Wan2.2-14B | 832×480、45 帧、16 fps | 50 步 DPM++、shift=12、seed 42、双阶段 CFG=(3,4)、boundary=.875 | DiT BF16；允许模型卸载，T5 留在 GPU |
| Wan2.1-1.3B | 832×480、81 帧、16 fps | 50 步 UniPC、shift=5、CFG=5、seed 42 | DiT BF16；model 和 T5 可卸载到 CPU |
| HunyuanVideo-13B | 640×480、65 帧、24 fps | 50 步 Euler、flow-reverse、shift=7、CFG=1、embedded-CFG=6、seed 42 | DiT BF16、文本编码器 FP16；CPU offload、VAE tiling、FlashAttention |
