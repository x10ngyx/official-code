# Official Code

Evaluation resources and reproducible utilities for diffusion model caching
research. The local workspace directory retains the historical
spelling `offical-code`, while the public repository is named `official-code`.

## 目录结构

- `TeaCache4Wan22/`：面向 Wan2.2-14B 的 TeaCache 实现项目。
- `Vbench200/`：从 VBench 944 个唯一 prompt 中固定随机抽取的 200 条开源测试集，含复现脚本与来源校验信息。
- `VbenchEvaluation/`：Vbench200 的官方 VBench 16维评测适配、版本锁定、权重缓存约定与分数聚合工具。
- `VideoMetrics/`：统一视频 RGB PSNR、SSIM、LPIPS 全参考评测包、命令行入口与回归测试；公式来源单独记录在包内上游锁和致谢中。
- `CalflopsEvaluation/`：基于 Calflops 的 forward-only FLOPs/TFLOPs 计数、未覆盖算子补偿与 cache trace 聚合工具；区分每视频总 TFLOPs 和吞吐率 TFLOP/s。
TeaCache Wan2.2 推理实现尚未开始；`Vbench200/`、`VbenchEvaluation/`、`VideoMetrics/` 与 `CalflopsEvaluation/` 已加入。

Model weights, generated videos, experiment results, local progress files,
and internal session logs are intentionally excluded from version control.
Each tool documents how to obtain its external resources and choose an output
directory.

No repository-wide license has been selected yet. Component-level upstream
attribution and version locks are recorded in their respective `NOTICE.md` and
`upstream_lock.json` files.
