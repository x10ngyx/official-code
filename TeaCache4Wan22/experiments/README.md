# Experiments

本目录用于保存 TeaCache4Wan22 的实验脚本。每项实验应放在独立子目录中，并在该子目录提供 `README.md`，说明配置、运行方式与输出位置。

实验产物不得直接写入本目录；应保存到仓库外部的独立实验结果目录。

当前实验：

- `fit_t2vcompbench70_wan22_t2v_a14b/`：使用 70 条分层抽样的 T2V-CompBench prompts，采集 full-compute `e/H/(H-Z)` 相邻 relative-L1，并为 Wan2.2 T2V-A14B 的 high/low 阶段分别拟合四次多项式。
- `smoke_t2v_a14b/`：运行单 prompt 的 matched no-cache/TeaCache 集成 smoke，并校验 trace、T5/DiT/VAE 分项时间与 TFLOPs、统一 PSNR/SSIM/LPIPS 及 custom-input VBench score。
- `performance_t2v_a14b/`：分别 profile high/low DiT 的 Calflops 成本，并按 Wan2.1 同款 timing trace 的实际 full/reuse 调用路径汇总 latency、TFLOPs 与两类 ratio-of-sums speedup。
- `threshold_scan_vbench8_t2v_a14b/`：目录名为兼容历史路径而保留；当前固定为 11 个覆盖全部 16 个维度的 Vbench200 prompts，在 `.15--.80` 上扫描，保存 T5/DiT/VAE 分项时间与 TFLOPs、统一 PSNR/SSIM/LPIPS 和 Vbench200 subset score，并定位 `1.8x/2.4x/3.0x` 最近实测 threshold。
