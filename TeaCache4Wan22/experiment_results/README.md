# Experiment Results

本目录只保存本地实验结果索引。实际实验产物应位于仓库外部；可以在此创建指向结果目录的符号链接，避免复制大体积文件。除本 README 外，目录内容不会提交。

当前结果索引：

- `t2vcompbench70_fit_20260829_133525`：70/70 条 T2V-CompBench
  full-compute 标定、high/low 四次拟合、来源等价性与独立交叉验证审计；
  `FIT_AUDIT_REPORT.json` 状态为 `pass/share_with_caveats`。
- `teacache4wan22_fixed_protocol_smoke_20260829_182100`：同 seed/prompt 的
  baseline 与 threshold `0.10` TeaCache 集成 smoke；trace 为 3/50 reuse，
  inference-only speedup `0.994812x`，RGB PSNR `30.951523 dB`。该结果只验证
  端到端链路，不是推荐阈值或正式性能结论。
- `teacache4wan22_vbench200_batch_preflight_thr029_20260904_123958`：正式
  VBench200 batch 管线的 threshold `.29` 单样本 GPU 验收，覆盖 persistent
  model lifecycle、50-step TeaCache trace、T5/DiT/VAE timing、真实 Calflops、
  性能聚合及 PSNR/SSIM/LPIPS；不是 200-prompt 正式结果。
- `teacache4wan22_vbench200_thr029_045_068_fullwall_staggered_gpu0123_20260904_133701`：
  正在运行的 `.29/.45/.68` 正式 VBench200 套件；四卡 persistent batch worker
  分两波错峰，直接复用 SeaCache baseline，生成结束后自动运行全部质量与性能评测。
