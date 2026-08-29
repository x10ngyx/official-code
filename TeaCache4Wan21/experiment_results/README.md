# TeaCache4Wan21 experiment results

大型实验结果统一保存在 `/mnt/hdd/xiongyuxiang/tmp/exp`。实验实际启动后，只在本目录
创建指向对应外部结果目录的符号链接；不要复制视频、checkpoint、评测权重或运行日志。

- `threshold_zero_smoke`：通过的正式零阈值冒烟测试；字节与逐帧像素完全一致，
  PSNR 100 dB、SSIM 1.0。
- `threshold_zero_raw_official_probe`：零阈值仍原样执行官方拟合判据时的失败诊断；
  用于记录拟合值为负导致残差复用的边界问题。
- `threshold_zero_pipeline_bypass_probe`：直接保留原始 pipeline 时的中间通过探针；
  已被最终“官方函数注入 + `-inf` full-compute”结果取代。
- `threshold_zero_profiled_smoke`：带精确进程/pipeline/DiT latency、逐 call block trace、
  Calflops 0.3.2 和 FlashAttention 补偿 TFLOPs 的完整冒烟结果。
