# TeaCache4Wan21 experiments

每个子目录是一套可独立复现的实验脚本与协议说明。生成视频、进程日志和评测产物
必须写到 `/mnt/hdd/xiongyuxiang/tmp/exp`，本目录只保存代码与小型配置。

- `vbench200_t2v/`：Wan2.1-T2V-1.3B full-compute 与官方 TeaCache 的
  Vbench200 同 prompt、同 seed 四卡对比；固定 832×480、81 帧、16 fps、50 步
  UniPC、shift 5、CFG 5、seed 42，并调用本仓库质量、VBench 与 Calflops 工具汇总
  PSNR/SSIM/LPIPS、Vbench200 score、latency 和 TFLOPs。
- `threshold_zero_smoke/`：Wan2.2 conda 环境兼容性与 TeaCache 零阈值严格一致性
  冒烟测试，比较 MP4 字节、逐帧 RGB 和自有 PSNR/SSIM。
- `threshold_calibration_vbench_subset/`：用 4 条跨类型 VBench200 prompt、50-step
  正式配置粗扫和细扫 threshold，以 pipeline generate 推理时间校准
  1.8×、2.4×、3.0× 三个目标档位。
