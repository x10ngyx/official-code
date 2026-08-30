# TeaCache4Wan21 experiment results

大型实验结果统一保存在 `/all/yiran07-disk3/huteng_data/exp`。实验实际启动后，只在本目录
创建指向对应外部结果目录的符号链接；不要复制视频、checkpoint、评测权重或运行日志。

当前目录没有正式结果链接。此前的 threshold-zero probe/smoke 数值产生于组件
timing/TFLOPs 与 VBench 纳入强制合同之前，只能作为历史诊断，不能标记为当前规范下
的正式合规结果。新启动器会用外部结果目录的 basename 创建软链接，并要求新 schema
同时包含 T5/DiT/VAE、统一 PSNR/SSIM/LPIPS 与 VBench score。
