# threshold=0 冒烟测试

这个实验用同一份锁定的 Wan2.1-T2V-1.3B 源码与权重，运行一条 no-cache
baseline 和一条显式 TeaCache `threshold=0` 诊断样本，验证 Wan2.2 conda 环境可
用于 Wan2.1 推理，并检查两个 MP4 的文件字节、逐帧 RGB 像素、本项目
PSNR/SSIM、推理 latency 和 DiT TFLOPs。零阈值按项目测试契约使用内部 `-inf`
比较 sentinel 禁用残差复用，但
仍实际执行官方 TeaCache 采样函数和 `forward`；所有正阈值路径仍使用官方实现。

固定配置为 `832x480`、5 帧、4 个 UniPC 采样步、seed 42。测试覆盖统一入口的
TeaCache 显式开关、官方函数注入和零值 full-compute 分派，并要求结果与原始
Wan2.1 路径字节一致。

目录文件：

- `run_smoke.sh`：检查输入、依次生成两个视频并运行比较与自有指标。
- `capture_environment.py`：记录 Python、PyTorch、CUDA、GPU、源码与命令配置。
- `compare_outputs.py`：比较 MP4 SHA256 和严格逐帧 RGB 像素。
- `run_timed_command.py`：用单调时钟记录包含加载、推理和导出的进程端到端 latency。
- `profile_wan21_dit.py`：用 Calflops 统计 DiT forward，并解析补偿 FlashAttention core。

## 性能口径

每条生成命令会在 `latency/` 输出三层 latency：pipeline 初始化（含权重加载）、
pipeline generate（T5 编码、denoising、VAE decode，不含 MP4 导出）以及 CUDA event
统计的 DiT forward 总时间；进程端到端时间另含导入和 MP4 导出。所有 CUDA 时间在
读取前同步，单次 model forward 不做强制同步，避免给每个 step 引入额外停顿。

`flops/calflops_profile.json` 同时保存：

- Calflops 0.3.2 自动统计的 Linear/Conv/可见算子；
- 对不可见 FlashAttention CUDA kernel 的 dense attention 理论补偿；
- 每次 DiT forward 和按实际 full/reuse call trace 聚合的每视频 TFLOPs；
- 用每视频 TFLOPs 除以 CUDA DiT 时间得到的估算 TFLOP/s。

这里 `TFLOPs` 表示 `10^12` 次运算的数量，`TFLOP/s` 才是吞吐率。计数范围是
DiT forward-only，不包含 T5、VAE、scheduler、导出和很小的 TeaCache 多项式控制器。
该 5 帧单次冷启动只用于冒烟验证；正式性能结论应增加 warm-up 和多次重复统计。

默认结果写入
`/mnt/hdd/xiongyuxiang/tmp/exp/teacache_wan21_threshold0_smoke`。脚本拒绝覆盖已有
结果；项目中的 `experiment_results/threshold_zero_smoke` 只应是指向该目录的
软链接。

首次运行前，在目标 Wan conda 环境中安装本仓库锁定的 Calflops 评测包：

```bash
/path/to/wan-env/bin/python -m pip install -e \
  /home/star/xiongyuxiang/tmp/work/TeaCache/CalflopsEvaluation
```

运行：

```bash
bash run_smoke.sh \
  /mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env \
  /mnt/hdd/xiongyuxiang/tmp/data/source/Wan2.1-65386b2 \
  /mnt/hdd/xiongyuxiang/tmp/models/Wan2.1-T2V-1.3B \
  /mnt/hdd/xiongyuxiang/tmp/exp/teacache_wan21_threshold0_smoke \
  0
```
