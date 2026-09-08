# TeaCache4Wan22 formal VBench200 benchmark

本目录是 TeaCache 正式 `.29/.45/.68` 三档测试管线。固定使用
Wan2.2-T2V-A14B、VBench200 的 200 条 prompt、`832x480`、45 帧、16 fps、
50-step DPM++、shift 12、CFG `(3,4)`、boundary `.875`、BF16、seed 42、
`offload_model=True`、`t5_cpu=False` 和 `use_ret_steps=False`。

## 关键合同

- baseline 必须通过 `--baseline-source` 直接复用 SeaCache 的 no-cache
  VBench200 目录；正式 runner 没有 baseline 生成路径。
- 复用前逐项检查 200 个视频/计时 ID、四份 shard 配置、prompt hash、checkpoint、
  生成协议、Wan2.2 commit，以及每条 timing 的 100 次 DiT 调用均执行全部 blocks。
- 每个 threshold 在四张卡上各启动一个 persistent batch worker；每个 worker 只加载
  一次 WanT2V 并顺序处理 50 条 prompt。GPU 0/1 与 2/3 分成两波，第二波只在第一波
  完成一次性模型初始化并达到显存门限后启动。
- 三档候选全部生成及性能汇总后，launcher 才自动执行 PSNR、SSIM、LPIPS 和
  16 维 VBench200 subset evaluation。baseline VBench 只计算一次，其余两档复用。
- Calflops profile 只计算一次并由后两档复用；Wan VAE tuple `scale_factor` 使用
  `ComponentMetrics` 中锁定的 `calflops_0.3.2_tuple_upsample_v1` 兼容修复。

## 启动

先准备锁定的源树：

```bash
WAN22_SETUP_PYTHON=/home/huteng/yes/envs/wan2.2/bin/python \
  bash ../../scripts/prepare_wan22.sh \
  ../../build/Wan2.2-42bf4cf-prepared
```

正式运行只需要给出位于外部实验根目录下的新 suite 目录：

```bash
bash launch_threshold_suite_029_045_068.sh \
  /all/yiran07-disk3/huteng_data/exp/<suite-name>
```

launcher 会在创建结果目录或启动 GPU worker 前完成 fail-closed preflight：prepared
source、共享 baseline、Calflops、VideoMetrics、Detectron2 C extension、全部 16 个
VBench dimension module、数据配置和所有 VBench checkpoint 的 SHA256 均需通过。

可通过 `WAN22_PYTHON`、`WAN22_SOURCE`、`WAN22_CKPT`、
`WAN22_BASELINE_SOURCE`、`TEACACHE_COEFFICIENTS`、`VBENCH_SOURCE`、
`VBENCH_EXTRA_DEPS`、`VBENCH_CACHE_DIR` 和 `VIDEO_METRICS_CACHE_DIR` 覆盖默认路径。
所有 Python/BLAS 进程均固定四个线程变量为 1。

## 输出

suite 下每个 threshold 目录包含：

- `baseline`：指向同一个 SeaCache no-cache baseline 的目录 symlink；
- `teacache/{videos,timings,traces,worker_status}`：候选视频及逐样本证据；
- `performance/`：T5/DiT/VAE 分项时间、TFLOPs 和 ratio-of-sums speedup；
- `evaluation/video_metrics/`：逐样本与汇总 PSNR/SSIM/LPIPS；
- `evaluation/vbench_*`：baseline/candidate VBench200 subset score；
- `benchmark_report.{json,csv,md}`：单档完整汇报。

顶层 `suite_report.{json,csv,md}` 汇总三档，并再次断言三档 baseline symlink 指向
同一个实体目录且 baseline VBench 分数完全一致。中断后重跑同一 launcher 会使用
锁定配置 resume；已完成的候选会被完整验证并跳过，不会为了进入评测阶段重新加载模型。
