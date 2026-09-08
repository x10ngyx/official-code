# MagCache4Wan22 Vbench200 batch

采用 SeaCache4Wan22 的持久 worker 规范。`run_vbench200.py` 是 controller，
`generate_vbench200.py` 是 worker，`run_vbench200.sh` 设置四个线程变量。
默认 200 条 Vbench200 prompt，每个 prompt 生成一个原生 baseline 和一个 MagCache 视频。
每个 worker 只构建一次 WanT2V，按 prompt round-robin 分片；同一 prompt 两侧固定同卡。
每个视频重新安装计时器并重置官方 MagCache 状态。多卡仅并行独立视频，单个视频不使用 FSDP/SP。

从项目根目录运行：

```bash
conda activate wan2.2
bash experiments/vbench200_t2v/run_vbench200.sh \
  --source "$PWD/build/Wan2.2-42bf4cf" \
  --checkpoint /home/huteng/xiongyuxiang/tmp/models/Wan2.2-T2V-A14B \
  --output-dir /all/yiran07-disk3/huteng_data/exp/magcache_wan22_vbench200 \
  --gpu-ids 0 1 --worker-launch-wave-size 1 \
  --threshold .06 --magcache-k 2 --retention-ratio .4 \
  --dry-run
```

去掉 `--dry-run` 才执行实验。指定的 GPU 必须已分配给本实验；启动波次只错开模型加载，
不会减少最终并发 worker 数。默认每进程一个完整 baseline warmup，不导出、不计入 headline；
`--warmup-videos 0` 可显式禁用。初始化时间单独记录一次，不能加入逐视频 latency。

- `--calflops-profile FILE`：复用严格匹配源码、权重路径与固定协议的组件 profile；省略则先在第一张指定卡进行 real-shape profiling。
- `--resume`：要求原 plan 的源码 SHA、prompt、参数、GPU 分片和 warmup 一致。完成视频验证 SHA、100-call 计时/trace 和 lifecycle 后跳过；未完成目录归档到 `incomplete_attempts/` 再生成。损坏的完成记录报错。
- `--defer-evaluation`：完成生成、组件性能汇总后保留 `quality_evaluation` pending，不写整个实验的 COMPLETE。之后原命令加 `--resume` 并移除此选项，直接继续评测。
- `--prompts FILE --vbench-mode standard`：允许固定 Vbench200 子集，必须覆盖全部 16 维；自定义输入必须显式选择 `custom`，报告单独标注十维 custom score。

结果目录由 README 描述：`runs/<condition>/<sample>/` 包含 MP4、run、timing、trace 和 SHA manifest；
`worker_status/` 包含 PID、GPU、初始化次数和 jsonl 索引；`videos/` 是软链接集合；
`performance.json` 汇总完整 generate latency 与 T5/DiT/VAE 秒数和 TFLOPs；
`evaluation/` 存储 VideoMetrics RGB PSNR/SSIM/LPIPS 和 VBench；`report.json` 汇总两者。
质量阶段可独立恢复，baseline 的 VBench 只计算一次。全部阶段成功才写 `COMPLETE.json`。
项目 `experiment_results/` 自动建立外部结果软链接。

本入口从当前原生主干生成 fresh baseline，不直接导入其他方法历史视频。
源码主干一致性不等于已经验证历史视频逐位一致；运行环境、权重、prompt 和协议也须匹配。
