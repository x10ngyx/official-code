# 官方 MagCache 参数组合扫描

`run_scan.py` / `run_scan.sh` 使用与 Vbench200 batch 相同的持久 worker。
`prompts.jsonl` 复用 TeaCache 标定实验的 11 条固定 Vbench200 prompt，覆盖全部 16 维；
此结果是小集合标定，不是全量 Vbench200 成绩。

扫描维度为官方 `magcache_thresh`、`K`、`retention_ratio`，不改算法、ratio 表或采样器。
默认 threshold=`.04,.06,.1,.2,.4,.8`，K=`2,3,4,6`，retention=`.4,.2,.1,.05`，
共 96 个组合。直接执行原始 gate AST 规划 100 个 cond/uncond 决策：
当前去重后 91 条 schedule，11 个共享 baseline + 91×11 candidates = 1012 个测量视频。
计划完整保留被合并的参数别名，可用 `--no-deduplicate` 分别实测所有组合。
`planned_full_calls` 仅为计算路径预览，不能作为实测加速比。

从项目根目录预览：

```bash
conda activate wan2.2
bash experiments/parameter_scan/run_scan.sh \
  --source "$PWD/build/Wan2.2-42bf4cf" \
  --checkpoint /home/huteng/xiongyuxiang/tmp/models/Wan2.2-T2V-A14B \
  --output-dir /all/yiran07-disk3/huteng_data/exp/magcache_wan22_parameter_scan \
  --gpu-ids 0 1 --worker-launch-wave-size 1 \
  --target-speedups 1.8 2.4 3.0 --target-tolerance .1 \
  --dry-run
```

去掉 `--dry-run` 才执行。可通过 `--thresholds .06 .2 .8 --ks 2 4 6 --retention-ratios .4 .1 .05`
缩小首轮网格；实际组合数与视频数以 dry-run 为准。每个 prompt 的 baseline 和全部候选在同一卡
同一持久 pipeline 中执行，每个视频按 seed42 重建噪声，缓存/计时状态逐视频重置。
每个新进程默认进行一次不计入指标的 full warmup。模型加载和 warmup 都单独记录。

`target_selection.json` 对每个目标使用同一 11 条 prompt 的
`sum(baseline.pipeline_generate_wall_seconds) / sum(candidate.pipeline_generate_wall_seconds)`，
寻找最近的**实测**组合。默认绝对误差不超过 0.1× 才标 `hit`，否则为 `unmet`；
不会把最近点或理论 FLOPs 比当成已经达标。每个候选保留 T5/DiT/VAE 时间和 TFLOPs。
该选择按速度距离排序，质量数据供进一步选择，不自动声称质量最优。

默认对全部测量组合计算 VideoMetrics RGB PSNR/SSIM/LPIPS 及标准 16 维 VBench score，
baseline 的评测共享。可用 `--defer-evaluation` 先完成速度标定，整个实验状态仍为质量评测待完成；
原命令加 `--resume` 并去掉该选项后继续评测。断点与损坏检测规则同
[`vbench200_t2v`](../vbench200_t2v/README.md)。输出结构同 batch，另有 `target_selection.json`；
所有视频、计量和报告位于外部 exp 根目录，通过 `experiment_results/` 软链接访问。

固定每档 R/K、在推荐起点附近细扫 E 的入口见
[`targeted_threshold_scan`](../targeted_threshold_scan/README.md)。
目前只验证脚本、官方决策和 dry-run，未运行完整 A14B 扫描，没有已标定参数。
