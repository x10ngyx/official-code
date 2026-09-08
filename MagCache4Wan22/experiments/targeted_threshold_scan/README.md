# 三档固定 R/K 的 E 标定

`presets.json` 保存三档起点、E 网格与容差；`run_scan.py` / `run_scan.sh` 调用公共持久 batch。
不改变官方 MagCache forward、ratio 表、baseline 或固定 Wan2.2 推理协议。

| 目标 | R | K | E 起点 | E 扫描（含端点） | 步长 |
|---|---:|---:|---:|---|---:|
| 1.8× | .2 | 2 | .075 | .060–.090 | .001 |
| 2.4× | .2 | 4 | .386 | .180–.420 | .001 |
| 3.0× | .1 | 5 | .200 | .180–.220 | .001 |

**这些是待实测起点，不是官方公布的 Wan2.2 三档配置，也不是已标定结果。**
R/K 参考公开 MagCache 工作点；E 根据官方 gate 的本地 50-step/双阶段 100-call 决策，
结合现有 SeaCache4Wan22 的 A14B 分项计时估计。三个起点分别 full/reuse 为
52/48、37/63、28/72；该耗时代理估计约 1.799/2.398/2.996×。
SeaCache 的缓存开销与 MagCache 不同，历史 baseline 的进程生命周期也不同，因此不能把这些估计作为达标证据。
来源、公式与限制见验证结果
[`recommendations.md`](../../experiment_results/magcache4wan22_targeted_scan_plan_20260905/recommendations.md)。

E 对应离散决策，不保证任意精确速度都存在。例如固定 R=.2/K=4，E=.385/.386/.387
对应 38/37/36 次 full；`.01` 粗网格可能漏掉中间点。实际执行直接使用原始 gate AST，
按完整 100-call 序列去重，保留全部参数别名；相同 full 次数但不同位置不会合并。
每档优先保留推荐 E 为该序列的实测代表。默认 313 个 E 点合并为 14 条序列（三档分别 3/6/5 条），
11 个共享 baseline + 14×11 candidates = 165 个测量视频；warmup、profile 与质量评测另计。
实际测量规模以 dry-run 为准。

在项目根目录预览：

```bash
conda activate wan2.2
export WAN22_PYTHON="$CONDA_PREFIX/bin/python"
bash experiments/targeted_threshold_scan/run_scan.sh \
  --source "$PWD/build/Wan2.2-42bf4cf" \
  --checkpoint /home/huteng/xiongyuxiang/tmp/models/Wan2.2-T2V-A14B \
  --output-dir /all/yiran07-disk3/huteng_data/exp/magcache_wan22_targeted_e_scan \
  --gpu-ids 2 3 --worker-launch-wave-size 1 \
  --dry-run
```

GPU 参数只是示例，执行前按实际空闲资源指定。去掉 `--dry-run` 才加载模型并生成视频。
默认使用 `../parameter_scan/prompts.jsonl` 的 11 条 prompt、同卡共享 baseline、每 worker 一次加载，
一次不计入 headline 的 full warmup。每个候选都记录完整 generate latency、T5/DiT/VAE 时间和 TFLOPs，
默认完成 RGB PSNR/SSIM/LPIPS 与标准 16 维 VBench subset score。
结果写入外部 exp 根目录，并建立本项目 `experiment_results/` symlink。

`target_selection.json` 在**该目标自己的 R/K 组内**，按相同 prompt 的
`sum(baseline.generate_wall_seconds) / sum(candidate.generate_wall_seconds)` 选最近实测 E。
默认容差为目标的 ±2%：1.8×的 [1.764,1.836]、2.4×的 [2.352,2.448]、3.0×的 [2.940,3.060]；
最近点超出范围即 `unmet`，不跨组修改 R/K，不把理论 full 比例或计时噪声当成精确达标。
容差是本脚本的初始约定，不是用户已确认的最终验收线。

- `--target-speedups 2.4`：仅标定指定档，可指定多个预设档。
- `--target-tolerance .03`：改用统一的绝对 ±.03×，覆盖预设相对容差。
- `--target-presets /path/to/custom.json`：替换整份预设；同一次运行禁止混用手工 R/K/E 网格参数。
  为保留源码锁，修改扫描范围时复制 JSON 到新的实验配置目录，再传入路径；不要改写已运行实验的配置。
- `--no-deduplicate`：每个 E 都单独计时；通常无需这样重复同一计算路径。
- `--defer-evaluation`：先标定速度，保留质量 pending；之后原命令加 `--resume` 并去掉该项完成评测。

计时波动与离散决策意味着精确到 1.800/2.400/3.000×不一定可达。若小集合找到命中点，
只能称该集合上的标定值；扩大到正式评测集后须重新报告实测加速比与质量。
## 固定 GPU1/2/3 分档运行

`run_gpu123.py --output-dir /all/yiran07-disk3/huteng_data/exp/UNIQUE_SUITE` 固定
GPU1→1.8×、GPU2→2.4×、GPU3→3.0×。先在GPU1完成一次共享的真实形状组件FLOPs profile，
再错开各卡模型加载；被其他任务占用的卡自动等待，其他空闲卡继续启动。
每卡生成自己的11个baseline，三档分别44/77/66个测量视频，总187个，另有三个full warmup。
各档自动完成速度选择及质量评测；顶层COMPLETE要求三个子实验全部完成。
源码、配置与profile锁定，断点使用原命令加`--resume`。
