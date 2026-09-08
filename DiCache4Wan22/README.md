# DiCache4Wan22

DiCache 官方 Wan2.1 方法在 Wan2.2-T2V-A14B 上的必要迁移。缓存算法直接来自锁定的官方源码，运行、计量、批处理和质量评测规范与本仓库其它对比方法一致。

## 官方算法保持不变

来源：[Bujiazi/DiCache@fdbe20b](https://github.com/Bujiazi/DiCache/blob/fdbe20b669c9174bbed5ec994de073fd881c8010/WAN2.1/run_wan_dicache.py)。`vendor/run_wan_dicache.py` 与上游逐字节一致。`runtime/official.py` 编译原文件第148–207行的 AST，不改写其中的算子、判断、历史更新或舍入方式。

- 每步执行前 **1个完整 DiT block**，按相邻浅层输出的 relative-L1 累积误差。
- cond/uncond 独立 gate、累计量、残差和历史；严格小于 threshold 才复用。
- 缓存为全部 blocks 输出减去 blocks 输入的残差；DCTA 用最近两次 full 历史及当前浅层残差计算标量 gamma，并 clip 到 `[1,2]`。
- 一份历史时直接复用；保留官方实际三份窗口、只使用末两份的更新行为。
- 保留 `delta_x`、`previous_output`、无 epsilon 的除法、`x += ...` 原地加法及其 BF16 舍入；没有增加新的误差估计器或保护性重算策略。
- 默认 `threshold=.08`、`retention_ratio=.2`、`probe_depth=1`，对应官方代码默认值；论文 v2 的 Wan 实验 threshold=.2 单独作为可选参数，不替代默认值。
- 原版前段 retention 的 `int(total_CFG_calls * ratio)` 和分支取整保持不变；不增加最后一步 full、最大 skip、Exact-K 或 CFG 同步决策。

阈值必须有限且非负；retention 在 `(0,1]` 内，且其取整结果至少保护初始两个 CFG 调用。固定50步时为 `[.02,1]`，拒绝原版会使用空缓存的设置。保留官方数值计算；若最终视频出现非有限值则该运行失败，不以有效视频发布。

## Wan2.2 的必要迁移

1. **接口：** 原生 Wan2.2 token-wise timestep embedding、block kwargs、head 和 unpatchify 保持原样，仅把 full-block 循环替换为官方 DiCache 运算段。源文件不打补丁，关闭缓存时仍直接使用原生 forward。
2. **专家隔离：** 状态属于每视频、每专家和每CFG分支，不放在模型类的共享可变属性中。
3. **阶段初始化：** retention 仍按全局100次CFG调用计算。低专家首次 cond/uncond 必须各 full 一次，通过官方初始化分支建立自己的历史；随后立即回到全局计数，不额外添加“低阶段前20%”保留期。仅一份历史时走原版回退，之后再用DCTA。
4. **生命周期：** 高专家缓存于权重切换前释放，低专家缓存于最后一次uncond结束后、VAE decode前释放。成功或异常均恢复模型forward与采样器准备接口。每个新视频重新初始化。

在默认50步协议下，全局step 0–9保留full；step32为低专家首次调用，需要full。其余调用完全由原版gate决定。迁移说明和历史调研分别见 [docs/migration.md](docs/migration.md) 和 [docs/wan21_official_research.md](docs/wan21_official_research.md)。

## 固定推理协议

conda `wan2.2`；Wan2.2-T2V-A14B、batch1、832×480、45帧、16fps、50-step DPM++、shift12、low/high CFG=(3,4)、boundary=.875、seed42、DiT BF16、单卡 `offload_model=True,t5_cpu=False`，关闭 FSDP/SP 和 prompt extension。

`baseline` 直接走原生模型；`dicache --threshold 0` 仍执行官方 probe/cache 记账，因此两者的执行开销不同。原生/all-full 输出等价性有独立测试。

## 目录

- `vendor/`：原版源码与上游README；`upstream_lock.json`：上游和包文件SHA。
- `runtime/`：官方算法接入、状态管理、计量、持久batch和参数扫描。
- `scripts/`：源码准备、审计、校验和单视频入口。
- `configs/`：固定模型协议。
- `experiments/performance_t2v_a14b/`：真实形状full/probe/head-only及T5/VAE Calflops和配对汇总。
- `experiments/vbench200_t2v/`、`paired_benchmark/`：持久配对生成和统一质量链路。
- `experiments/parameter_scan/`：threshold/retention的实测速度标定。
- `experiments/targeted_threshold_scan/`：固定官方retention=.2，自动粗扫/细化1.8×、2.4×、3.0×并评测所选配置。
- `experiments/smoke_official_core/`：小尺寸随机权重CUDA/BF16检查。
- `tests/`：CPU官方一致性、计量、恢复与批处理测试。
- `experiment_results/`：指向外部结果盘的symlink；`docs/`、`PROGRESS.md`、`logs/`：说明与交接。

## 使用

以下命令从本目录执行；无需修改共享环境。

```bash
conda activate wan2.2
export WAN22_PYTHON="$CONDA_PREFIX/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
bash scripts/prepare_wan22.sh "$PWD/build/Wan2.2-42bf4cf"

bash scripts/run_t2v_a14b.sh \
  --source "$PWD/build/Wan2.2-42bf4cf" \
  --checkpoint /path/to/workspace/models/Wan2.2-T2V-A14B \
  --output-dir /all/yiran07-disk3/huteng_data/exp/dicache_wan22_example \
  --prompt 'A man is talking to a woman in the office room.' \
  --mode dicache --threshold .08 --retention-ratio .2
```

`--mode baseline`生成原生基准；`--dry-run`检查源文件、模型注册路径与参数并输出计划，不加载权重或创建结果目录。已有源码或结果目录拒绝覆盖。`WAN22_REPOSITORY`支持使用本地Git克隆准备固定上游。

配对Vbench200：

```bash
bash experiments/vbench200_t2v/run_vbench200.sh \
  --source "$PWD/build/Wan2.2-42bf4cf" \
  --checkpoint /path/to/workspace/models/Wan2.2-T2V-A14B \
  --output-dir /all/yiran07-disk3/huteng_data/exp/dicache_vbench200 \
  --gpu-ids 0 --threshold .08 --retention-ratio .2
```

每worker只加载一次pipeline，同一prompt的baseline和candidate在同一卡配对，默认一次完整warmup；每视频重新安装cache和profiler。`--resume`严格核验输入、源码、profile、GPU、warmup和产物SHA，已完成的视频直接复用，未完成尝试保留后重做。`--defer-evaluation`允许质量暂缓，但此时不生成整套COMPLETE标记。可用`--calflops-profile`复用相同源码和权重的DiCache profile。

标定使用 `experiments/parameter_scan/run_scan.sh`，共同参数同上，并可指定 `--thresholds .04 .08 .12 .2 .3 .5 --retention-ratios .2 --target-speedups 1.8 2.4 3.0`。默认11条覆盖16维的prompt、6档候选、77个测量视频。DiCache决策依赖实时feature，不按预计算schedule合并参数；目标未达到时记录unmatched。

三档自动标定入口及执行命令见 [targeted_threshold_scan/README.md](experiments/targeted_threshold_scan/README.md)：首轮8个阈值，最多4轮，默认目标±2%，最终报告实测阈值和所选配置质量。

## 计量和质量

- Headline latency：CUDA同步的完整 `WanT2V.generate()` 墙钟时间，包含T5、DiT、probe、DCTA、CFG/scheduler、offload搬运与VAE；排除pipeline初始化、MP4写盘和离线评测。
- 分别保存T5/DiT/VAE的CUDA时间与host span，初始化和warmup单独记录。
- Headline DiT TFLOPs：逐专家、逐CFG调用使用实测的full或**1-block probe路径**成本；另保存embeddings/head-only成本用于审计。Calflops未覆盖的FlashAttention核心用dense计数补偿，T5/VAE独立profile。Gate/DCTA、残差加法及scheduler等未计FLOPs明确披露；TFLOPs表示运算量，不是TFLOP/s。
- 质量使用同仓库VideoMetrics的RGB PSNR/SSIM/LPIPS，以及VbenchEvaluation的VBench score。默认standard子集必须覆盖16维；显式custom模式标记十维raw score。
- 完整报告仅在全部生成、性能汇总和质量阶段通过后完成。没有实测标定前，不将参数值宣称为某个目标加速比。

## 验证范围

```bash
WAN22_PYTHON="$CONDA_PREFIX/bin/python" bash tests/run_tests.sh
python scripts/validate_source.py --source "$PWD/build/Wan2.2-42bf4cf"
```

CPU对照直接导入完整官方Wan2.1脚本，检查12组参数/精度组合的1200个调用，输出和block决策逐项相同；另覆盖独立CFG、严格阈值、原版0/0行为、默认三份历史、单历史回退、多视频与异常清理、持久batch恢复和probe计量。

小尺寸真实WanModel CUDA/BF16验证覆盖100个native/all-full精确相等输出、100个重复cache精确相等输出及真实Calflops full/probe/head成本分离。CPU真实block检查仅将CUDA attention替换为SDPA。以上均为实现验证，尚未运行完整A14B视频、完整组件profile或质量/速度标定。

GPU1/2/3、固定retention=.2、约1.5×–3.5×范围的阈值扫描入口：`experiments/speedrange_gpu123/run_scan.sh`（实验目录内见`speedrange_gpu123/README.md`）。

正式三档VBench200及复用baseline入口：[vbench200_reuse_gpu123](experiments/vbench200_reuse_gpu123/README.md)。

当前修正后的正式VBench200入口：`experiments/vbench200_balanced_gpu123/`（th=.075/.172/.396、三卡按预计耗时均衡分配、复用已验证结果和baseline）。历史`vbench200_reuse_gpu123`的.072中档已被用户更正，保留原入口仅用于溯源。
