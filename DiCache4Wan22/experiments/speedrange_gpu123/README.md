# GPU1/2/3：约1.5×–3.5×阈值扫描

用户指定物理GPU1、2、3，固定retention=.2/probe1，仅扫threshold。配置在presets.json；入口run_scan.sh调用公共自适应扫描器。
初始阈值为.10/.20/.40/.80，这是探索起点，尚不是已测速度映射；55个测量视频（11 baseline+44 candidates），另有3次full warmup和GPU1真实形状组件profile。
按prompt轮转分配三卡（4/4/3条），同prompt所有条件在同一卡生成，每worker持久pipeline、错峰加载。

最多4轮，后续只细化1.8/2.4/3.0×附近，默认相对容差±2%。不继续细化两个端点均低于1.5×或均高于3.5×的区间；
仅当边界实测速度仍在目标同一侧时向相应方向扩展。初始探索可能测到区间外点，这些点如实保留；
不会预先假定threshold与速度的精确映射，也不为覆盖完整加速区间增加采样点。
每轮新baseline配对，复用首轮组件profile；最终自动完成所选三档VideoMetrics与16维VBench子集质量。

从项目根目录运行：

```bash
conda activate wan2.2
export WAN22_PYTHON="$CONDA_PREFIX/bin/python"
bash experiments/speedrange_gpu123/run_scan.sh \
  --output-dir /all/yiran07-disk3/huteng_data/exp/UNIQUE_DICACHE_SCAN
```

`--dry-run`仅校验计划；断点追加`--resume`。结果与symlink规范、计量范围和未命中处理见
[公共扫描说明](../targeted_threshold_scan/README.md)。此入口不会抢占既有GPU进程，启动者须确认三卡空闲。
