# MagCache4Wan21

来源：https://github.com/maverick-ssh/wan21-cache-benchmark @ 8a70ce22c9577c33ba38d7a641b8cf9cef8e3938。

保留远端magcache.py原样：官方1.3B的100项raw ratio、分支独立累计误差、K限制及残差复用。没有加入ratio sqrt或其他算法修改。

运行环境为conda `wan2.2`。固定Wan2.1-T2V-1.3B、832×480、81帧、16fps、50-step UniPC、shift5、CFG5、seed42、BF16计算；单张48GB GPU、所有组件常驻，`offload_model=False`、`t5_cpu=False`。模型在根`models/`，结果在`/all/yiran07-disk3/huteng_data/exp`，本目录`experiment_results/`只保存symlink。

- 核心.py文件：方法、原生Wan21接入和组件计时。
- `experiments/fixed_protocol/`：本地单卡持久生成入口，`--help`查看参数。
- `tests/`：CPU回归及一致性测试。
- `upstream_lock.json` / `NOTICE.md`：来源、哈希和授权边界。
- `PROGRESS.md` / `logs/`：状态及交接。

完整baseline、生成、组件TFLOPs、PSNR/SSIM/LPIPS/VBench200使用说明见[Wan21Benchmark](../Wan21Benchmark/README.md)。公共工具直接使用本地既有版本；远端服务器路径、offload配置和历史结果没有导入为本地正式实验结果。remote实现中与本机绑定的调度/标定/交接脚本仍保留在审查快照中，运行入口已换成本地适配器。

```bash
conda activate wan2.2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m unittest discover -s tests -v
```

上述测试在当前目录执行。此次代码接入未启动新的完整模型推理或质量评测。
