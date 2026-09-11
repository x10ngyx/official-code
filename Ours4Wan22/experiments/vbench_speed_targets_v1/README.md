# Ours4Wan22 VBench 多加速档测试

参考 Ours4Wan21 `vbench5_speed_targets_v1` 的固定 prompt、target→K、共享同卡 baseline、生成→质量评测→报告流程，调用 Wan22 正式 `main.py generate/evaluate`。本脚本在指定的一张物理 GPU 上顺序运行各档，每次生成进程内部持久加载 pipeline，并排除一次完整预热。所有模型和固定推理协议沿用 Ours4Wan22；默认标定为20260912新表，不使用 Wan21 的 K 映射。

- `run.py`：输入冻结、命令计划、baseline、多个目标、完整评测、阶段恢复及汇总。
- `run.sh`：wan2.2 Python入口及四项 BLAS 线程限制。
- `test_suite.py`：CPU prompt、标定、命令、dry-run及恢复合同测试。

默认使用 Wan21 同一固定五提示词 `001/016/056/135/159`，三个目标 `1.8/2.4/3.0×`，新K为 `24/31/36`。`--prompt-ids`可选其他VBench200 ID；`--all-prompts`使用完整200集。五条或其他子集调用官方 custom-input 十维 VBench，不能当作16维VBench200聚合分数；完整200集核对每个ID及实际prompt文本后运行VBench200模式。始终计算VideoMetrics RGB PSNR/SSIM/LPIPS及VBench score，没有默认省略评分。

在项目 `Ours4Wan22/` 目录、conda `wan2.2` 下运行：

```bash
bash experiments/vbench_speed_targets_v1/run.sh \
  --policy ../../../models/ours22_random2323_cnn_g1_20260911_223523/selected.pt \
  --wan22-root ../SeaCache4Wan22/build/Wan2.2-42bf4cf-prepared-modnorm-fix \
  --profile /path/to/calflops_profile.json \
  --gpu 1 \
  --output /all/yiran07-disk3/huteng_data/exp/ours22_vbench5_new_calibration \
  --dry-run
```

`--profile`必须为同模型、同prepared source的真实组件Calflops profile，可复用SeaCache生成的profile（本机已存在的路径见其标定实验plan.json）。`--dry-run`只读取输入、核对prompt及标定并打印命令，不建结果目录或运行GPU；去掉该参数执行真实测试。也可设置 `WAN22_PYTHON=/home/huteng/yes/envs/wan2.2/bin/python`。

- 完整测试：增加 `--all-prompts`。
- 五档测试：增加 `--targets 1.5 2.0 2.5 3.0 3.5`，对应K18/27/32/36/38。
- baseline默认只生成一次并供所有档位共用。`--baseline /path/to/baseline`只接受**本项目正式generate产出的完整native baseline**，必须同prompt列表、协议、模型、prepared、profile、物理GPU UUID，逐文件SHA核验；SeaCache历史结果的不同schema不能直接传入。
- 继续执行：同一命令加`--resume`。输入/源码/参数和物理GPU必须不变，已封存完整阶段核验后复用；残缺或尚未封存的目录保留并报错，不覆盖、不假装完成。需要重跑残缺阶段时使用新的suite输出，可显式复用已完成baseline。

全部输出在指定外部实验目录，项目experiment_results只建symlink。根目录含plan.json、jobs.json、gpu.json、STATUS.json、logs/、stages/；baseline/target/metrics目录以suite名称作前缀，避免通用入口symlink重名。summary.json保留各档完整质量、T5/DiT/VAE及CNN分项时间/TFLOPs、VBench分项；results.csv与REPORT.md汇总目标/K、实际generate speedup及VBench。COMPLETE.json仅在全部生成和评测成功后写入。目标标定来自SeaCache，未含Ours CNN/feature开销，实际加速比以测试计时为准。

CPU验证：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m unittest discover -s experiments/vbench_speed_targets_v1 -p 'test_*.py' -v
```
