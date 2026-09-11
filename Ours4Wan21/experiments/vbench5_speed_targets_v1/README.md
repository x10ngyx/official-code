# VBench200 五提示词三档加速流水线

该目录实现 12 次 400-epoch 训练后的自动化闭环：读取每组既有的 post-300 稳定性选择结果，冻结一个与质量/速度无关的 VBench200 五提示词子集，按既有标定固定的 Exact-K 运行 1.8×、2.4×、3.0× 三档匹配测试并生成技术报告。

## 目录

- `subset/`：固定五提示词、prompt map、源 metadata、选择清单和 SHA256。
- `build_subset.py`：在 200 条记录上用 bitmask 动态规划最大化五条记录覆盖的 VBench metadata 维度；并列时取 sample ID 字典序最早组合。
- `pipeline_lib.py`：固定目标/K、模式清单和结果合同检查。
- `run_pipeline.py`：四卡编排、baseline、固定 K23/K29/K35 的36个目标条件、质量评测、分析和 HTML 报告打包。
- `analyze_results.py`：fail-closed 审计，输出 aggregate/per-prompt/calibration 三张表和指标定义。
- `build_report.py`：从已验证结果生成 canonical `artifact.json`；流水线随后用共享报告打包器生成自包含 `report.html`。
- `queue_after_training.sh`：等待 12×400 训练编排完成后启动整套流水线；它只等待 completion marker，不解析或监控训练曲线。
- `handover_without_vbench.py`：在旧调度器暂停后，等待其现有生成子进程完成，再接续
  `--resume --skip-vbench`，避免丢弃正在计算的完整条件。

## 固定实验设计

五条 prompt 是 `vbench200_001/016/056/135/159`，在五条记录的限制下覆盖 10 个 VBench metadata 维度。五条无法覆盖官方 16 维，因此 VBench 只调用官方 `custom_input` 支持的 10 个维度实现并报告未加权原始均值；所有产物明确标注这不是完整 VBench 或 VBench200 官方分数。

K23/K29/K35 直接来自既有 SeaCache 分支 reuse-count 拟合：`K≈(18.52184909×speedup+13.48716729)/2`。流水线核对拟合来源 SHA 和公式后冻结 1.8×→K23、2.4×→K29、3.0×→K35；按用户要求不做逐组重标定、PAVA 或自适应补测。每组使用自己的 selected checkpoint、固定物理 GPU 和本卡 native baseline 实测完整 `pipeline.generate`，并如实汇报名义目标偏差，但不因此改变 K。

每个最终条件保留：完整 generate latency、T5/DiT/VAE CUDA 时间、DiT/T5/VAE TFLOPs、predictor 时间与 TFLOPs、latent feature wall time、PSNR、SSIM、LPIPS 和 VBench custom 诊断分数。Headline speedup 是五条 prompt 的 baseline 时间总和除以 candidate 时间总和；baseline 与 candidate 必须使用同一物理 GPU UUID。

物理 GPU UUID 优先读取 PyTorch device properties；兼容未暴露 `uuid` 的 PyTorch
版本时，先用当前 CUDA 进程向 `nvidia-smi` 反查物理 UUID，再以流水线唯一的
`CUDA_VISIBLE_DEVICES` 和 `PCI_BUS_ID` 顺序兜底。无法唯一解析时必须 fail-closed，
不能退化为设备名称或逻辑 `cuda:0`。

## 运行

训练已完成时直接运行：

```bash
/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python \
  experiments/vbench5_speed_targets_v1/run_pipeline.py
```

需要在当前长训后自动接续时：

```bash
bash experiments/vbench5_speed_targets_v1/queue_after_training.sh
```

大结果根为 `/mnt/hdd/xiongyuxiang/tmp/exp/<run-name>/`，项目的 `experiment_results/<run-name>` 只有一个 suite 级 symlink；各 baseline/candidate 是该 suite 内的嵌套结果，避免产生几十个顶层 symlink。中断后用完全相同参数加 `--resume`；完整条件会严格核验后复用，残缺条件不会被静默覆盖。

本次用户明确这是非正式测试，不运行 VBench 评分。`--skip-vbench` 保留全部生成、
性能与 PSNR/SSIM/LPIPS 评测，并在结果根写入 `VBENCH_SKIPPED_BY_USER.json`；后续
resume 自动遵守该持久化设置。最终 CSV 的 VBench 字段留空，报告隐藏 VBench 图表，
不将未测值记为零。此标记仅改变评测范围，不修改已冻结的 prompt、checkpoint 或 K。
# Completed-result readout

`readout_completed.py` independently verifies existing training hashes, all 36 generation/quality
conditions, video hashes and same-GPU/Exact-K pairing, and recomputes speed and quality means.
It is read-only: it neither repairs the report pipeline nor writes a completion marker.
