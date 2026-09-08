# DiCache 三档阈值扫描

`run_scan.py` / `run_scan.sh` 是完整扫描入口，`presets.json` 保存搜索范围、预算与目标。
固定官方 **retention=.2、probe=1**，仅调整 threshold；不改官方 gate、DCTA 或 Wan2.2 冻结推理协议。

## 扫描流程

1. 粗扫 `.04/.08/.12/.2/.3/.5/.8/1.2`，使用 `../parameter_scan/prompts.jsonl` 中覆盖16个VBench维度的11条prompt。
   每个worker常驻一套pipeline，一次完整warmup；同prompt的baseline与全部候选在同一卡生成。
   首轮99个测量视频，另加每卡一次warmup和首轮真实形状组件profile。
2. 用 `sum(baseline.generate_wall_seconds) / sum(candidate.generate_wall_seconds)` 选三档最近实测点。
   默认容差为目标±2%：1.8×=`[1.764,1.836]`、2.4×=`[2.352,2.448]`、3.0×=`[2.940,3.060]`。
   这是可调整的扫描约定，并非用户指定的最终验收线。
3. 未命中档继续在最近点两侧及所有实测跨越目标的区间取中点；边界可减半/翻倍，阈值上限8、最小间隔.005。
   最多4轮（1粗扫+3细化）。不假定速度单调、不预测或合并不同阈值的gate序列。
   每轮仅测新阈值，并重新生成11个同卡baseline，避免跨轮计时漂移；后续轮共用首轮profile。
   后续视频数为 `11 × (1 + 本轮新阈值数)`，实际规模写入每轮plan。
4. 扫描结束后，只对三档最近候选及各自配对baseline计算统一RGB PSNR/SSIM/LPIPS、标准16维VBench subset score。
   若多个档选中同一候选，评测一次。选参以实测速度距离为准，质量如实汇报；质量分数不参与搜索。

搜索耗尽仍未达标时保留 `unmatched`，不改变retention来凑速度。
标定集的命中只代表该集合；这11条prompt上自适应选择结果不是独立泛化验证，也不保证VBench200速度一致。
每轮保存完整generate和T5/DiT/VAE分项时间、分项TFLOPs、逐CFG缓存trace与视频。
DiT复用成本包含一个probe block；cache gate/DCTA等未计FLOPs项沿用主README的披露。

## 执行

从项目根目录预览（GPU编号需按当前资源指定；脚本不会自动抢占或等待）：

```bash
conda activate wan2.2
export WAN22_PYTHON="$CONDA_PREFIX/bin/python"
bash experiments/targeted_threshold_scan/run_scan.sh \
  --output-dir /all/yiran07-disk3/huteng_data/exp/dicache_wan22_targets_scan \
  --gpu-ids 1 2 3 --dry-run
```

默认源码为本项目 `build/Wan2.2-42bf4cf`，模型为工作区 `models/Wan2.2-T2V-A14B`。
可用 `--source` / `--checkpoint` 覆盖。去掉 `--dry-run` 才加载模型；默认完成扫描及选中配置质量评测。
GPU1–3的例子不表示当前已获空闲资源。GPU0已有历史计时异常记录，宜使用健康卡标定。

- `--defer-evaluation`：先扫速度，质量保持pending。原命令加 `--resume` 并去掉此项即可补齐选中配置质量。
- `--resume`：逐轮复核plan、源码、GPU、prompt、profile和全部已完成视频SHA后复用；异常尝试由公共runner保留。
- `--presets /path/to/copied.json`：改变网格、目标容差或预算。应复制配置到独立实验脚本文件夹；已执行实验不修改原配置。
- `--calflops-profile`：可选，只接受同DiCache源码锁/模型的真实形状profile。

结果位于外部exp根目录，本项目 `experiment_results/` 自动建symlink。
`REPORT.md` / `scan.csv` / `target_selection.json` 给出实测配置；`performance.json`含逐视频及分组件数据，
`round_*/`含可审计的baseline、candidate与trace，`quality.json`含选中候选的质量。
`SCAN_COMPLETE.json`表示搜索结束，`COMPLETE.json`另外要求所选质量完成；两者的
`all_targets_matched`才说明是否三档均命中，不能仅凭文件名认为达标。
