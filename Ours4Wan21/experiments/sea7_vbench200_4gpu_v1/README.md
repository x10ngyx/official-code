# SEA7 VBench200 四卡接续测试

固定 SEA7 post-300 selected checkpoint（本批 e328），K23/K29/K35，对完整 200 条
VBench200 prompt 测试。四卡按 prompt 索引取模分片，每卡 50 条，同卡生成 native
baseline 及三档候选。共 200 baseline + 600 candidate 视频，每条件另有 native warmup。

- `run_suite.py`：准备冻结配置、等待五提示词任务 COMPLETE、四卡生成/三项质量、
  合并视频 symlink、四条件并行 VBench standard 16维评测、性能/质量 CSV 与 HTML 报告。
- `test_suite.py`：分片覆盖、前置完成与指标汇总合同测试。
- `--prepare-only`：仅冻结配置和排队信息，不启动 GPU。
- 默认运行：等待前置任务完整完成且四卡无计算进程后执行。`--resume` 复用完整结果；
  不完整目录保留并报错，不能直接覆盖。

结果位于 `/mnt/hdd/xiongyuxiang/tmp/exp/ours21_sea7_e328_vbench200_4gpu_v1/`，项目
`experiment_results/` 只存 suite symlink。目录结构为 `prompts/`、`shards/gpu{0..3}/`
（baseline、K23、K29、K35）、`merged/`（各条件视频软链接）、`vbench/`、`logs/`、
`analysis/`。运行过程和完成状态保存在 `status.json`、`COMPLETE.json`。

协议：Wan2.1-T2V-1.3B，832×480、81帧、16fps、UniPC50、shift5、CFG5、seed42、
BF16、单卡全部模型驻留 GPU。T5/DiT/VAE 和 predictor 分项计时/TFLOPs 全部保留。
headline speedup 为 200 条同卡配对完整 generate 时间总和比，排除加载、warmup、
MP4保存与评测。VBench 使用官方 16维归一化和加权公式，名称为 VBench200 subset
score，不称为官方完整排行榜分数。五提示词任务取消 VBench 的设置不影响本任务。

```bash
/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python \
  experiments/sea7_vbench200_4gpu_v1/run_suite.py --prepare-only
# 后台队列使用相同入口：
/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python \
  experiments/sea7_vbench200_4gpu_v1/run_suite.py --resume
```
