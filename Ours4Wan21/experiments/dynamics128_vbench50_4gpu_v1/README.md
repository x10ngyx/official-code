# Dynamics128 随机 VBench50 四卡测试

## 当前评测范围（用户最新要求）

仅计算PSNR、SSIM、LPIPS，不计算VBench分数。结果根的`VBENCH_SKIPPED_BY_USER.json`
是持久scope override；冻结原config保留，明确登记旧/新调度脚本SHA，其他来源仍严格校验。
三项质量完成后直接生成results.csv、RESULTS.md、VALIDATION.json和根COMPLETE.json，
其中vbench_status为skipped_by_user，不依赖VBench目录或分数。
`handoff_quality_only.py`等待既有生成子进程正常结束后替换已暂停的旧调度器，再resume；
不终止生成、不覆盖视频。以下VBench描述保留为原始设计，不代表当前启用。

固定已选 Dynamics128 e391 (`sea7_dynamics_raw_sea128`)；不重新训练或选checkpoint。
从冻结VBench200的200条prompt以Python `random.Random(42).sample(rows, 50)`一次无放回
抽样，再按sample_id排序、索引模4分片为13/13/12/12条。生成seed另固定42。
不按质量筛选、不为覆盖率重抽、不排除原5prompt（实际重合056），本次恰好覆盖16维。

三档固定K23/K29/K35，共50 native baseline + 150 candidates。各卡生成自己的baseline，
与同卡候选配对；每个生成进程额外一次native warmup，不计入推理耗时。
固定Wan2.1-1.3B、832×480、81帧、16fps、50步UniPC、shift5、CFG5、BF16、无offload。
记录完整推理及T5/DiT/VAE/predictor/feature时间、各组件TFLOPs、逐步trace；调用
VideoMetrics的PSNR/SSIM/LPIPS，VBench standard 16维用于相同50prompt子集。
三档速度是名义目标，实际速度单独汇报；不做自适应K补测。

- `run_suite.py`: 冻结随机清单和来源哈希；四卡生成→质量→四条件VBench→CSV/Markdown汇总。
- `test_suite.py`: 随机抽样、分片覆盖、逐视频质量字段/唯一键、性能配对和真实元数据覆盖测试。
- `plot_k29_traces.py`: CPU只读验收既有e391 K29全部50条trace，导出完整/分半PNG与SVG、
  完整prompt/双CFG逐step CSV及来源哈希；图合同见`TRACE_K29_CHART_CONTRACT.md`。
- `--prepare-only`: 只创建冻结实验；`--resume`: 执行并复用完整子阶段。
  不完整生成/质量目录拒绝覆盖；VBench由完成标记确认，不把目录存在视为完成。

结果：`/mnt/hdd/xiongyuxiang/tmp/exp/ours21_dynamics128_e391_vbench50_random42_4gpu_v1/`，
子项目experiment_results为symlink。`prompts/`保存50清单、4shard及full-info；`shards/`
保存每GPU生成/质量；`merged/`为视频symlink；`vbench/`为4条件分数；`analysis/`为汇总；
`logs/`与status.json为运行状态。此任务不依赖旧五提示词报告，不修改SEA7全量队列。
启动前确认GPU空闲；GPU期间持有本实验锁和共享咨询式GPU锁。

```bash
/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python experiments/dynamics128_vbench50_4gpu_v1/run_suite.py --prepare-only
/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python experiments/dynamics128_vbench50_4gpu_v1/run_suite.py --resume
```

仅称为随机VBench50子集结果，不是官方全量排行榜成绩。VBench通用入口的历史静态
dataset标签仍为VBench200；本套件冻结的50条full-info与外层scope.json明确真实范围。
