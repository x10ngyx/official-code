# Increase 与 dynamic e391 实测 trace 对比

`plot_traces.py` 只读已有结果，选取 increase 实验已冻结的同四条 prompt
（040、041、144、155），对比 increase 五档与 Dynamics128 e391 的 K23/K29/K35/K38。
每档四条，共36条；按实际 skip 数升序，再按方法、档位、prompt ID 排序。
不根据质量或 trace 形状选样，不启动推理或评测。

图表契约：问题是不同 skip 数下两种方法的重算位置如何分布；采用完整36×50
离散动作矩阵，所有step保留。蓝/橙色实心格为两种方法的重算，白色格为skip；
方法与档位同时用行标签标识，右侧精确标出skip/重算数。研究图采用Matplotlib
导出独立PNG/SVG以供本地查看与论文使用。只描述已选轨迹，不推断总体质量或等速优势。

运行：
```bash
/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python work/official-code/SeaCache4Wan21/experiments/increase_e391_trace_comparison_v1/plot_traces.py
```

结果根：`/mnt/hdd/xiongyuxiang/tmp/exp/wan21_increase_e391_trace_comparison_v1/`；
包含PNG/SVG、完整prompt与逐步CSV、36条汇总CSV、来源SHA256及验证记录。
对应链接位于本项目 `experiment_results/`。
