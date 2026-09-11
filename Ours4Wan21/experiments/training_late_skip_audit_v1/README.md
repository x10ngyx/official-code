# 训练数据后段集中 skip 覆盖审计

`analyze.py` 读取 e391 实际使用的 Dynamics128 transitions.pt 与冻结 manifest，逐条核对
3000 条原始 trace 的 SHA256、双 CFG 动作和训练 tensor，分别统计 train/evaluation/test。
参照既有 increase 五档实测20条trace，严格按相同总skip数对比后25步skip数，另加
后25步最长连续skip作为更严格的补充。后20/15步与skip位置重心用于敏感性检查。
不按threshold是否递增替代实测动作，也不把验证/测试数据计入训练样本。

运行：
```bash
/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python work/official-code/Ours4Wan21/experiments/training_late_skip_audit_v1/analyze.py
```

结果位于 `/mnt/hdd/xiongyuxiang/tmp/exp/ours21_training_late_skip_audit_v1/`，通过本项目
`experiment_results/`软链接访问。包括逐轨迹CSV、同K覆盖率、阈值敏感性、actor可学习
动作数、原始来源哈希、可复现notebook和技术报告；仅CPU分析，不新增采集或训练。

图表契约：比较同K的increase参照与训练分布；采用按K分组柱图展示后段skip覆盖率，
另用训练数据的后段skip数直方图展示全量支持范围。明确分母与原始参照数，不插值或
外推未测K。报告采用canonical artifact的本地图表与portable HTML。
