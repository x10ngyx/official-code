# BLOC feature redesign v2

`DESIGN.md`记录实现审计、信息边界、公式与分阶段比较协议。
`features.py`导入共享的`ours4wan21/bloc_features.py`；`test_features.py`检查因果性、
量化一致、缓存/历史重置及真实Controller双CFG Exact-K一致性。
`audit_and_benchmark.py`审计冻结划分和输入归档，并测量单条训练轨迹的GPU提取开销。
`prepare.py`支持initialize/extract/finalize，四卡入口为`launch_prepare.py`；
一次提取ABC144，再切出A64/AB128/AC80/ABC144，构建沿用SEA7动作、奖励和划分的训练cache。
`train_candidates.py`按相同400epoch、显式训练seed并行训练候选，再做验证集post300选点。
特征排名必须另用验证集闭环结果；训练完成不代表特征有效或完整实验完成。

结果只写`/mnt/hdd/xiongyuxiang/tmp/exp/`，由项目`experiment_results/`建立软链接；
权重写统一`models/`目录。历史特征定义和已有checkpoint不覆盖。

当前审计/微基准正式归档：`ours21_bloc_features_v2_audit_benchmark_v2/`。
3000条轨迹、1000个prompt、150000个输入latent元数据和路径完整，ID/规范化文本无跨split。
元数据检查不是全量tensor内容检查；重提取会逐份加载并验证、保存SHA。
10项新增检查和64项现有项目回归通过。尚无新候选训练或闭环质量结论。

运行示例（Python为既有Wan2.2环境）：

```bash
python experiments/bloc_features_v2/test_features.py
python experiments/bloc_features_v2/launch_prepare.py --output-dir /mnt/hdd/xiongyuxiang/tmp/exp/ours21_bloc_features_v2_features_v1
# 上一步完整结束后执行；不会使用smoke跳过正式训练合同。
python experiments/bloc_features_v2/train_candidates.py --feature-root /mnt/hdd/xiongyuxiang/tmp/exp/ours21_bloc_features_v2_features_v1 --suite-name ours21_bloc_features_v2_screening_v1 --seed 42
```

全生成入口现保存每视频allocated/reserved CUDA峰值，计时仍排除模型加载、warmup和视频编码。
推理seed保持42；`train.py --training-seed`只改变训练RNG，不改变默认400epoch和其他IQL参数。
