# CNN计算量静态估算

calculate.py：按明确列出的候选层尺寸计算单样本actor的Conv/Linear MAC与参数量，不调用GPU。三分支CNN未定稿，本结果是条件性算术估算，不是最终架构或速度基准。结果写HDD exp并通过experiment_results链接。
