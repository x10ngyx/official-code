# Experiments

本目录用于保存 TeaCache4Wan22 的实验脚本。每项实验应放在独立子目录中，并在该子目录提供 `README.md`，说明配置、运行方式与输出位置。

实验产物不得直接写入本目录；应保存到仓库外部的独立实验结果目录。

当前实验：

- `fit_t2vcompbench70_wan22_t2v_a14b/`：使用 70 条分层抽样的 T2V-CompBench prompts，采集 full-compute `e/H/(H-Z)` 相邻 relative-L1，并为 Wan2.2 T2V-A14B 的 high/low 阶段分别拟合四次多项式。
