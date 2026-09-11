# 线性递增 threshold 的同速对照

**当前已完成40条轨迹和VideoMetrics。用户明确要求不算VBench，已停止且不纳入报告。最终入口为 `finalize_metrics.py`；不要运行旧的VBench续算入口。**

Wan2.1-T2V-1.3B 固定协议；同4条既有随机50中的速度验证prompt，每卡固定1条，复用同物理GPU原生baseline。

- `schedule.py`：继承现有corrected SeaCache，仅替换当前step的threshold；两个CFG分支使用相同50步path，独立状态及原边界不变。
- `worker.py`：每卡模型常驻，1次排除计量的native warmup后运行5条increase，等待统一标定，然后5条fixed。
- `run_experiment.py`：冻结输入；20条increase实测速率→旧标定相邻实测点插值→20条fixed；VideoMetrics与VBench custom；校验40条及组件计量后汇总。
- `test_schedule.py`：固定path与原控制器精确parity、索引/边界、标定拒绝外推及非法输入。

5个path首末阈值为0.04→0.24、0.08→0.40、0.12→0.60、0.16→0.80、0.20→1.00；公式 `a+(b-a)*step/49`，step=0…49（采样执行顺序，非训练timestep）。
固定阈值依据4条baseline总时间/4条increase总时间，由原标定CSV相邻实测点插值；不用质量选参、不额外补推理。
实测固定/递增耗时差≤5%标记近似等速；不符合则报告偏差，不作等速质量结论。
共40条正式trajectory；4条历史baseline复用，4次native warmup不作为实验样本。保存T5/DiT/VAE分项时间及TFLOPs。
VBench score为10维custom-input原始分数均值，不是官方VBench标准总分。

结果在 `/mnt/hdd/xiongyuxiang/tmp/exp/seacache_wan21_linear_increase_matched_5x4_v1/`，经本项目experiment_results软链接访问。
使用Wan2.2环境Python运行 `run_experiment.py --prepare-only` 冻结，然后 `run_experiment.py` 启动/续接；受共享四卡文件锁保护。`--finalize-only`只复核并汇总。

`finalize_metrics.py`按显式用户覆盖标记汇总40条VideoMetrics与计量，`audit.py`独立复核；`resume_evaluation.py`和`finish_vbench.py`保留为过程代码，目前不再使用。

`plot_traces.py`只读既有40条轨迹，导出全部50步对照图至结果的`analysis/trace_comparison/`，附CSV/完整prompt/来源与可视检查记录；无需GPU或VBench。
