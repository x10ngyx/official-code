# 八组IQL训练图即时读出

用户要求先展示训练分析图。`run.py`仅读取八组已完成的400轮记录及Q统计，
不等待视频质量、不修改正在运行的suite冻结源码或配置。
结果保存到原suite的`training_readout/`，与自动最终报告figures/分开。

图表：SEA7与Dynamics128各一张2×3训练/验证Q/V/actor损失图，损失纵轴明确使用log；
一张2×2 mean_Q/Q-IQR图，比较原参数与A1–A4完整400轮，标记各组选定epoch。
三图均为连续epoch趋势，采用同一蓝色层次和不同线型，原参数深灰。
训练损失目标随expectile/优势权重改变，曲线不是跨档视频质量排名。
Q=min(Q1,Q2)，同13738验证自主状态×2动作；Q-IQR为midpoint P75−P25。
保存完整CSV、来源SHA、窗口均值及PNG/SVG，以实际图像检查记录验收。

`final_results.py`在suite完成后独立复核252份证据、24条件均值及240条skip path，
并将原e391限制到同10prompt，验证GPU和native视频SHA后输出`final_readout/`配对CSV与VALIDATION。
运行：`python3 experiments/iql_training_readout_v1/final_results.py`。
