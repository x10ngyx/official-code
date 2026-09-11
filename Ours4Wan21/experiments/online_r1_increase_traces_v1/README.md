# 在线R1 / Increase实测trace

`plot.py`读取Dynamics128 e391在线A2/A3中间档8轮run的R1全部100条采集结果，
加入之前Increase五档各一条（固定VBench155），按实际skip数全局升序分3页各35条。
R1采集使用起始e391及sampling_seed，不是R1训练后模型的确定性评测。
R1为OpenVid训练prompt，Increase为VBench155；仅路径结构参考，不作配对画质结论。

图表契约：完整50步二值矩阵，白色skip、蓝色R1重算、橙色Increase重算，
文字标签和分隔线补充颜色；后25步边界虚线，右侧skip数/后25数/后段最长连续数/PSNR/实测速度。
105条/5250格/10500次CFG调用，核对封存trace、timing、双CFG动作与实际blocks。
固定独立CPU脚本，无新增GPU任务，不修改运行中源码。

输出在在线run的`analysis/r1_increase_traces/`，由项目原结果symlink访问：
3张PNG/SVG，全部trace/逐步/CFG/prompt CSV，VALIDATION与视觉验收。
运行：Wan环境Python执行本目录`plot.py`。导出后实际查看全部3张PNG。
