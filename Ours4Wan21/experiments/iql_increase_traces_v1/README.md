# 八组IQL / Increase实测trace图

CPU读取完成结果，不新增推理。运行Wan环境Python的`plot.py`，输出原八组suite的
`trace_comparison/`（HDD归档，沿用项目experiment_results软链接）。

图表契约：比较50步中skip的位置、后25步数量与最长连续段，而非只看总数。
采用逐步二值矩阵，PNG/SVG科学图；完整格子、不做平均路径。白色skip，蓝色训练策略重算、
橙色Increase重算，方法文字标签与分隔线提供非颜色区分。按实际skip数升序排列。
右侧显示实际skip、后25步skip、后25步最长连续skip、PSNR与实际加速。

选择：唯一公共prompt155用于严格配对总图（24条训练+5条Increase）。
展开图每组每K固定4prompt：先保留公共155，再按ID选另外3条008/023/024；不按质量/路径挑选。
Increase保留原4条040/041/144/155，星号标唯一公共prompt。
K23图加入Increase1，K29图加入Increase2/3，K35图加入Increase4/5：
依次覆盖相邻实际skip范围21–22、27/31、34/36，不能视为等K/等速对照。
三张展开图包含96条训练+20条Increase，共116条唯一trace/5800格/11600次CFG调用。

输出包含4张图各PNG/SVG、完整trace/逐步/CFG调用/prompt CSV和来源哈希及QA。
渲染后实际查看每张PNG，检查文字溢出、50步完整性、行排序和图注范围。
