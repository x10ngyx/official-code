# Mixed3500 / e391 / Increase二十条trace

`plot.py`只读取已封存输出，用Matplotlib导出PNG/SVG；结果存于训练suite的analysis/trace20_comparison。
selection.json冻结选样时刻的已完成清单和规则：两新模型共同完成的K23前3个、K29前2个prompt ID，配同prompt同K原e391，共15条。
Increase从本轮500条取目标1.5/2/2.5/3/3.5最近样本（并列按trajectory ID），共5条；不按路径/画质选样。
按实际skip数升序，同预算中按prompt ID和来源排列。保守浅蓝、激进浅粉、e391浅橄榄、Increase浅黄；深色重算、白格复用。
同三模型配对保留50步；Increase来自OpenVid而非VBench，只作路径结构参考。当前未完成的质量评分不补算、不填入图。
原始2000次CFG与blocks、完整封存SHA、选中ckpt及同GPUbaseline核对，1000格无删减。CSV和VALIDATION保留来源和选样。
