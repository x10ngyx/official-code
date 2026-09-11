# SEA7 / Dynamics128 × 四档激进IQL

每种特征各四组、每组从头400epoch，batch256/seed42/lr1e-4等保持不变；
当前已在训练的Dynamics128 A3纳入八组，不重算、不新增第九组。

| 档位 | expectile τ | 标准化优势 β | 权重上限 |
|---|---:|---:|---:|
| A1 | .70 | 1.5 | 30 |
| A2 | .80 | 2 | 50 |
| A3 | .90 | 3 | 100 |
| A4 | .95 | 5 | 200 |

原组为.60/1/20。提高expectile、优势选择性和上限是联合实验，不能分离单参数贡献，
也不保证学到更靠后的skip。β作用于batch标准化优势，不能直接数值对标官方原始优势温度。
参考：[官方actor](https://github.com/ikostrikov/implicit_q_learning/blob/master/actor.py)、
[官方expectile .9例](https://github.com/ikostrikov/implicit_q_learning/blob/master/configs/antmaze_config.py)。
A4是本实验更强的探索档，不是官方推荐配置。

`suite.py`：冻结八组、输入SHA和10prompt；四卡各跑两组，选点/Q分析后统一四卡推理、质量评测和报告。
`common.py`：参数/路径、固定抽样、完整性检查。
`q_worker.py`：每组与相应原组的全400轮mean_Q/Q-IQR和训练图。
`generate_worker.py`：每卡只初始化/原生预热一次，切换八组checkpoint和三个K，逐视频可校验恢复。
`report.py`：240候选/19440对帧、逐步动作和组件计量审计、24行结果和跨组PNG/SVG。
`test_suite.py`：八组边界、复用第三档、固定10条、trace和真实checkpoint读验。
`test_report.py`：以明确的临时合成夹具验收240视频/4000条Q数据的最终报告，夹具不作为实验结论。

10prompt从原冻结online20按GPU 3/3/2/2条、seed42无放回抽样；不按结果挑选。
全部组使用完全相同10条、同物理GPU原native baseline。三档1.8/2.4/3.0×固定K23/K29/K35。
沿用用户此前要求：VBench提示词测试，但不计算VBench分数，仅PSNR/SSIM/LPIPS和性能。
选点仅使用e301–399双侧actor一致率和Q漂移，若不能达到.96/.95门槛，报告实际回退门槛，
不得将相对稳定当作已收敛。视频测试结果不进入选点。

图表合同：标准Matplotlib独立PNG/SVG；每种特征四档加原组，横轴epoch1–400；
mean_Q=min(Q1,Q2)在同13738个验证自由状态×2动作取均值，Q-IQR用midpoint P75−P25。
颜色用蓝色层次、原组深灰，辅以线型/标记，两个特征分面；未经最终图像检查不宣称视觉验收。
质量图按特征分面、实际K顺序，保留逐prompt CSV和实际速度，单seed描述性比较。

运行：Wan2.2环境 `python experiments/iql_aggressiveness_2x4_v1/suite.py`；`--resume`可恢复已冻结的suite。
结果 /mnt/hdd/xiongyuxiang/tmp/exp/ours21_iql_aggressiveness_2x4_v1；
各组训练/分析外部目录通过groups/链接，模型在models/对应目录。
STATUS.json为当前阶段；TRAINING_COMPLETE.json表示八组训练/选点/Q图完成，
COMPLETE.json必须240候选及全部质量和报告通过后才发布。旧单组20prompt后处理已被替代。
