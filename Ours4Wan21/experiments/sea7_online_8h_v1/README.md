# SEA7 e328：四卡八小时在线微调

`prepare_inputs.py`核验原离线train轨迹，按实测复用K分组导出速度先验，冻结3000-prompt训练池，并按四卡均衡采集、组件加载/预热、质量计算、IQL和末轮三档评测估算完整轮数。结果写入外部`ours21_sea7_e328_online_8h_v1_setup/`。

起点为原稳定选点SEA7 e328；不是offline最小validation loss点。采集目标1.5–3.5×、100条/轮、5 critic+20 joint epoch、e11–20选点，评测使用既有20条VBench50子集和baseline、K23/K29/K35，无VBench评分。轮数通过`online.py prepare --rounds`冻结；每4轮及末轮评测。时间估算不代表到8小时强行中断。

`run.sh`记录本机正式启动环境与冻结run路径。实际预算与来源见setup的`budget.json`，正式run的`STATUS.json`记录当前阶段。

本次已冻结2轮，含末轮评测的点估计为7.855小时：R1采集/质量/训练3.217小时，R2为3.147小时，末轮三档离线/在线对比1.491小时。3轮估计10.89小时。196个新训练baseline、200个在线候选、60个离线参考候选和60个末轮候选；20个评测baseline直接复用。实际时间随GPU温度和加载/IO变化，不在8小时强制杀进程。

启动：`bash experiments/sea7_online_8h_v1/run.sh prepare`；随后`bash experiments/sea7_online_8h_v1/run.sh run`。本次后台会话为`ours21_sea7_online_8h`，正式结果位于`/mnt/hdd/xiongyuxiang/tmp/exp/ours21_sea7_e328_online_8h_v1/`，checkpoint在同名models目录。
