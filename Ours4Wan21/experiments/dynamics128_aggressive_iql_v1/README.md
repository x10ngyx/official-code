# Dynamics128 aggressive IQL：单组400轮对照

用户要求检验较保守IQL参数是否限制策略。本组从随机初始化重新训练，复用原Dynamics128
cache、3000条冻结selection与train/val/test=2400/300/300，仅使用train更新参数。

| 参数 | 原组 | aggressive_v1 |
|---|---:|---:|
| expectile tau | 0.6 | 0.9 |
| 标准化优势的指数系数 beta | 1 | 3 |
| actor优势权重上限 | 20 | 100 |

其他保持：400epoch、batch256、seed42、3×256 MLP、AdamW lr=1e-4/weight_decay=.01、
gamma=1、target_rho=.999、grad_clip=1、终局绝对RGB PSNR、Exact-K及actor_mask。
这是联合参数实验，不能区分三个参数各自的贡献。更激进指更偏向高价值/高优势动作，
不保证自动变成后段集中skip，也不能补出未覆盖的数据。

本地优势先在batch自主动作上做标准化，再计算 `min(exp(beta*z), weight_max)`。
z=1时权重从2.718升至20.086；z=2从7.389升至100（截断）。官方IQL采用原始优势，
温度不能直接数值对标；本组不改标准化和IQL内核。
参考：官方expectile=.9例 https://github.com/ikostrikov/implicit_q_learning/blob/master/configs/antmaze_config.py ，
权重上限100 https://github.com/ikostrikov/implicit_q_learning/blob/master/actor.py 。

- `run.py`：冻结来源/参数，校验原cache和e391数据来源，在指定空闲GPU从头训练400轮，
  自动执行原post300验证集选点、Q分析、online20三档测试与完成审计。
- `analyze_extra.py`：原组/激进组完整400轮mean_Q和Q-IQR、训练曲线；同13738个验证自主状态×2动作。
- `evaluate20.py`：冻结在线微调用的20prompt和同GPU baseline，测试K23/K29/K35；
  记录PSNR/SSIM/LPIPS、组件时间/TFLOPs，并配对旧e391后段skip；不测VBench分数。
- `test_profile.py`：默认合同不变、显式参数边界、真实CPU训练/保存参数元数据验收。

运行：`python experiments/dynamics128_aggressive_iql_v1/run.py --gpu 0`。
结果根：`/mnt/hdd/xiongyuxiang/tmp/exp/ours21_dynamics128_aggressive_iql_v1/`，
`training/`为400轮日志和配置，`analysis/`为曲线与post300选点，`logs/`为进程日志。
所有权重位于`/mnt/hdd/xiongyuxiang/tmp/models/ours21_dynamics128_aggressive_iql_v1/`。
`evaluation20/`保存60个候选的三项质量、组件性能、配对e391结果和后段skip统计。
选点仅使用e301–399双侧验证actor一致率与Q漂移，测试集/视频质量不参与选点。
