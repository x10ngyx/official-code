# Increase500 数据增强

复用原随机离线集的1000个baseline；冻结500个不同prompt，各生成一条线性递增threshold轨迹。
`run.py`包含prepare、四卡直接采集、正式采集、完整性审计、标准表发布、混合数据索引和VBench。

- 固定50步路径 `threshold(step)=scale*(1+4*step/49)`，与既有increase的1:5首末比一致。
- 正式500条按原prompt split分为400/50/50，每个0.4×目标区间各100条，包含1.5×和3.5×端点。
- 用户最新要求不做新标定。复用原随机数据绑定的历史mean-threshold映射，均值0.098868–0.551644对应估计1.5–3.5×；scale=mean/3，使路径均值保持一致。目标速度明确为估计，实际加速独立记录。旧calibration产物与配置保留为停用来源，不进入正式500、混合3500或均值映射。
- 复用原collector/controller/runtime和VideoMetrics，保留50个FP16输入latent、100次CFG决策、
  完整推理时间、T5/DiT/VAE时间与TFLOPs、81帧PSNR/SSIM/LPIPS。审计逐个读取latent并验证训练episode可加载。
- 正式目标均匀覆盖不代表每条实测精确达标；实测加速单独保存。旧baseline只有卡序号来源、无物理UUID，
  维持原shard到GPU0/1/2/3映射，不能宣称完成了原baseline物理UUID校验。
- `mixed/`以小型completion记录和selection索引接入原3000+新增500条；原数据和大体积latent不复制。
  split为2800/350/350。加载器只为显式mixed schema放行3500，原3000约束保留。
- 原在线任务暂停状态保留；四卡共享锁与60秒空闲检查通过才启动。每卡模型常驻；native warmup不计入样本。
- 用户已确认其余流程并取消本任务VBench。结果根VBENCH_SKIPPED_BY_USER.json控制收尾标记skipped_by_user；DATA_COMPLETE与COMPLETE在数据审计/混合发布后生成。
- `finish_existing.py`接替旧orchestrator并运行原续采流程。已完成样本不重新生成；替换时4条未封存bundle已归档，标准collector重算未完成样本。

执行环境：`/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python`。
依次运行`run.py prepare`、`run.py run`；失败保留完整样本，`LAST_ERROR.json`给出异常。
产物：`/mnt/hdd/xiongyuxiang/tmp/exp/ours21_increase500_v1/`，项目experiment_results中有软链接。
`test_increase.py`验证抽样/split、目标覆盖、校准拒绝外推及混合加载器对泄漏/策略伪装的拒绝。
