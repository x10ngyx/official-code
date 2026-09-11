# Increase500 + random3000：两组Dynamics128 IQL自动管线

用户要求当前采集结束后合并数据、训练保守/激进两组、自动选择checkpoint并用之前相同20个VBench prompt测试。
本管线等待Increase500的COMPLETE及mixed selection通过验证，不改变采集进程。

- `pipeline.py prepare|run`：冻结输入并等待采集，特征准备、训练cache、双组400轮训练、选点、四卡测试、三指标及报告。
- `common.py`：本suite路径与既有训练/trace/来源审计函数。
- `generate_worker.py`：沿用已验证的resident推理worker，绑定同物理GPU baseline与选中checkpoint，支持逐视频恢复。
- `test_pipeline.py`：测试阶段依赖、两组参数差异、同20prompt/三档120条任务合同。

训练模式两组均为sea7_dynamics_raw_sea128（128 Dynamics + 7 SEA）。从头训练400epoch、batch256、seed42、lr1e-4、3×256 MLP；
保守组baseline：tau=.60,beta=1,weight_max=20；激进组aggressive_v1/A3：tau=.90,beta=3,weight_max=100。
其他训练参数与原400轮一致。混合数据3500轨迹=2800/350/350（train/val/test），175000 transitions；
每条50步，只有最终PSNR作为terminal reward。训练normalizer只由混合train拟合，两组共用同一cache。

复用原3000轨迹的小型特征文件，按trajectory ID匹配到新selection序号、逐文件SHA检查；仅500新增轨迹需要提取。
沿用原十组feature文件合同，新增500计算合同要求的全部特征，但只构建/训练Dynamics128模式。
四worker还会重验已有3000特征的原latent SHA和动作/trace身份，不把baseline latent用作状态。

两组并行训练在GPU0/1；随后e301–399双侧actor一致率.96→.95→最大观察值门槛，最小化Q漂移/IQR选点；
完整复用现有analyze_training.py，验证集选点，test和VBench结果不参与选择。

评测直接读取ours21_online_eval20_from_vbench50_random42_v1的完整20条，不重新抽样。
固定K23/29/35（名义1.8/2.4/3.0×），每组60、总120候选；复用同UUID native baseline，无新增baseline存档。
测实速、T5/DiT/VAE/预测器时间与TFLOPs、PSNR/SSIM/LPIPS，9720对帧；不计算VBench score。
输出逐视频CSV、六条件CSV和REPORT.md；COMPLETE要求训练、选点、120视频和质量证据通过审计。

输出根：/mnt/hdd/xiongyuxiang/tmp/exp/ours21_increase500_iql2_v1/；项目experiment_results建立软链接。
模型权重：/mnt/hdd/xiongyuxiang/tmp/models/ours21_increase500_iql2_v1_{conservative,aggressive}/。
共用GPU锁等待采集释放后执行，进度STATUS.json，异常FAILED.json。完整阶段可跳过；训练中途异常会明确停止，
不覆盖未完成训练目录或静默重训。所有数值输出必须以实际运行结果为准。

运行恢复：两组训练完成后，系统用户态驱动580.178.04与运行内核580.173.02不匹配导致选点CUDA804。`resume_compatible_runtime.sh`使用结果runtime/中的本地匹配库续跑，仅影响本任务，不改系统或重训。包/库SHA见runtime/COMPATIBILITY.json；失败分析保留在recovery/driver_library_mismatch。
