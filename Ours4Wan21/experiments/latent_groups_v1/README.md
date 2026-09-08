# Wan2.1：两个标量对照与十个 latent 实验组

`run.sh`按阶段顺序运行12组；第二参数可指定单组。`CANDIDATES.md`定义全部公式与时序；`validate.py`是CPU合成数据端到端验收。结果写外部exp，模型写workspace/models，项目experiment_results仅保存链接。不会启动本机正式实验。

先按父README第1节配置远端环境，按第3节冻结已有random3000得到`COLLECTION_ROOT`和`SELECTION_DIR`。所有12组必须使用同一selection；scalar5/SEA7不读取latent。训练仍是400epoch、batch256、3层256、seed42、Exact-K与终局PSNR，保留原prompt split、仅train拟合normalizer。

```bash
cd "$OURS_PROJECT"
export SUITE_NAME=ours21_random3000_12groups_v1
# 一次读取3000×50个候选input latent，提取全部10组；不读取baseline latent。
# CPU可运行；GPU提取可设FEATURE_DEVICE=cuda并指定一张CUDA_VISIBLE_DEVICES。
export FEATURE_DEVICE=cpu
bash experiments/latent_groups_v1/run.sh features
bash experiments/latent_groups_v1/run.sh cache all
CUDA_VISIBLE_DEVICES=0 bash experiments/latent_groups_v1/run.sh train all
CUDA_VISIBLE_DEVICES=0 bash experiments/latent_groups_v1/run.sh analyze all
```

`features`需要原trace的50个`step_records`和它们指向的FP16 `[16,21,60,104]`候选输入latent、真实sigma。不要使用输出latent或baseline latent。原absolute路径需在远端有效，与现有采集cache合同一致。特征提取不重新生成/解码视频，按步流式读取，不把3000条原latent全部载入内存；中间只保存小特征及原文件SHA。该步骤有大量磁盘读取和FFT/quantile运算，CPU可能较慢，尚无远端实测耗时。

若特征提取中断，可用`RESUME_FEATURES=1 bash experiments/latent_groups_v1/run.sh features`；已完成轨迹会核验输入SHA并复用。完整feature cache拒绝重复写入。`cache/train/analyze`均为新目录入口，训练尚无resume；重复实验改`SUITE_NAME`。每组训练cache约 `150000 × 输入维数 × 4 × 2` bytes（state/next_state），最大1031维约1.24GB，另需训练内存及各epoch权重空间。各组第一层参数量不同，不能宣称同参数比较。

按组并行时，每个进程显式指定不同GPU和不同组；不要重复启动同组目录。示例：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/latent_groups_v1/run.sh train sea7_cache_update192
```

`analyze`为每组生成全400epoch指标曲线、post300验证集动作/Q稳定性普查、排名及`selected_model.pt`。沿用父README第5节规则，test和VBench不参与选择。

先按父README第6节生成或确认同协议、同prompt、同物理GPU的VBench200 baseline，然后：

```bash
export SKIP_BUDGET=25  # 示例操作点，不代表已标定speedup
# VBENCH_BASELINE、FLOPS_PROFILE、WAN21_ROOT、CHECKPOINT_DIR、OFFICIAL_CODE沿用父README。
CUDA_VISIBLE_DEVICES=0 bash experiments/latent_groups_v1/run.sh vbench all
CUDA_VISIBLE_DEVICES=0 bash experiments/latent_groups_v1/run.sh evaluate all
```

每个K对应12×200个候选视频。需要其他K时显式更改SKIP_BUDGET，结果目录区分K。相同K不等于相同实测速度，FFT/quantile等特征成本会影响完整generate latency；最终按实际速度与质量比较。逐组质量及性能输出与父README第7节一致：PSNR/SSIM/LPIPS、16维VBench200聚合、完整latency及T5/DiT/VAE计量。

推理每步在cond原生forward进入patch embedding之前捕获当前raw latent，观察一次、按实际动作commit一次；uncond共用该向量。所有强制步也维护历史，只有第0步清空，第32步不清空，第49步保留当前latent特征但强制recompute。在线特征FP16→FP32量化与原archive一致，不改模型/scheduler输入。

每视频`timings/*.json`保留`predictor`网络TFLOPs、network CUDA时间、decision wall，并新增`latent_feature`的50次逐步/合计提取wall时间（含输入处理至CPU向量返回）。`components.json`/`performance.json`保留特征总时间，trace也记录；这些时间已包含在DiT/generate，不能再次相加。网络FLOPs按各组实际输入维数自动计算；feature FFT/quantile/reductions尚无完整FLOPs估计，明确为null，不能把小网络TFLOPs当整个方法TFLOPs。

各组名称：`scalar5`、`sea7`、`sea7_dynamics_raw_sea128`、`sea7_cache_update192`、`sea7_local_drift1024`、`sea7_spatial_gradient96`、`sea7_channel_geometry240`、`sea7_distribution256`、`sea7_spectral_drift512`、`sea7_spectral_phase512`、`sea7_spectral_shape576`、`sea7_spectral_dynamics1024`。
