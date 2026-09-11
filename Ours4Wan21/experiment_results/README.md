# Experiment results

Generated results are stored under `/mnt/hdd/xiongyuxiang/tmp/exp`.
Only symlinks to those external archives belong here.

- `ours21_sea7_e328_vbench200_4gpu_v1/`: SEA7 e328, full 200 prompts, K23/K29/K35,
  four-GPU matched baselines, VideoMetrics and VBench 16 dimensions; see status.json.

- `wan21_increase_e391_trace_comparison_v1/`：手工increase五档与dynamic e391四档，同4prompt/档，36条实测trace按skip数升序对比；PNG/SVG及完整CSV。

- `ours21_online_eval20_from_vbench50_random42_v1/`: 在线微调固定20prompt参考包；原VBench50的seed42子集与同GPU baseline视频/计时/trace软链接，三档K23/K29/K35，无VBench评分，含来源与CPU验证。

- `ours21_sea7_e328_online_8h_v1/`: 本次SEA7 e328四卡2轮在线微调，含20×3末轮评测，关闭VBench。
- `ours21_sea7_e328_online_8h_v1_setup/`: 训练池、train-only实测K映射、来源哈希及8小时轮数估算。

- `ours21_training_late_skip_audit_v1/`：2400条train后段skip覆盖审计；同K34强集中5/212、K36后25步23次skip为0/237；HTML、CSV、notebook和全量来源校验。

BLOC v2 input audit and extraction microbenchmark: `ours21_bloc_features_v2_audit_benchmark_v2/`; compact feature cache: `ours21_bloc_features_v2_features_v1/`. These are not closed-loop quality results.

- `ours21_increase500_v1/`：新增increase500数据增强，链接HDD归档；计划400/50/50，混合后2800/350/350。进展以归档STATUS与完成标记为准。

- `ours21_increase500_iql2_v1/`：Increase500采集后自动合并3500轨迹、两组Dynamics128 IQL训练/选点、原20prompt三档测试，无VBench score。

- `ours21_cnn_compute_estimate_v1/`：候选三分支CNN逐层算量；非最终架构或实测速度。

- `ours21_cnn_training_benchmark_v1/`：CNN短跑计时cache、逐轮结果与估时；非质量训练。

- `ours21_cnn_mixed3500_v1/`：三分支CNN正式四组cache/训练/选点，状态在STATUS.json，模型存统一models目录。

- `ours21_cnn_vbench20_v1/`：CNN四组同20prompt自动评测，等待200轮及e170后选点；逐视频封存、同GPU baseline复用。

- ours21_g1_ablations_mixed3500_v1/：HDD三组G1方法消融训练、cache及选点结果软链接。
