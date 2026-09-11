# Ours4Wan21 experiments

正式方法 **CNN+G1** 由顶层 [main.py](../main.py) 统一入口直接调用，定义见 [METHOD.md](../METHOD.md)。

- `cnn_mixed3500_v1/`：正式G1网络、特征、训练与选点；同目录G2/G3/G4为对照。
- `cnn_vbench20_v1/`：正式CNN策略加载、因果特征和Exact-K推理接入；四组launcher用于历史比较复现。
- `g1_ablations_mixed3500_v1/`：SEA7、16×20空间CNN和G1特征MLP消融，均不替代原8×8 CNN+G1。

以下RL/latent/online入口保留历史实验用途。


- rl_training_v1/: frozen local offline IQL launcher.
- rl_inference_v1/: fixed Wan21 resident inference launcher.
- rl_validation_v1/: synthetic CPU end-to-end validation.

Collection experiments remain under ../data_collection/experiments/. All results live in the external experiment root; model weights live under models/.

- rl_cache_v1/: build scalar caches from the frozen 3000-random selection.
- rl_analysis_v1/: all-epoch metrics/plots and post300 validation-only checkpoint selection.

- `latent_groups_v1/`: remote scalar5/SEA7 controls plus ten individual SEA7+latent groups; one-pass feature cache, training, selection, VBench and CPU validation.
- `online_finetuning_v1/`: complete remote offline-to-online pipeline, frozen 5+20 epochs, e11–20 actor-agreement selection, 20 prompts from prior VBench50 at three targets every four rounds, reused same-GPU baselines and no VBench scoring, and CPU validation.
- `vbench5_speed_targets_v1/`: post-training fixed calibrated K23/K29/K35 matched five-prompt tests for nominal 1.8x/2.4x/3.0x, with component metrics and a portable technical report.
- `sea7_vbench200_4gpu_v1/`: queued SEA7 e328 full 200-prompt K23/K29/K35 evaluation, four generation shards, paired video metrics and 16-dimension VBench200 scores.
- `dynamics128_vbench50_4gpu_v1/`: Dynamics128 e391 random 50-of-200 prompts (sampling seed42), four matched baseline shards, K23/K29/K35, video metrics and VBench50 subset scores.
- `vbench50_speed36_pair_v1/`: reuse the same50 native references; calibrate/verify nominal3.6x for SeaCache and Dynamics128 e391, then100 formal candidates; PSNR/SSIM/LPIPS only.

- `sea7_online_8h_v1/`: SEA7 e328四卡八小时预算任务；train-only实测K先验、固定均衡分片、2轮在线训练和末轮20×3质量/性能对比。

- `training_late_skip_audit_v1/`：读取e391实际训练输入并核对3000条原始trace，统计同K下后段集中skip覆盖，输出技术报告和notebook。
- `bloc_features_v2/`: causal compact BLOC feature audit, extraction microbenchmark, four-worker cache preparation and equal-budget training; closed-loop selection required.

- `dynamics128_aggressive_iql_v1/`: Dynamics128 tau=.9/beta=3/weight_max=100 从头400轮，完整Q诊断与post300稳定选点，接续同在线20prompt三档质量/性能测试，无VBench评分。

- `iql_aggressiveness_2x4_v1/`: SEA7与Dynamics128四档激进IQL、各400轮、四卡八组；自动post300选点/Q图和冻结在线10prompt三档240候选，无VBench评分，接管旧Dynamics128 A3训练。

- `iql_aggressiveness_overlap_v1/`: 剩余SEA7 A3单卡训练与三卡就绪组评测并行，追加第八组后补齐原10prompt三档240候选；原suite配置和训练模型保持。

- `iql_training_readout_v1/`: 已完成八组及原组的400轮训练/Q曲线即时读出，独立PNG/SVG和来源CSV，不依赖视频质量完成。

- `iql_increase_traces_v1/`：八组训练与手工Increase实测50步trace图，同155配对及每组4条展开，仅读取既有结果。

- `online_r1_increase_traces_v1/`：R1全部100条与5条Increase按skip排序，三张完整路径图，仅读取已完成结果。

- `increase500_iql2_v1/`：Increase500采集后自动合并3500轨迹、两组Dynamics128 IQL训练/选点、原20prompt三档测试，无VBench score。

- `increase500_trace20_v1/`：混合训练两组与原e391、Increase的20条排序trace比较图，含来源核验及PNG/SVG/CSV。

- `increase500_final_comparison_v1/`：mixed3500两组与同20prompt原e391/SeaCache最终比较、配对差值与来源核验。

- `increase500_yuv_psnr_v1/`：同20prompt的220视频YUV 6:1:1 PSNR重算与RGB一致性验证。

- `increase_vbench20_v1/`：直接Increase同20prompt三档生成、质量/YUV与审计自动管线。

- `cnn_compute_estimate_v1/`：三分支候选CNN相对原MLP的静态Conv/Linear MAC与参数量计算，无GPU训练。

- `increase20_all_methods_readout_v1/`：Increase与e391/SeaCache/两新训练组同20prompt最终读出和全量核验。

- `increase18_paired_traces_v1/`：1.8档Increase与两训练组60条同prompt trace配对PNG/SVG及核验。

- `cnn_training_benchmark_v1/`：三分支CNN完整IQL短跑，真实轨迹cache重复扩展到全epoch行数，四卡并行计时。

- `cnn_mixed3500_v1/`：正式四组三分支CNN：mixed3500全量cache、保守IQL各200轮、验证集post170选点与训练读出，可恢复。

- `mixed3500_q_statistics_v1/`：两组缓存epoch300–400的mean_Q/Q-IQR对比及选点标记。

- `mixed3500_q_accuracy_v1/`：选中critic对自身实测轨迹RGB PSNR的误差、CPU回放与散点图。

- `cnn_vbench20_v1/`：四组CNN选点后自动同VBench20、K23/29/35测试，共240候选，RGB/YUV611与组件计量。

- `cnn_comparison20_v1/`：四组CNN与同20prompt的e391、SeaCache、Increase配对比较，含来源校验和实速差。

- g1_ablations_mixed3500_v1/：选定G1后，SEA7、16x20空间G1 CNN、G1特征MLP三卡各200轮中间档IQL训练及选点。
