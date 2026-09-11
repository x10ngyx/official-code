# SeaCache / Dynamics128 e391 随机50 prompt，名义3.6×补测

复用`ours21_dynamics128_e391_vbench50_random42_4gpu_v1`冻结的50条prompt和每卡baseline，
保持四卡13/13/12/12分片、模型协议、seed42与同物理GPU配对。不重新生成baseline。
每进程一条native warmup仅用于稳定计时，不作为新的reference，也不计入正式速度。

- `run_pair.py`：冻结来源/参数/标定子集；8条速度验证→参数锁定→100条正式候选→
  PSNR/SSIM/LPIPS及计量汇总。`--prepare-only`冻结，`--resume`执行。
- `sea_worker.py`：corrected filtered-boundary SeaCache驻留GPU批量推理及实际trace计量。
- `test_pair.py`：插值、整数K选择、分片/速度容差和真实来源合同测试。

3.6×旧mapping声明域只到3.5，不能直接外推。使用原始10prompt threshold_summary实测
0.50→3.231601×、0.60→3.630293×的包围点内插，初始threshold=0.5924018125325419。
Dynamics128使用刚完成的同50prompt K23/29/35的latency拟合`t=a+bK`，目标解K≈37.853，
按预测完整速度距3.6最近选择整数K38（K37≈3.398×，K38≈3.637×）。
在每个baseline shard用独立Random(360+gpu).choice固定1条标定prompt，各方法4条速度
验证（共8条，质量盲选）。两方法各自baseline/candidate总时间比须在3.6±5%内，否则
停止并报告，不自动扩大扫描、不调整正式K、不按质量选点。通过后全50条使用同一阈值/K。
标定8条单独归档，不计入正式100条质量与速度；不复用标定计时冒充正式测试。

固定Wan2.1-1.3B、832×480、81帧16fps、50-step UniPC、shift5、CFG5、seed42、BF16、
无任何CPU/model offload。只评测VideoMetrics PSNR/SSIM/LPIPS，不跑VBench分数。
报告完整generate速度/耗时、DiT/T5/VAE及predictor时间和TFLOPs、50步trace。
3.6×为名义目标，不宣称两方法实际速度或执行计算量严格相等。

外部归档`/mnt/hdd/xiongyuxiang/tmp/exp/wan21_vbench50_speed36_seacache_dynamics128_e391_v1/`，
Ours4Wan21和SeaCache4Wan21的experiment_results均通过symlink引用，不复制视频。
`prompts/`冻结50条及4条标定prompt；`references/`链接旧baseline视频；`calibration/`
保存8条验证、计量与CALIBRATED.json；`shards/gpu*/{seacache,dynamics128}/`保存正式
候选/质量，`analysis/`最终CSV/Markdown/验收，`logs/`运行日志。部分目录拒绝覆盖。
