# Experiments

Each subdirectory is a standalone experiment entry. `performance_t2v_a14b` profiles/aggregates component FLOPs; `paired_benchmark` and `vbench200_t2v` generate paired videos with persistent workers and evaluate quality; `parameter_scan` calibrates threshold/retention against measured speedups; `smoke_official_core` checks small real CUDA/BF16 models. All results belong under `/all/yiran07-disk3/huteng_data/exp`, indexed by symlinks in `../experiment_results`.

`targeted_threshold_scan` fixes official retention/probe and adaptively calibrates the three measured speed targets, then evaluates selected quality.

GPU1/2/3、固定retention=.2、约1.5×–3.5×范围的阈值扫描入口：`experiments/speedrange_gpu123/run_scan.sh`（实验目录内见`speedrange_gpu123/README.md`）。

`vbench200_reuse_gpu123/` validates scan/shared baseline equality and runs three 200-candidate targets on GPU1/2/3 with reused baseline videos.

当前修正后的正式VBench200入口：`experiments/vbench200_balanced_gpu123/`（th=.075/.172/.396、三卡按预计耗时均衡分配、复用已验证结果和baseline）。历史`vbench200_reuse_gpu123`的.072中档已被用户更正，保留原入口仅用于溯源。
