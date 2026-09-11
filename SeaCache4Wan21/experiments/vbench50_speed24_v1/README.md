# SeaCache 同随机50 prompt 名义2.4×四卡补测

复用Dynamics128 e391随机50实验的同GPU baseline与13/13/12/12分片，seed42。
沿用已完成五提示词2.4×标定映射threshold=0.3026171698，不重新扫描或按质量调参。
固定Wan2.1-1.3B、832×480、81帧16fps、UniPC50、shift5、CFG5、BF16、全部驻GPU。

- `run_suite24.py`：冻结来源与baseline哈希，四卡生成50候选→VideoMetrics三指标→汇总。
- `audit_previous.py`：只读复核已结束3.6×实验的全部配对、哈希和逐视频均值。
- `test_suite24.py`：固定阈值、分片、汇总与参数合同CPU测试。

使用Ours4Wan21既有3.6×管线的SeaCache worker与验收函数，保持实现一致，原脚本不修改。
每GPU一次native warmup仅稳定计时，不作新reference；不运行VBench。结果同时列实测速度。
结果归档`/mnt/hdd/xiongyuxiang/tmp/exp/seacache_wan21_vbench50_speed24_4gpu_v1/`，
本项目experiment_results以symlink引用。prompts/、references/、shards/、analysis/、logs/分开。

Wan2.2环境Python运行`run_suite24.py --prepare-only`，再以tmux运行`--resume`。
共享四卡锁避免与其他遵守锁的任务冲突；不完整输出拒绝覆盖。
