# SeaCache / Dynamics128 e391 同50prompt名义3.6×补测

- `plot_traces.py`：CPU读取正式100视频trace，校验10000条双CFG决策与实际blocks，
  导出完整和分半PNG/SVG对比图、原始逐step CSV；图合同见`TRACE_CHART_CONTRACT.md`。

统一脚本位于`../../../Ours4Wan21/experiments/vbench50_speed36_pair_v1/`，不复制两套入口。
复用已完成Dynamics128随机50实验的同GPU baseline；固定协议、四卡13/13/12/12分片。
SeaCache用原始10prompt实测0.50/0.60包围点插值，threshold≈0.592402；Dynamics128 e391
以同50条K23/29/35实际latency拟合选择K38。每卡1条、每方法共4条速度验证通过后锁定
参数正式各跑50条，仅PSNR/SSIM/LPIPS，不跑VBench，不按质量筛选或自适应补测。
5项CPU合同/100配对最终汇总测试通过。具体脚本、标定政策与恢复说明见统一README。

结果在`/mnt/hdd/xiongyuxiang/tmp/exp/wan21_vbench50_speed36_seacache_dynamics128_e391_v1/`，
本项目及Ours4Wan21的experiment_results均使用symlink。`calibration/`是8条独立速度
验证；`shards/`是100条正式候选；`references/`链接原baseline；`analysis/`是最后汇总。
