# SeaCache 同随机50prompt名义3.0×四卡补测

复用已冻结随机50prompt、13/13/12/12分片及同GPU baseline，不重新推理reference。
沿用原calibrated mapping的3.0×阈值0.4384497621，不自适应调参或重新扫描。
固定Wan2.1-1.3B、832×480、81帧、16fps、UniPC50、shift5、CFG5、seed42、BF16，
模型组件全部驻GPU，关闭offload/扩写。每GPU一次native warmup排除计时，不作baseline。

- `run_suite30.py`：由已验收2.4×调度器机械派生，独立冻结来源，四卡50候选→
  VideoMetrics PSNR/SSIM/LPIPS→分组件计量、CSV/Markdown和最终COMPLETE；无VBench。
- `test_suite30.py`：阈值边界、真实分片、源码解析、模拟50配对质量/性能汇总CPU测试。
- `audit_speed24.py`：只读验收上一轮2.4×结果哈希、同卡配对、全部5000条执行trace、
  视频质量SHA与逐视频均值，不重新解码或计算质量。

保留原2.4×和3.6×冻结脚本不修改；共享同一SeaCache worker及质量工具。
先Wan2.2环境Python运行`run_suite30.py --prepare-only`，再tmux运行`--resume`。
共享四卡锁/GPU空闲检查，部分结果拒绝覆盖；nominal3.0×与实测速度分列。
结果`/mnt/hdd/xiongyuxiang/tmp/exp/seacache_wan21_vbench50_speed30_4gpu_v1/`，
项目experiment_results以symlink引用；prompts/、references/、shards/、analysis/、logs/分开。
