# SeaCache 与 Ours 十二组五提示词对比

当前入口 `run_4gpu.py` 使用现有 SeaCache 标定映射，对名义 1.8/2.4/3.0×各运行5条
固定prompt（001/016/056/135/159）。15个组合按索引模4分给GPU0/1/2/3，共4/4/4/3条，
复用Ours五提示词归档中相同物理卡native baseline。每卡一次不计入测量的完整native warmup，不重复训练，
不做自适应阈值/K补测，不计算VBench评分。

使用仓库 SeaCache4Wan21 corrected filtered-boundary、独立CFG分支控制器，不是
learned SEA7。协议与Ours一致：Wan2.1-1.3B、832×480、81帧、16fps、UniPC50、
shift5、CFG5、seed42、BF16、单卡全部常驻GPU。计量复用相同profiler与Calflops文件。
不同方法实际加速可能不同；按名义档位并列并报告实测速度，不宣称严格等速。

结果位于 `/mnt/hdd/xiongyuxiang/tmp/exp/seacache_wan21_vbench5_ours_comparison_4gpu_v1/`，
本项目 `experiment_results/` 使用symlink。`config.json`冻结prompt、阈值、baseline
SHA和代码；`target_*/`保存生成/trace/计量/质量，`analysis/`保存13组共39行CSV和
Markdown对照；`logs/`保存运行输出，`status.json`与`COMPLETE.json`表明真实状态。

```bash
/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python \
  experiments/vbench5_ours_comparison_v1/run_4gpu.py --prepare-only
# 在同一环境后台运行，完整产物可复用，不覆盖部分目录
/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python \
  experiments/vbench5_ours_comparison_v1/run_4gpu.py --resume
```

`test_comparison.py`检查四卡15组合精确覆盖、标定插值、外推拒绝、逐视频`*_mean`字段及缺失样本拒绝。
旧 `run_comparison.py` 三卡入口仅保留作来源与公共校验函数；旧三卡任务在native warmup
期间按用户四卡要求取消，没有正式候选产物，旧目录完整保留，不应恢复旧三卡入口。

`validate_readout.py`只读复核完整39行对照：配置/视频哈希、4分片、同卡reference、
100次branch trace、实际DiT TFLOPs、总耗时比与质量均值，不修改完成标记。

`audit_psnr_trace.py`重新解码SeaCache/SEA7共30个视频、复算2,430帧PSNR并输出
3,000条双分支动作记录与SQLite。`prepare_trace_widgets.py`在核验CFG分支动作完全
相同后生成三张50列×10行的横向动作图配置；原始双分支证据不丢弃。

`plot_trace_comparison.py`直接导出中文标注的三面板50步PNG/SVG到上述审计目录的
`figures/`。运行前检查全部3,000行、执行blocks与动作一致性、CFG分支相等及复用总数；
输出包含逐视频PSNR、实际加速比与来源哈希，不调用GPU或重新运行推理。

`plot_psnr_comparison.py`从原始195条逐视频质量数据复核13方法×3档PSNR均值，导出
`analysis/psnr_13methods/`中的中文PNG/SVG、完整表、逐视频来源与校验哈希。
绘图规范见`PSNR_CHART_CONTRACT.md`；每格同时显示实际速度，中档不视为等速对照。
