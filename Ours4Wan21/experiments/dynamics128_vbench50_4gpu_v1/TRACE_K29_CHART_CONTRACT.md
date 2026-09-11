# e391 K29 trace 图合同

- 问题：固定随机50条 prompt 的50个采样step，e391在哪些step复用或重算？
- 图族：Matrix & Cohort；二元动作矩阵，按sample_id排序，保留全部50×50格。
- 结论边界：描述实际动作，不从动作与PSNR的并置推断因果。名义2.4×与实测速率分列。
- 数据：冻结清单50条、每条cond/uncond各50次决策；与实际执行blocks逐项核对。
  两分支完全一致才合并为一行，否则拒绝输出；不得填补缺失trace。
- 输出：沿用此前静态trace图片交付，Matplotlib PNG/SVG；全50行及两个25行阅读版。
- 颜色：single-root preferred，蓝色#2563A6为复用、白色开放填充为重算；浅灰边界。
- 信息：step0–49，prompt ID，右侧同视频PSNR(dB)；CSV保留完整prompt和双分支原始决策。
- 路径：既有HDD实验结果的analysis/trace_k29_e391_50prompts/，项目既有symlink可访问。
- QA：50×100条分支决策、K29、实际blocks、checkpoint配置、汇总均值及PNG实际图片检查。
