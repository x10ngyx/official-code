# 3.6×双方法trace图合同

问题：同随机50prompt下SeaCache/e391在50个采样step的复用/重算位置有哪些差异？
图族Matrix & Cohort，paired binary action matrix；按prompt ID排序，每条两行，
上SeaCache下e391，step0–49不省略。描述动作位置，不根据与PSNR并置推断因果。
数据100正式视频×2CFG×50step=10000条；标定8视频不入图。先验证两个CFG分支
决策覆盖、reuse/recompute path和实际blocks，再在分支一致时无损合并为5000格。
分支不同或缺失时拒绝输出，不填补。逐视频PSNR及名义/实际速度来自已完成正式结果。
继续用户选择的静态trace图片交付，Matplotlib PNG/SVG；全图100行，阅读版各25prompt/50行。
hard two-root cap：SeaCache复用蓝#2563A6，e391复用橙#C87524，重算开放白色。
行顺序、明确方法标签和填充/开放编码确保不只依赖颜色；prompt之间加间隔。
完整prompt及原始双分支记录保留CSV；图内不截断采样step，右列PSNR单位dB。
输出原3.6×归档analysis/trace_comparison_50/；通过已有项目symlink访问。
QA：数值与来源SHA、全部10000条blocks匹配、实际打开两张阅读版PNG，标签/图例无裁切。
