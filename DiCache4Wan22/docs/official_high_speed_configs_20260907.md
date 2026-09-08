# 官方高加速配置核查

2026-09-07 14:48 CST

DiCache官方高加速配置已复核：可完整读取的arXiv v2中，Wan2.1-1.3B threshold=.2/probe1为2.45×；Hunyuan threshold=.2/probe1为2.90×；FLUX threshold=.4/probe1为3.22×；Hunyuan+SVG为组合3.08×。官方Wan脚本默认threshold=.08、retention=.2且quickstart未覆盖参数，未找到明确对应Wan≥3×的参数预设；也未找到官方为3×改retention=.1的依据。

本次为官方论文v2、项目页及仓库只读核查，无新增GPU/推理实验；OpenReview会议PDF被访问挑战拦截，未据此声称完成会议版全文核对。扫描配置未改，既有阈值扫描执行计划保持。

来源：
- https://arxiv.org/html/2508.17356v2 （4.1、Table1/2/3；论文未明确列出retention参数，不能把代码默认视为各表格行已证实配置）
- https://arxiv.org/abs/2508.17356 （当前列出的arXiv最新版为v2）
- https://raw.githubusercontent.com/Bujiazi/DiCache/main/WAN2.1/run_wan_dicache.py （threshold .08 / ret_ratio .2 / probe1）
- https://raw.githubusercontent.com/Bujiazi/DiCache/main/WAN2.1/run_wan_dicache.sh （仅prompt与seed覆盖，无高加速预设）
- https://github.com/Bujiazi/DiCache （SVG集成代码Coming Soon）

此前retention=.1只是本地预算分析建议，不是官方3×配置。不同模型结果不能作为Wan2.2阈值映射。未改源码、源锁、模型或扫描预设。
