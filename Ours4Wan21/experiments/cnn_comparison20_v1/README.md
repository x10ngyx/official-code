# CNN同20prompt历史对照

report.py等待CNN评测COMPLETE，验证结果和历史证据哈希，核对同20prompt及baseline时间，重新汇总四组CNN与e391、SeaCache、Increase。输出到HDD ours21_cnn_vbench20_v1/analysis/comparison20/，包含报告、逐视频/配对CSV及验证记录。SeaCache低档缺项，实速不同的名义档位须保留速度差。

用户明确以RGB PSNR对比：rgb_report.py通过官方VideoMetrics的--metrics psnr四CPU分片汇总全部240候选，再校验1920个sealed文件、同20 baseline SHA及历史来源，输出analysis/rgb_comparison20/。完整SSIM/LPIPS管线独立继续。report.py保留为完整指标落盘后可运行的通用汇总，当前主交付为RGB报告。
