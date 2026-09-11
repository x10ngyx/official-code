# Increase on the fixed VBench20

pipeline.py prepares a frozen60-candidate experiment and runs four resident workers, VideoMetrics RGB PSNR/SSIM/LPIPS, YUV611 PSNR and audited report. Native baselines and exact20 prompts are reused. No calibration or VBench score. Threshold mapping and start:end=1:5 exactly reuse increase500, so target speeds are approximate.
worker.py uses the existing ScheduledController and Wan2.1 fixed protocol. Four discarded increase warmups are excluded from candidate timing. Resume verifies sealed files; source hashes and same GPU UUID enforced.
Results: /mnt/hdd/xiongyuxiang/tmp/exp/ours21_increase_vbench20_v1. config.json frozen identities and thresholds; candidates/, quality_inputs/, quality/, yuv/, analysis/, logs/; COMPLETE.json seals final output. Launch.sh selects compatible NVIDIA libraries only if kernel remains580.173.02.
