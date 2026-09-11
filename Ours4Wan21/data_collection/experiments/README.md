# Experiments

- `random_threshold_collection_v1/`: plan, baseline, candidate, publication,
  and audit launcher.
- `seacache_threshold_collection_v1/`: matched 1,000-prompt/3,000-candidate
  fixed-threshold SeaCache collection launcher.
- `seacache_speedup_calibration_v1/`: four-GPU, ten-sample fixed-threshold
  scan and linear speedup-to-threshold/skip-count calibration.
- `calflops_profile_v1/`: one-time Wan2.1-T2V-1.3B real-shape DiT profile.

- `increase500_v1/`：复用原1000 baseline，按既有threshold均值近似估速，直接采集500条线性increase（不做新标定）；标准数据审计、原3000+500混合索引和VBench。
