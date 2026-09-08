# experiment_results

Symlinks only, pointing to `/all/yiran07-disk3/huteng_data/exp`; no local copies of large artifacts.

- `magcache4wan22_official_core_validation_20260905/`: small WanModel CUDA/BF16 official parity, 100 calls; not full A14B.
- `magcache4wan22_batch_scan_validation_20260905/`: 19 CPU tests, baseline source audit, syntax/SHA checks and four CLI dry plans; no measured target speedups.
- `magcache_public_configs_audit_20260905/`: primary-source configuration/speed audit; separates Wan2.2 from Wan2.1 and wall latency from NFE proxies.
- `magcache4wan22_targeted_scan_plan_20260905/`: provisional R/E/K starts, historical timing heuristic with source SHA, 23 CPU tests and fixed-R/K E scan plans; no measured MagCache A14B speedups.
- `magcache4wan22_targeted_e_scan_gpu123_20260905_165530/`: GPU1/2/3 assigned to 1.8x/2.4x/3.0x; one measured shared FLOPs profile, fresh same-GPU baselines and 187 planned measured videos. See its status.json for live progress.
