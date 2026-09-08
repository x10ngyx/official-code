# DiCache VBench200 corrected balanced run

Active thresholds: **0.075 / 0.172 / 0.396**, retention=0.2, probe1. This supersedes the historical 0.072 middle target in `../vbench200_reuse_gpu123/`; old scripts are kept unchanged for provenance. All other fixed Wan2.2 inference parameters and shared baseline reporting policy remain unchanged.

- `prepare.py`: validates the 200 shared baseline artifacts and scan equality evidence, creates new target plans, imports only completed and fully validated unchanged 0.075/0.396 runs by directory symlinks. Every 0.172 candidate is generated anew; 0.072 is excluded.
- `rebalance_gpu123.py`: adapts the MagCache balanced runner; assigns remaining target/prompt pairs exactly once using longest-processing-time greedy assignment. DiCache uses observed dynamic probe/full traces, not a static schedule. Empty middle target initially uses the adjacent 0.175 scan mean as an estimate only. Each GPU receives all three targets with near-equal estimated total seconds.
- `run_gpu123.sh`: launches the coordinator in conda `wan2.2`, setting all four BLAS/OpenMP thread limits to 1.

Each GPU loads one persistent pipeline, runs one unmeasured full warmup, then handles its assigned videos across targets. Model initialization is staggered by verified worker PID readiness. Cross-GPU reuse validates GPU1/2/3 lifecycle, original protocol/settings, run/video/timing/trace SHA and probe-aware FLOPs. The historical GPU suffix in target directories indicates the evaluation GPU; it does not restrict generation placement.

After all workers finish, three target evaluators run concurrently, each calculating PSNR/SSIM/LPIPS and baseline/candidate standard 16-dimensional VBench. Only then is root COMPLETE.json written. All baseline videos are reused; no measured baseline is generated. `--resume` skips validated completed candidates. Source hashes are frozen in the assignment.

```bash
/home/huteng/yes/envs/wan2.2/bin/python prepare.py --previous /all/yiran07-disk3/huteng_data/exp/PREVIOUS --output-dir /all/yiran07-disk3/huteng_data/exp/NEW
bash run_gpu123.sh --output-dir /all/yiran07-disk3/huteng_data/exp/NEW --resume
```

Results and import/assignment evidence reside in the external output; `../../experiment_results/` indexes it by symlink. Tests: `../../tests/test_balanced_vbench.py` covers exact coverage/load balance and rejection of wrong threshold or GPU lifecycle. Preparation additionally checks the real 600-item coverage and every imported result.
