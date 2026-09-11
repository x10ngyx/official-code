# SEA7 e328 online fine-tuning from offline train prompts

Historical 4-round configuration, paused and superseded by `../sea7_online_offline800_8rounds_v1/` after the user requested 8 rounds. Do not resume this archived run.

`prepare_inputs.py` seals native baseline references for exactly 800 offline training prompts and estimates complete runs against an eight-hour budget. `run.sh prepare` freezes the run; `launch.sh` executes it with persistent logs. Outputs are under `/mnt/hdd/xiongyuxiang/tmp/exp/ours21_sea7_e328_online_offline800_8h_v1`, linked in `experiment_results/`; weights are in the matching models directory.

Four rounds, 100 uniform prompt draws with replacement per round (independent deterministic prompt/target streams), targets 1.5–3.5×, 5 critic warmup + 20 joint epochs per round. Each round restores the selected checkpoint; replay accumulates. No training native baseline generation: all 800 reference videos already exist. Native warmup remains necessary per worker. Original shard GPU indices are reconstructed from the source launcher; archived baseline UUIDs were not recorded, and this limitation is retained in reference metadata.

Evaluation at round 4: 20 prompts sampled from the prior VBench50, nominal 1.8/2.4/3.0× (K23/29/35), offline e328 versus selected online checkpoint, reused prior evaluation baselines. PSNR/SSIM/LPIPS and component latency/TFLOPs, no VBench scoring.

Estimated four rounds including final evaluation: 7.982 hours; five rounds: 10.430 hours. Estimate conservatively places the slowest shard last in serialized loading/warmup; IO and actual runtime can vary. The old two-round run was stopped before weight updates and retained separately.
