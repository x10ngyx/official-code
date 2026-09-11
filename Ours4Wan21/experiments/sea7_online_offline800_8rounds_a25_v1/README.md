# SEA7 online IQL, midpoint of A2/A3

Superseded by `../dynamics128_online_e391_8rounds_a25_v1/`: user explicitly selected Dynamics128 e391. This SEA7 task was stopped before completed candidates or online updates; retain historical inputs.

User requested A2–A3-level aggressiveness for the ongoing four-GPU, eight-round online run. Named profile `aggressive_a2_a3_v1`: tau=0.85, beta=2.5, weight_max=75, the arithmetic midpoint of the completed offline A2=(0.8,2,50) and A3=(0.9,3,100). This midpoint itself was not tested in the previous suite. Default online settings were (0.7,1,20). Only these three IQL settings change; larger beta multiplies batch-standardized advantages, without altering their normalization.

`run.sh prepare` freezes inputs; `launch.sh` starts the persistent pipeline. Start at SEA7 e328, 8 rounds × 100 trajectories from offline train800 with native baseline reuse, targets 1.5–3.5×. Five critic warmup and 20 joint epochs per round, selection e11–20, actor/critic LR 4e-5/1e-4, gamma1, cumulative replay. R4/R8 evaluate the same 20 VBench50 prompts at K23/29/35 with paired PSNR/SSIM/LPIPS and component metrics, no VBench scoring.

Results: `/mnt/hdd/xiongyuxiang/tmp/exp/ours21_sea7_e328_online_offline800_8rounds_a25_v1`, project experiment_results symlink; models directory uses the same name. Original default-profile v2 stopped in R1 before any completed candidate or online update; retain its artifacts. Existing timing model (~15.1 hours) remains a rough estimate; no fixed inference speed or quality gain is promised by the IQL profile.
