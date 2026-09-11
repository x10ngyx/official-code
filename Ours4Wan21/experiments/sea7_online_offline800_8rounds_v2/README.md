# SEA7 e328 four-GPU online fine-tuning, 8 rounds

Superseded by `../sea7_online_offline800_8rounds_a25_v1/` following the user request for A2–A3 aggressiveness. This default-profile run stopped before any completed candidate or online update; retain as historical configuration.

Current user-authorized launch after the priority-task pause. `run.sh prepare` freezes inputs with the current shared source; `launch.sh` runs persistently with logs. Previous v1 stayed unstarted and is preserved because contracts.py gained offline IQL profile helpers during the pause. Online algorithm and experiment inputs remain identical.

8 rounds × 100 trajectories, uniform draws with replacement from the 800 offline train prompts; native training baselines reused. Train targets 1.5–3.5×; 5 critic warmup + 20 joint epochs, e11–20 selection and accumulated replay. Start at original SEA7 e328. R4/R8 evaluate the same 20 VBench50 prompts at 1.8/2.4/3.0× (K23/29/35), reuse evaluation baselines, PSNR/SSIM/LPIPS and component timing/TFLOPs, no VBench scoring.

Results: `/mnt/hdd/xiongyuxiang/tmp/exp/ours21_sea7_e328_online_offline800_8rounds_v2`, symlinked in experiment_results; weights under the matching models directory. Reuses existing offline800 setup and reference bundles. Prior four-GPU timing estimate is 15.10 hours including R4/R8 evaluation, not a hard deadline.
