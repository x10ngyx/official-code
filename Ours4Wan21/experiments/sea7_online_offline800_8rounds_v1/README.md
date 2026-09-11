# SEA7 e328 online fine-tuning: 8 rounds, archived preparation

Superseded by `../sea7_online_offline800_8rounds_v2/` for the authorized launch. Shared contracts source changed while paused; v2 freezes current source with identical experiment parameters. The historical preparation below is retained.

User explicitly changed the total to 8 rounds after pausing for a higher-priority task. Do not start automatically. `run.sh prepare` performs CPU-only validation and freezes the new run. `launch.sh` starts execution only after a later user instruction to resume.

Source: SEA7 e328 offline checkpoint and its original cache. Each round samples 100 prompts with replacement from exactly 800 offline train prompts and reuses native baselines. Targets 1.5–3.5×, 5 critic warmup + 20 joint epochs, selection from e11–20, cumulative replay. Evaluate at R4/R8 using the frozen 20 VBench50 prompts at 1.8/2.4/3.0×, reuse baselines, no VBench scoring.

Results: `/mnt/hdd/xiongyuxiang/tmp/exp/ours21_sea7_e328_online_offline800_8rounds_v1` (linked in `experiment_results/`); weights: matching directory under models. Reuse inputs from `ours21_sea7_e328_online_offline800_8h_v1_setup`. The former 4-round frozen run remains archived and paused; it completed no formal candidates or online training rounds.

Existing four-GPU timing model estimates 15.097 hours for all 8 rounds including R4/R8 evaluation. Eight hours no longer determines the round count. This is an estimate, not a deadline.
