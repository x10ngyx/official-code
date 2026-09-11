# Experiments

Each runnable experiment has its own directory. fixed_protocol/ exposes the formal CLI under the required thread settings.

vbench_speed_targets_v1/ runs the complete multi-target VBench test: canonical five-prompt subset or all 200 prompts, new speed-to-K calibration, one shared same-GPU native baseline, generation, RGB PSNR/SSIM/LPIPS, VBench and final reports. Includes dry-run and sealed-stage resume; see its README for commands.

local_consistency_audit/ compares the local Wan22 IQL/Exact-K and SEA filter implementations on CPU; results are stored externally.
