# PSNR comparison chart contract

Question: how do all 12 training groups and SeaCache compare on the same five prompts?
Surface: continue the user's image-based comparison, with static PNG/SVG and exact CSV.
Form: annotated 13-row × 3-column heatmap, one shared 20–30 dB sequential blue scale;
each cell directly labels PSNR and measured speedup. No interpolation. Bold star marks the
highest mean in a column; darker/lighter tone and explicit values provide non-color access.
Keep two controls, ten features, and SeaCache in fixed semantic order. Expected 39 conditions,
195 video rows, 81 frames/video; recompute means from original per-video metrics and verify
against merged CSV. Retain checkpoint/K, actual speed, source paths and hashes in companion CSV.
Takeaway: SeaCache highest at low target, Dynamics128 highest at middle target, SpectralPhase512
highest at high target. Middle target is not speed-matched; small single-seed sample is descriptive.
Footprint: 12×10-inch PNG/SVG; inspect actual PNG for clipping and legibility before handoff.
