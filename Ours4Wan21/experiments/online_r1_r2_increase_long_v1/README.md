# R1 / R2 / Increase sorted long trace figure

`plot.py` reads completed R1/R2 collection artifacts and the same five Increase references (VBench155, one per level) used previously. It exports one complete PNG/SVG long figure plus source tables and validation to the active online run's `analysis/r1_r2_increase_long/`. No changes to live training source or GPU tasks.

Chart contract: compare placement of recompute/reuse decisions at similar realized skip counts. Static categorical matrix, 205 rows × 50 steps, all rows preserved, ascending actual skip count; ties ordered R1, R2, Increase, then trajectory ID. R1 uses original e391 sampling; R2 uses R1-selected joint e12 sampling. Neither collection is an evaluation of that same round's trained checkpoint. OpenVid training prompts differ from Increase's VBench155; no paired quality claim.

Palette: three source categories with pale blue (R1), pale pink (R2), pale gold (Increase) row backgrounds as explicitly requested. Common dark filled blocks = recompute, empty background-colored cells = reuse. Explicit row labels supplement color. Repeated skip-count headers retain step coordinates and metrics while scrolling. Show skip count, last25 skip count, longest last25 run, PSNR and measured speedup.

Audit: both CFG branch actions and executed transformer blocks, source checkpoint/sampling seed, sealed trace/timing/measurement and quality reference hashes. Full prompt and step/branch tables retain provenance. PNG/SVG standalone renderer uses Matplotlib with CJK fonts; inspect exported top/middle/bottom at readable resolution before handoff.
