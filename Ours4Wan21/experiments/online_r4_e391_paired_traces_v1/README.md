# R4 vs e391 paired evaluation traces

`plot.py` uses sealed R4 evaluation and reused e391 reference adapters. Three standalone PNG/SVG figures cover K23/K29/K35, each with the same 20 prompt pairs, ordered by prompt ID; offline immediately above online. Audit both CFG branch actions against executed blocks, checkpoint identities, same baseline SHA, and recorded quality/time.

Chart contract: paired 40×50 categorical matrices per K, 6000 cells total. Pale blue background = e391, pale pink = R4 joint e11; common dark blocks = recompute, unfilled = skip. Model/prompt labels and pair separators supplement color. Dashed line divides steps 1–25 from 26–50. ΔN on R4 row = differing actions out of 50 steps. Right-side exact lookup columns show total/late skip, longest consecutive skip in the late window, PSNR and measured per-video speedup. No omitted rows or new inference. Scientific PNG/SVG exported with Matplotlib; inspect actual exports.

Output: active online run `analysis/r4_e391_paired_traces/`, covered by the existing experiment_results symlink. Full prompts, paired metrics, traces and CFG rows accompany the figures.
