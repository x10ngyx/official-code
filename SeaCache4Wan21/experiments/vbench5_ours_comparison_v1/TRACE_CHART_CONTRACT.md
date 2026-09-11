# Exact action-trace chart contract

## Static image delivery

The user requested a visible image after the widget handoff. Export a reproducible Matplotlib
PNG and SVG: three vertically stacked binary action matrices, each 10 rows × 50 steps.
Keep all five prompts and both methods; collapse CFG branches only after exact equality checks.
Use a blue filled cell for reuse and an open neutral cell for recompute (non-color distinction).
Label every zero-based step, each method/prompt, per-video PSNR and reuse count. Nominal panel
targets are distinct from actual measured speedups. Canvas: 20 × 14 inches, 160 dpi.
Takeaway: SEA7 begins reuse later and has longer consecutive reuse runs; this is descriptive,
not a causal quality attribution. Check all 3,000 input actions, export source hashes and
inspect the actual PNG for legibility. Files live in `analysis/trace_psnr_audit/figures/`.

- Question: are PSNR measurements comparable, and where do SeaCache/SEA7 action paths differ?
- Expected evidence: 2 methods × 3 target bands × 5 prompts × 2 CFG branches × 50 steps = 3,000 binary actions.
- Form: three 50-column, 20-row binary heatmaps, one per nominal target. Every prompt retains both branches.
- Surface: inline native Data Analytics heatmap widgets; source SQLite and full CSV retained. Static fallback only if widget delivery fails.
- Palette: single-root sequential blue plus neutral background; values limited to 0/1, with explicit state labels and numeric hover values. No inferred interpolation between steps.
- Footprint: full-width chart, horizontal/vertical exploration for 50 steps and 20 labeled rows, exact exported CSV available.
- Validation: compare declared reuse paths with each decision and actual timing block execution, not only reuse totals. Recompute all 30 videos' PSNR from RGB decoded frames using shared kernels; compare baselines by hash across GPUs and metadata across all 39 quality summaries.
- Caveat: K29 and nominal 2.4× SeaCache are not equal-speed or equal-compute settings. Trace differences alone do not prove a causal explanation for quality differences.

## Final delivery adjustment

The widget accepts at most500 records, so after proving cond/uncond action equality for every
method/target/prompt, delivery uses3 losslessly collapsed50-column×10-row heatmaps. Raw3,000
branch actions remain inCSV/SQLite. Native heatmap renderer uses its category rows as the
vertical axis and series as columns: bindtrace to x, reuse to y, step_label to color/series.
Three corrected horizontal widget calls returnedok with no quality warnings. Host browser
pixels were not accessible; encoding shape/row count/column count and raw-action parity were checked.
