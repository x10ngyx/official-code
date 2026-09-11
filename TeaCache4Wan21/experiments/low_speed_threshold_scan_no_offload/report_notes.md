# Report source notes

Audience: technical. Delivery mode: self-contained HTML.

Required technical-report roles map to the artifact blocks as follows: title (`title`), technical
summary (`technical_summary`), key findings (`speedup_finding`, `target_intro`,
`tradeoff_finding`), scope/definitions (`scope`), methodology (`methodology`), limitations and
robustness (`robustness`), recommendations (`next_steps`), and open questions
(`further_questions`). No required role is omitted.

Chart map:

- `speedup_chart`: asks which measured threshold is closest to 1.8×; categorical bar chart over
  seven deliberate threshold points; x=`threshold_label`, y=full-inference speedup; tooltip retains
  DiT speedup, PSNR, and reuse. Single blue-root palette with axis labels as the non-color identity.
  A bar chart is used because seven discrete calibration points are too sparse for a continuous
  trend claim and below the preferred scatter size.

The quality evidence uses an exact audit table instead of a second chart because PSNR, SSIM, and
LPIPS have different units and directionality; combining them would require normalization that
would obscure the measured values. The table also preserves timing, compute, and quality on the
same threshold rows.

## Paper-comparison supplement

`build_paper_comparison_artifact.py` builds a separate protocol-aware technical supplement. Its
required roles map to `title`, `technical_summary`, `compute_finding`, `threshold_finding`,
`timing_finding`, `speed_finding`, `scope`, `methodology`, `limitations`, `recommendations`, and
`further_questions`.

Chart map:

- `matched_compute_chart`: categorical bar chart over four TeaCache observations at exactly 25
  equivalent NFE; x=`source`, y=`psnr_db`; tooltips retain threshold, reported speedup, timer scope,
  frames, and prompt count. It supports only the claim that the local PSNR is within the published
  compute-matched interval. A relationship/scatter chart is intentionally omitted because four
  observations are insufficient for a meaningful fit.

The matched-threshold, approximately-2×, and timing-consistency evidence remains in exact tables.
The approximately-2× table is intentionally not charted because mixed timer scopes would make a
visual ranking look more comparable than the evidence supports. The timing table preserves the
small relative gaps needed to audit compute/time alignment.
