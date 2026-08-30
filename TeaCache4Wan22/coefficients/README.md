# Coefficients

This directory stores protocol-bound TeaCache polynomial coefficients.

The runtime accepts only the validated `teacache4wan22_coefficients_v1`
schema emitted by `scripts/package_coefficients.py`; raw fitter JSON is
intentionally rejected.

The default Wan2.2 T2V-A14B file is produced from the 70-prompt calibration in
`experiments/fit_t2vcompbench70_wan22_t2v_a14b/`. A coefficient file is valid
only for the exact model and runtime protocol recorded inside it. The runtime
fails closed when task, geometry, sampling steps, solver, shift, CFG,
high/low boundary, dtype, or `use_ret_steps` differs.

`wan22_t2v_a14b_50step_dpmpp_nonretention.json` is the validated default:

- SHA256: `8b9550f7bd190aafcfac7871bb12d387fbfb4f6afe61e285d08a5b641b0c2970`;
- calibration: 70/70 prompts, seven T2V-CompBench categories of 10 prompts;
- fit population: 6,580 runtime gate-eligible cond/uncond records;
- high stage: R² `0.278456`, MAE `0.066099`, RMSE `0.086818`;
- low stage: R² `0.630407`, MAE `0.016613`, RMSE `0.021881`.

Independent leave-one-prompt and leave-one-category-out checks are recorded by
the experiment's `FIT_AUDIT_REPORT.json`. The high-stage fit has weak
explanatory power but stable held-out error, so it is published as the
intended TeaCache gate heuristic with an explicit caveat. This coefficient
file does not recommend a threshold; threshold claims require matched-seed
end-to-end speed and fidelity evaluation.
