# Threshold/retention scan

`run_scan.sh`/`.py` uses the same persistent paired suite. `prompts.jsonl` contains the existing 11-prompt calibration subset covering 16 VBench dimensions. Defaults: thresholds .04/.08/.12/.2/.3/.5 and retention .2, giving 77 measured videos plus one warmup per worker. `--thresholds ... --retention-ratios ...` controls the grid; probe depth remains the official 1.

Select targets with `--target-speedups 1.8 2.4 3.0 --target-tolerance .1`. Selection uses measured complete generate speedup; unmet targets remain `unmatched`. DiCache schedules depend on live features, so the suite never predicts schedules or merges different parameter settings as supposedly equivalent. Only exactly duplicated parameter tuples are removed.

For automatic 1.8×/2.4×/3.0× coarse-to-fine calibration with fixed official retention=.2, use [the targeted scan](../targeted_threshold_scan/README.md).
