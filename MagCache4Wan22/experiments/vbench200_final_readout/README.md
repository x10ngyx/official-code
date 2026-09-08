# VBench200 final readout

`audit_results.py` performs a CPU-only, fail-closed audit of a completed three-target VBench200 suite. It checks committed report hashes, all 600 candidate manifests/timings/traces/videos, the fixed Wan2.2 protocol, official MagCache schedules, reused baseline identity, performance sums, 9000 paired frames per target, three candidate 16-dimension VBench records, and the committed shared-baseline VBench record. It writes `FINAL_SUMMARY.json`, `FINAL_VALIDATION.json`, and `RESULTS.md` into the external suite directory.

Run with the `wan2.2` interpreter and all four BLAS thread limits set to one:

```bash
python audit_results.py /all/yiran07-disk3/huteng_data/exp/SUITE
```
