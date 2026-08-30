# Tests

Run the CPU-only controller, per-forward timing, trace-weighted performance,
threshold-scan planning/target selection, manifest/speedup integration,
syntax, and shell checks with:

```bash
WAN22_PYTHON=/path/to/python bash tests/run_tests.sh
```

`scripts/prepare_wan22.sh` performs an additional end-to-end patch application
test against the exact upstream commit and writes a validation manifest into
the prepared checkout.
