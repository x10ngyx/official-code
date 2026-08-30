# Tests

Run the CPU-only suite with:

```bash
WAN22_PYTHON=/path/to/wan2.2/bin/python bash tests/run_tests.sh
```

The prepared-tree check in `scripts/prepare_wan22.sh` additionally applies the
patch to the exact upstream commit and verifies all resulting hashes.

The suite also validates per-DiT-call full/reuse timing and trace-weighted
Calflops aggregation for matched baseline/SeaCache manifests.
