# CPU validation

Run `WAN22_PYTHON=/path/to/wan2.2/bin/python bash tests/run_tests.sh` after preparing the pinned native source. `test_official_parity.py` imports the complete unchanged upstream script as an independent oracle and checks 12 parameter/precision combinations, strict threshold equality, original NaN behavior, expert initialization, independent CFG, dtype/window semantics, lifecycle and real small WanModel native outputs (CPU attention substituted with SDPA). `test_inference_timing.py` checks observer bookkeeping; `test_reporting.py` checks probe-aware component costs, pairing and tamper rejection; `test_batch_scan.py` checks one-load workers, resume, quality completion and measured target selection.

`test_targeted_scan.py` checks relative target tolerance, nonmonotone refinement, search bounds, profile reuse, round resume and deferred selected-candidate quality.
