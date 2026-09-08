# tests

run_tests.sh uses conda wan2.2 with all four BLAS thread variables set to 1. Tests compare the scoped adapter with a full import of the original official script, independent scalar schedules, calibration, cleanup, measured-call timing and reporting. CPU fixtures are not GPU performance results.

`test_batch_scan.py` verifies original-gate plan/forward equivalence, schedule
aliases, unmet speed targets, native generation arguments, persistent one-load
workers, same-GPU paired shards, verified resume without loading weights, and
quality-pending versus complete suite status. Worker orchestration uses small
fixtures; this does not run full A14B inference.

The three recommended preset centers also pass the complete original-forward
comparison. Preset tests cover decimal grids, preservation of threshold aliases,
fixed R/K target selection, relative/absolute tolerance, subset targets and CLI conflicts.
