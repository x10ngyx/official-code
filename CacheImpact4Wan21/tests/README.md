# CPU validation

`test_single_skip.py` checks all 49 intervention positions, separate CFG caches,
cache age one, ordered plans, input safety, prefix/G1 consistency, real-path
trace validation, small raw RGB errors and resume corruption rejection.
Run from the project root in conda `wan2.2` with all four BLAS thread variables=1.
These are not substitutes for the collector's mandatory native GPU equivalence check.

`test_dataset_storage.py` checks 50 raw FP16 latent files including the post-intervention suffix, 50 step/100 CFG rows, baseline pairing, frozen manifests, immutable three-table publication, and rejection of missing or corrupted artifacts. The suite contains 12 CPU tests.
