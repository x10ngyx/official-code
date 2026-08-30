# Scripts

- `prepare_wan22.sh`: obtain, validate, and patch the locked Wan2.2 source.
- `validate_prepared_tree.py`: verify upstream hashes or a prepared tree.
- `run_t2v_a14b.sh`: run the frozen single-GPU baseline/TeaCache protocol.
- `write_run_manifest.py`: validate the Wan2.1-style pipeline/DiT timing trace
  and record one generation's protocol and provenance.
- `compare_runs.py`: validate matched manifests/traces and all 100 DiT calls,
  then report the `pipeline_generate_wall_seconds` ratio-of-sums speedup.
- `validate_calibration_source_equivalence.py`: compare a legacy/noncanonical
  calibration sample against the pinned upstream or prepared source before
  coefficient packaging.
- `package_coefficients.py`: validate and package the 70-prompt fit for public
  runtime use; equivalent HTTPS repository URLs may differ only by an optional
  trailing `.git` suffix.
