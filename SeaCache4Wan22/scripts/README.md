# Scripts

- `prepare_wan22.sh`: clone, verify, patch, install runtime, and validate.
- `validate_prepared_tree.py`: verify upstream/prepared hashes and method scope.
- `run_t2v_a14b.sh`: fixed-protocol baseline or SeaCache inference.
- `write_run_manifest.py`: record protocol, SeaCache parameters, inference
  timing/trace artifacts, paths, and SHA256.
- `compare_runs.py`: validate matched manifests and report inference-only
  ratio-of-sums speedup; it is also shared by the TFLOPs aggregator.
