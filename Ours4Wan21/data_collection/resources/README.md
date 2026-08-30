# Bundled immutable resources

`prompts/` contains the exact OpenVidHD balanced-5000 JSONL snapshot consumed
by the deterministic plan builder. It remains here because its originating
workspace source is outside the transferred `offical-code/` tree.

Large model weights and the locked Wan2.1 source tree are intentionally not
vendored here; their exact external contracts are documented in
`REMOTE_DEPLOYMENT.md` and checked by `ours4wan21_data.preflight`.
