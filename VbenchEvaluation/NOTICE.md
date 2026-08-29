# Upstream notice

This evaluation package interoperates with the official
[VBench](https://github.com/Vchitect/VBench) implementation. VBench is
licensed under the Apache License 2.0. The exact upstream commit and checksums
used to define this package are recorded in `upstream_lock.json`.

The wrappers in this directory do not claim to replace or modify the official
VBench metric implementations. `dimensions.json` transcribes the official
normalization ranges and weights from `scripts/constant.py` at the locked
commit so that aggregation is reproducible and auditable.
