# Upstream notice

SeaCache4Wan21 runs against
[Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) at commit
`65386b2e03c490796eede31b0325a6a595cc684e`, distributed under Apache-2.0.
The duplicated Wan sampling/model structure in `wan21_integration.py` retains
the Alibaba Wan Team copyright notice. Exact compatibility hashes are recorded
in `upstream_lock.json`; the Apache-2.0 text is available in the sibling
`TeaCache4Wan21/LICENSE.upstream.txt` file in this repository and in the Wan2.1
upstream repository.

The SeaCache branch-local gate, SEA filter, accumulated relative-L1, and
residual-cache behavior are derived from
[jiwoogit/SeaCache](https://github.com/jiwoogit/SeaCache) commit
`8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2`. The controller is a clean
reimplementation rather than a byte-for-byte copy. It intentionally corrects
the official Wan2.1 forced-boundary behavior by storing an SEA-filtered feature
instead of a raw feature, so the following relative-L1 compares two filtered
representations. Reference hashes and this divergence are recorded in
`upstream_lock.json`; experimental block cache, CFG cache, ZEUS, TeaCache, and
learned policy code were intentionally not copied.

Quality evaluation is delegated to the sibling `VideoMetrics/` and
`VbenchEvaluation/` projects rather than to any method-specific evaluator.
