# Upstream notice

SeaCache4Wan22 applies a minimal patch to
[Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) at commit
`42bf4cfaa384bc21833865abc2f9e6c0e67233dc`, distributed under Apache-2.0.
Modified Wan files retain the Alibaba Wan Team copyright header. Exact source,
patch, runtime, and prepared-tree hashes are recorded in `upstream_lock.json`.

The controller is a clean extraction from this workspace's reviewed Wan2.2
CFG-synchronized SeaCache implementation. Development-reference hashes are
recorded in `upstream_lock.json`. The package intentionally excludes the
historical source tree's experimental block cache, CFG cache, ZEUS, TeaCache,
and learned-policy code.

Quality evaluation is delegated to the sibling `VideoMetrics/` and
`VbenchEvaluation/` projects.
