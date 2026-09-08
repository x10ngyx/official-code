# Sources and attribution

DiCache algorithm: Jiazi Bu and collaborators, [Bujiazi/DiCache](https://github.com/Bujiazi/DiCache), pinned commit `fdbe20b669c9174bbed5ec994de073fd881c8010`. The original script and copyright header are preserved verbatim in `vendor/run_wan_dicache.py`; source SHA256 is recorded in `upstream_lock.json`. Its block algorithm is compiled without statement changes. The upstream root tree at this commit does not contain a LICENSE file; this package does not assign a new license to the vendored code.

Wan2.2 model/sampling source: [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2), pinned commit `42bf4cfaa384bc21833865abc2f9e6c0e67233dc` (Apache-2.0). Source is prepared separately, without model patches.

Batch, timing, profiling, source-audit and quality orchestration follow this repository’s MagCache4Wan22/SeaCache4Wan22 conventions. Shared ComponentMetrics, CalflopsEvaluation, VideoMetrics and VbenchEvaluation retain their own source attribution. See `docs/migration.md` for the Wan2.2 interface and lifecycle changes.
