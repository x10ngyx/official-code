# Upstream notice

The collection runtime executes
[Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) at commit
`65386b2e03c490796eede31b0325a6a595cc684e`, distributed under Apache-2.0.
The adapted Wan sampling/model-forward structure in `data_collection/src/`
retains the Alibaba Wan Team provenance.  Exact compatibility hashes are in
`upstream_lock.json`; the Apache-2.0 text is available in the sibling
`TeaCache4Wan21/LICENSE.upstream.txt` and upstream repository.

The dynamic threshold controller is adapted from the clean shared-CFG,
branch-residual implementation in sibling `SeaCache4Wan21/`.  It changes only
the scalar threshold into a frozen 50-step path and adds behavior-data trace
fields.  It does not include block cache, CFG cache, ZEUS, TeaCache, or a
learned policy.

Canonical PSNR, SSIM, and LPIPS are computed through sibling `VideoMetrics/`
using protocol `rgb_full_reference_v1`. Calflops accounting uses sibling
`CalflopsEvaluation/` and locked `calflops==0.3.2`.
