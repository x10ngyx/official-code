# Upstream notice

`upstream/teacache_generate.py` is an unmodified copy of
`TeaCache4Wan2.1/teacache_generate.py` from
[ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) at commit
`7c10efc4702c6b619f47805f7abe4a7a08085aa0`. The source retains the Alibaba
Wan Team copyright header and is distributed upstream under Apache-2.0.

The compatible Wan2.1 source is
[Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) at commit
`65386b2e03c490796eede31b0325a6a595cc684e`, also distributed under
Apache-2.0. Exact source hashes are recorded in `upstream_lock.json`.

`generate.py` and `teacache.py` are local integration code. The former executes
the locked original Wan2.1 entry point unchanged for baseline inference. Only
the explicit TeaCache path binds the three official functions from the locked
reference file and configures the official coefficients/state.

The batch-generation and evaluation orchestration does not copy or invoke
TeaCache's bundled PSNR, SSIM, LPIPS, or VBench evaluation scripts. Those
measurements are delegated to the sibling `VideoMetrics/` and
`VbenchEvaluation/` projects.
