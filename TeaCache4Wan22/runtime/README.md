# Runtime

`teacache.py` is the canonical TeaCache4Wan22 runtime implementation.
`inference_timing.py` mirrors TeaCache4Wan21's outer pipeline profiler: it
records CUDA-synchronized `WanT2V.generate()` wall time plus CUDA-event spans
and actual block execution for each high/low cond/uncond DiT call. The
preparation script copies both modules plus the shared
`ComponentMetrics/component_timing.py` helper into the pinned Wan2.2 checkout,
then applies the small integration patch. This keeps prepared trees under
`TeaCache4Wan22/build/` self-contained instead of depending on their parent
directory layout.

The controller owns one gate state per high/low model stage and one cached
block residual per `(stage, CFG branch)`. It does not cache the final denoiser
output and does not skip the Wan head or unpatchify path.

The timing recorder is output-neutral. It wraps both expert models and counts
the blocks that actually run, so the TeaCache path does not need a separate
timing implementation or schema.
