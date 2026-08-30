# ComponentMetrics

Shared instrumentation used by the method packages in this repository.

- `component_timing.py` records CUDA-event and host spans for T5 calls and VAE
  decode calls without synchronizing inside the measured pipeline.
- `component_flops.py` profiles fixed-shape UMT5 encoder forwards and Wan VAE
  decode with the locked Calflops implementation.
- `fixed_protocol.py` rejects any Wan2.1-1.3B invocation that differs from the
  workspace-fixed model, sampling, memory, distributed, or prompt-extension
  settings.
- `reporting.py` strictly extracts schema-v2 component timing and FLOPs so old
  partial artifacts cannot be accepted as compliant results.
- DiT timing remains owned by each method package because its call trace also
  records cache-specific block execution.

Formal reports keep complete `generate()` latency and DiT TFLOPs as headline
metrics while also retaining T5 and VAE component measurements required by the
workspace protocol.
