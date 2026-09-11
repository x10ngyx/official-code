# Contracts

protocol.json locks Wan22 model inference. features.json locks CNN+G1 raw latent/state semantics.

speed_to_k_full_generate_20260912.json is the default VBench generation target-to-K table for 1.5–3.5x. It contains the corrected 600-row SeaCache calibration plus 17 GPU1/2/3 supplemental measurements, source artifact hashes and measured support points. Integer K18–39 entries preserve measured medians and interpolate missing K linearly in normalized latency (1/speedup). It is a SeaCache reference adopted for Ours target selection, not a measurement of CNN controller overhead. Explicit --calibration overrides it; --skip-budget bypasses it.
