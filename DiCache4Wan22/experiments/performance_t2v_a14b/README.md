# Component FLOPs and paired performance

`profile_calflops.py --wan22-root SOURCE --checkpoint-dir CHECKPOINT --output /all/yiran07-disk3/huteng_data/exp/RUN/profile/calflops.json` measures high/low and cond/uncond full, one-probe-block and embeddings/head paths; adds dense FlashAttention-core correction, then profiles T5 and VAE separately. Run with conda `wan2.2` and the four thread variables set to 1.

`aggregate_performance.py --baseline-manifest BASELINE --dicache-manifest CANDIDATE --calflops-profile PROFILE --output EXTERNAL_JSON` verifies source/protocol/manifest/trace hashes and sums costs per actual CFG call. Reuse uses `estimated_probe_flops`, not embeddings/head-only cost. Headline speedup uses ratio of summed complete generate times. Gate/DCTA, residual-add and scheduler FLOPs are excluded and disclosed.
