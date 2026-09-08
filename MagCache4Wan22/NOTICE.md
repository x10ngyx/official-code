# Attribution

MagCache: Fast Video Generation with Magnitude-Aware Cache, Zehong Ma,
Longhui Wei, Feng Wang, Shiliang Zhang and Qi Tian (NeurIPS 2025).
Official repository: https://github.com/Zehong-Ma/MagCache
Commit: `df81cb181776c2c61477c08e1d21f87fda1cd938`.
The unchanged source and its Apache-2.0 license are in `vendor/`.

The original script credits the Alibaba Wan Team. Wan2.2 is prepared from
https://github.com/Wan-Video/Wan2.2 at
`42bf4cfaa384bc21833865abc2f9e6c0e67233dc`, under its Apache-2.0 license.

Pipeline timing and the real-shape profiling scaffold are adapted from this
repository's SeaCache4Wan22 measurement utilities. MagCache's decision logic
is executed from its own official source. Shared ComponentMetrics,
CalflopsEvaluation, VideoMetrics and VbenchEvaluation retain their respective
source attributions and locks.
