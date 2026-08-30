# Attribution and upstream locks

This reproduction integrates the TeaCache block-residual reuse mechanism with
Wan2.2 T2V-A14B.

- Wan2.2 source: `Wan-Video/Wan2.2` commit
  `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`, Apache-2.0.
- TeaCache reference implementation: `ali-vilab/TeaCache` commit
  `7c10efc4702c6b619f47805f7abe4a7a08085aa0`, Apache-2.0.
- Calibration prompts: `KaiyueSun98/T2V-CompBench` commit
  `4fa8be2c46d49796a16678c245ea16e3f12bc4c1`.

The integration files and modified upstream files carry notices identifying
their changes. Exact source locks and original file hashes are recorded in
`upstream_lock.json`. Model weights are not redistributed by this repository.
