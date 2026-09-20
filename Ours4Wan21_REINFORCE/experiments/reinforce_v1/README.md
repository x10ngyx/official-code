# REINFORCE v1

See [project README](../../README.md) for the algorithm, frozen generation protocol and commands.

- `core.py`: independent random actor, online population moments, versioned checkpoint, strict episode validation and one REINFORCE optimizer step.
- `runtime_reinforce.py`: causal G1 history, stochastic training/argmax evaluation actor, shared-CFG Exact-K integration with explicit offload opt-in.
- `artifacts.py`: source/weight hashes, prompt split validation, existing MP4 baseline references, deterministic batch plans and sealed small artifacts.
- `worker.py`: persistent single-GPU Wan worker; pooled features/trace/PSNR only. Temporary training videos are deleted even if metric computation fails; no raw latent serialization.
- `run.py`: explicit-budget prepare, train/resume, global worker barrier and atomic update commit.
- `evaluation.py`: final-checkpoint held-out VideoMetrics and VBench custom score, with temporary video cleanup.
- `test_reinforce.py`: CPU tests for gradients, normalization timing, masks, causality, RNG, split isolation and restart behavior.

Default optimizer/baseline choices are documented configuration, not tuned results. Production trajectories are never synthesized by tests. CPU tests use temporary small tensors; real GPU smoke results go to the external result root.

## Epoch训练

`main.py prepare --epochs 8 --batch-size 32 ...`冻结800训练prompt的逐epoch随机排列，每轮每prompt一条新轨迹，共6400条、200次更新。K独立均匀20–40。epochs与旧batches参数互斥；支持末尾不足整批，状态及完成收据计数真实轨迹。
