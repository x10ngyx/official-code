# Automatic CNN VBench20 evaluation

正式方法已确定为原8×8 **CNN+G1**；主入口为 `../../main.py`，方法定义及本目录角色见 [METHOD.md](../../METHOD.md)。G2/G3/G4、16×20 CNN与G1特征MLP属于对照或消融。


Waits for all four mixed3500 CNN 200-epoch runs and validation-only selections,
then acquires the existing four-GPU lock. Evaluates selected G1–G4 on the exact
existing 20 prompts and K23/K29/K35: 240 candidates, 20 reused native baselines.
Baseline GPU UUID pairing, fixed Wan21 protocol, actual block counts and file
hashes are validated. No VBench score, matching the established VBench20 scope.

`adapter.py`: strict CNN policy, selected-group causal features, existing Exact-K
controller and unchanged Wan21 integration. The trace's decision `state` is SEA7;
`latent_features.pt` stores the complementary raw feature vector once per step.
`generate_worker.py`: existing resident generator adapted to CNN and extra artifact.
`common.py`: immutable source/config helpers and same20 baseline contract.
`pipeline.py`: wait, dispatch, RGB metrics, YUV611, audits and report.
`test_adapter.py`: real-feature equality, FP16 normalization parity, exact-K CFG
trace and reset checks. `launch.sh`: environment wrapper for a persistent tmux run.

All outputs: `/mnt/hdd/xiongyuxiang/tmp/exp/ours21_cnn_vbench20_v1/`, symlinked
from project experiment_results. Per-video sealed outputs support resumption.
CNN timing includes only its selected feature group, with raw groups avoiding
unneeded SEA filters. Full generate latency includes feature/predictor overhead;
DiT, T5, VAE and predictor timing/FLOPs are reported separately. Predictor FLOPs
are Conv/Linear multiply-add estimates; feature FFT/pooling FLOPs are not counted.
