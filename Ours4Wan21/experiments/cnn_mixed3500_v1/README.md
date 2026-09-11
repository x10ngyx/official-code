# Mixed3500 four-group CNN IQL

正式方法已确定为原8×8 **CNN+G1**；主入口为 `../../main.py`，方法定义及本目录角色见 [METHOD.md](../../METHOD.md)。G2/G3/G4、16×20 CNN与G1特征MLP属于对照或消融。


Authorized production run: G1 raw three states, G2 raw two deltas, G3 SEA three
states, G4 SEA two deltas. Every input has direct 3D mean pooling (16,4,8,8),
full-time mean then spatial pooling (16,8,8), and full-time population variance
then spatial pooling (16,8,8). History follows actual logged recomputes.

`features.py` defines causal extraction; `model.py` defines the exact benchmarked
32/32/64 dual-path encoder, fusion128 and 2x256 MLP. Actor/Q1/Q2/V are independent.
`pipeline.py` freezes/audits the original mixed3500 source, extracts four variants
in one four-GPU pass, builds normalized FP16 RAM-friendly caches, trains four
independent seed42 intermediate IQL runs for 200 epochs, and selects checkpoints
using the existing validation-only post170 rule. No video generation is launched.
`test_contract.py` verifies causality, variance order, numerical cache encoding,
shape/parameter compatibility and real IQL backward/checkpoint roundtrip.

Raw extracted rows use FP32 to avoid overflowing SEA variance. Final caches use
train-only per-coordinate mean/std (population std floor1e-6), then FP16, then
FP32 at the model. All raw/current history tensors are quantized FP16->FP32 to
match archived latents. SEA operates on every history tensor at current sigma.
No sample-specific magnitude normalization and no future/reference state input.

Resume: `launch.sh` starts/reattaches the durable pipeline. Completed row hashes,
cache manifests and epoch checkpoints are verified before reuse. Failed steps
stop the pipeline; rerunning the same launch resumes valid completed work.
Source hashes are frozen, preventing silent mixing after code changes.

Results: `/mnt/hdd/xiongyuxiang/tmp/exp/ours21_cnn_mixed3500_v1/`.
Weights: `/mnt/hdd/xiongyuxiang/tmp/models/ours21_cnn_mixed3500_v1/`.
The global `wan21_benchmark_4gpu.lock` protects GPU allocation. Status and worker
logs are inside the result directory. The experiment has its own versioned
checkpoint schema; existing MLP checkpoints and loaders remain untouched.

User revision: 200 epochs; census e170–e200 and two-sided candidates e171–e199. Cache extraction/build continues, but training waits for explicit IQL parameter confirmation via TRAINING_PARAMETERS_CONFIRMED.json matching the frozen training signature.

Confirmed IQL profile: aggressive_a1_v1 (intermediate relative to baseline/full aggressive), tau=.7, beta=1.5, weight_max=30. Other optimizer/IQL settings unchanged; 200 epochs, candidates e171–e199.
