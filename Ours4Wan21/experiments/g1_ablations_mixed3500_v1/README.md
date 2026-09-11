# G1 method: three mixed3500 ablations

正式方法已确定为原8×8 **CNN+G1**；主入口为 `../../main.py`，方法定义及本目录角色见 [METHOD.md](../../METHOD.md)。G2/G3/G4、16×20 CNN与G1特征MLP属于对照或消融。


User selected G1 as the method and authorized three GPU training jobs using the previous settings.
- GPU0 SEA7: last seven original scalar state coordinates, original MLP hidden256 x3, LayerNorm+SiLU.
- GPU1 G1_CNN_16x20: current/previous/last-recompute raw latent; direct mean pool [16,4,16,20], full-time mean then spatial mean pool [16,16,20], full-time population variance then spatial mean pool [16,16,20]. Same 32/32/64 convolution channels, final encoder pooling, fusion128 and 2x256 head as G1. Input resolution only changes; parameter count is unchanged. No SEA filtering.
- GPU2 G1_MLP: exact existing G1 normalized 8x8 cache, flattened 18439-dimensional input into original hidden256 x3 MLP, LayerNorm+SiLU; no CNN/fusion encoder.

Each has independent actor/Q1/Q2/V encoders and target Q networks. Same source transitions and original 2800/350/350 splits, terminal RGB PSNR rewards, Exact-K masks and sigma/actions. 200 epochs, batch256, lr1e-4, AdamW weight decay.01, tau.7, beta1.5, weight_max30, gamma1, target_rho.999, grad clip1, seed42, FP32 networks, original TF32 policy. Census e170–200; select two-sided stable candidate e171–199 with the original rank_checkpoints implementation.

model.py defines portable ablation networks/checkpoints; features.py defines large-spatial causal extraction. pipeline.py verifies prior dataset/cache identity and runs three independent workers, so SEA7/MLP start without waiting for the large feature extraction. launch.sh configures the existing Wan environment and NVIDIA compatibility library, initializes then runs the durable pipeline. Completed checkpoints and caches can be resumed; frozen code/input identity changes fail closed. User authorization is already explicit; there is no additional confirmation gate.

SEA7 reuses the G1 normalizer's last7 dimensions and FP16 encoded scalar coordinates. G1_MLP symlinks the original full cache and normalizer. Large-spatial extraction reads the original FP16 archived latents and checks their SHA against original extraction manifests. It recomputes all three branches at 16x20, fits per-coordinate mean/population std on train rows only, then stores normalized FP16 and feeds FP32 to networks. Variance is across full original time before spatial pooling. First-step latent features are zero; history advances only after actual recorded action commit.

Results: /mnt/hdd/xiongyuxiang/tmp/exp/ours21_g1_ablations_mixed3500_v1; model weights: /mnt/hdd/xiongyuxiang/tmp/models/ours21_g1_ablations_mixed3500_v1. experiment_results contains only a symlink. Existing global GPU experiment lock is respected; GPU3 receives no job. Training and checkpoint selection are automatic; this suite does not launch video evaluation.
