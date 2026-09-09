# Configuration reference

training.json exports the frozen local 400-epoch settings; it is not a CLI override. speed_to_k.pending.json documents an empty Wan21 empirical mapping and must fail inference until genuinely calibrated. Ten concrete sea7_<group> modes are registered in ours4wan21/latent_features.py; their dimensions and causal contracts are frozen in checkpoints. The generic sea7_latent placeholder still requires an explicit group.

[online_finetuning_review.md](online_finetuning_review.md) records the confirmed 20 joint replay epochs, post-e10 actor-agreement selection (e11–e20), and VBench20 evaluation every four rounds with prompts selected on the remote machine, audits historical e385 settings, and distinguishes the remaining parameter proposals. The implementation now uses 5 warmup epochs, tau=.7 and beta1/cap20 as the discussed defaults.

`online.json` exports `OnlineConfig` defaults; it is not a CLI override. See [remote online instructions](../experiments/online_finetuning_v1/README.md) for preparation, calibration, execution and recovery. Frozen runs reject input/code changes.
