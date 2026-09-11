# Configuration reference

正式方法 **CNN+G1** 的配置以 [METHOD.md](../METHOD.md) 及
`../experiments/cnn_mixed3500_v1/{model.py,features.py,pipeline.py}` 为准：G1、8×8池化、mixed3500、200epoch、τ=.7/β=1.5/cap30、e171–199选点，当前e182。
本目录 `training.json` 的400epoch/MLP默认值仅用于历史实验，不是正式CNN配置；正式入口为 `../main.py`。


training.json exports the frozen local 400-epoch settings; it is not a CLI override. speed_to_k.pending.json documents an empty Wan21 empirical mapping and must fail inference until genuinely calibrated. Ten concrete sea7_<group> modes are registered in ours4wan21/latent_features.py; their dimensions and causal contracts are frozen in checkpoints. The generic sea7_latent placeholder still requires an explicit group.

[online_finetuning_review.md](online_finetuning_review.md) retains historical parameter discussion. Current online defaults use target range [1.5,3.5], 5 critic warmup + 20 joint epochs, and e11–e20 actor-agreement selection. R4/R8 evaluate 20 frozen prompts selected from the previous VBench50 at nominal 1.8/2.4/3.0× (K23/K29/K35), reuse their original same-GPU native baselines, and skip VBench scoring.

`online.json` exports `OnlineConfig` defaults; it is not a CLI override. See [remote online instructions](../experiments/online_finetuning_v1/README.md) for preparation, calibration, execution and recovery. Frozen runs reject input/code changes.
