# RL implementation

正式方法为 **CNN+G1**，顶层入口 `../main.py`，精确合同见 [METHOD.md](../METHOD.md)。
已验证的CNN网络/训练实现在 `../experiments/cnn_mixed3500_v1/`，CNN策略与控制器接入在 `../experiments/cnn_vbench20_v1/adapter.py`；正式入口直接复用它们。
本目录的 `local_iql.py` 和 `runtime.py` 提供共用IQL/Exact-K底座；`policy.py`、`train.py`、`inference.py` 的原接口属于历史MLP流程，不能直接加载CNN checkpoint。下文是这些共用与历史模块的索引。


contracts.py defines state/order/hyperparameters; data.py validates completed collection traces; local_iql.py preserves the exact local loss/update kernels; train.py runs offline IQL; policy.py loads mode-checked FP32 actors; runtime.py installs the learned gate over sibling SeaCache4Wan21; inference.py handles fixed-protocol generation and component measurement.

selection.py freezes quality-blind random subsets; overhead.py records/validates predictor-only counts and timing shares; shared.py propagates declared remote roots; evaluation.py writes predictor-aware performance summaries; analysis.py produces full training curves and the local post300 stable-checkpoint selection.

Online continuation: online_common.py freezes contracts/prompts/restart seals; online_setup.py builds the isolated prompt pool and measured K mapping; online_generation.py runs persistent resident workers; online_training.py handles cumulative uniform replay, 5+20 epochs, e11–20 selection and optimizer/RNG continuation; online_pipeline.py orchestrates eight rounds and VBench20 at R4/R8.

`bloc_features.py` provides independently versioned compact cache-relative and short-history features shared by offline extraction and the live controller.
