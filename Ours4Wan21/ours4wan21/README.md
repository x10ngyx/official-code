# RL implementation

contracts.py defines state/order/hyperparameters; data.py validates completed collection traces; local_iql.py preserves the exact local loss/update kernels; train.py runs offline IQL; policy.py loads mode-checked FP32 actors; runtime.py installs the learned gate over sibling SeaCache4Wan21; inference.py handles fixed-protocol generation and component measurement.

selection.py freezes quality-blind random subsets; overhead.py records/validates predictor-only counts and timing shares; shared.py propagates declared remote roots; evaluation.py writes predictor-aware performance summaries; analysis.py produces full training curves and the local post300 stable-checkpoint selection.

Online continuation: online_common.py freezes contracts/prompts/restart seals; online_setup.py builds the isolated prompt pool and measured K mapping; online_generation.py runs persistent resident workers; online_training.py handles cumulative uniform replay, 5+20 epochs, e11–20 selection and optimizer/RNG continuation; online_pipeline.py orchestrates eight rounds and VBench20 at R4/R8.
