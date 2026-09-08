# Fixed Wan21 inference launcher

run.sh invokes generate.py using conda wan2.2 and explicit BLAS limits. Select one GPU with CUDA_VISIBLE_DEVICES. Inputs are a frozen Wan21 source tree, trained policy, explicit K or empirical mapping, prompts and a Wan21 component FLOPs profile. One native warmup is followed by persistent-pipeline inference. See ../../README.md for baseline and quality evaluation.
