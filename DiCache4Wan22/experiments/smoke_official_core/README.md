# Small CUDA/BF16 check

`run_smoke.py --source SOURCE --output-dir EXTERNAL_DIR` creates small random-weight WanModel experts in memory, uses real CUDA attention, compares 100 native/all-full outputs and 100 repeated cached outputs, checks expert initialization and finiteness, and measures full/probe/head Calflops on the small network. No model weights are saved; no A14B checkpoint or video benchmark is run.
