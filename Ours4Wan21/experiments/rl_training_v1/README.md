# Offline IQL launcher

run.sh invokes train.py using conda wan2.2 and explicitly sets all four BLAS thread limits. It accepts the documented dataset/state-mode/output-dir/checkpoint-dir arguments. Fixed 400 epochs, complete validation/checkpoint each epoch; weights only under models, results only under the external experiment root with project symlinks. See ../../README.md for full commands.
