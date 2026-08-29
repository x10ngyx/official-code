# Tests

- `test_core.py` compares the PSNR and SSIM kernels against direct
  transcriptions of the locked upstream definitions.
- `test_lpips.py` compares batched evaluation against the locked per-frame
  AlexNet `spatial=True` call on CPU.
- `test_integration.py` creates an MP4 pair in a temporary directory, checks
  strict decoding/evaluation, and exercises the repository-wide PSNR entry
  point with `--protocol rgb_full_reference_v1`.

Run these tests in the `wan2.2` environment with all BLAS thread limits set to
one, as shown in the parent README.
