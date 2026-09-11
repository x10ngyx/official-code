# CPU tests

Run `python -m unittest discover -s tests -v` in wan2.2 with OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS and NUMEXPR_NUM_THREADS set to 1. Tests use synthetic CPU fixtures, not a trained actor or full Wan22 GPU inference.
