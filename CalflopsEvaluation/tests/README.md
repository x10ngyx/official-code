# Tests

CPU-only tests cover analytic attention counts, trace aggregation, strict mapping validation and the Calflops profile path.

运行时显式限制 BLAS 线程：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m unittest discover -s tests -v
```
