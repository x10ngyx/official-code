# Tests

The standard-library regression tests validate the source lock, the direct
original-Wan2.1 baseline branch, lazy TeaCache injection, official coefficient
tables, and the Vbench200 command boundary. They do not require model weights,
PyTorch, or a GPU.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```
