# Upstream notice

This package reimplements the metric definitions published in the official
[TeaCache](https://github.com/ali-vilab/TeaCache) evaluation directory. The
locked upstream commit, source-file hashes, and compatibility decisions are
recorded in `upstream_lock.json`. TeaCache is licensed under Apache-2.0.

The metric kernels intentionally match TeaCache, while dataset orchestration
is stricter: input shapes must match exactly and aggregation is performed over
individual videos rather than over equal-weight batches. These corrections
are described in `README.md` and are not presented as byte-for-byte execution
of TeaCache's `eval.py`.
