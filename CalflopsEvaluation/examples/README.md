# Examples

- `toy_adapter.py`：两个 tiny full-forward case、一个 controller case，以及手工 attention/reuse component。
- `toy_mapping.json`：baseline/recompute/reuse 在 high/low stage 对应的组件集合。
- `toy_trace.jsonl`：两个样本、每个四步的示例 cache trace。

这些文件只用于 CPU smoke test，不代表 Wan2.2 的正式 FLOPs 数值。
