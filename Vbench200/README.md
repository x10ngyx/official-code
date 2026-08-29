# Vbench200

Vbench200 is a fixed test subset of 200 English prompts sampled uniformly
without replacement from the 944 unique `prompt_en` strings in VBench's
`VBench_full_info.json`.

## Files

- `prompts.txt`: one prompt per line, convenient for inference runners.
- `prompts.jsonl`: the same prompts with stable IDs, original indices,
  dimensions, and auxiliary metadata.
- `VBench200_full_info.json`: selected source records in the schema consumed
  by VBench standard-mode evaluation.
- `selection_manifest.json`: source provenance and the complete selection
  contract.
- `build_vbench200.py`: deterministic builder.
- `SHA256SUMS`: checksums for the generated dataset and selection manifest.

## Selection protocol

The source contains 946 records but only 944 unique English prompt strings.
Records are deduplicated by exact `prompt_en` equality in first-occurrence
order. Metadata from duplicate records is merged. The builder then applies a
simple random sample without replacement using Python's
`random.Random(42).sample`. The selected set is written in ascending original
prompt order so diffs remain stable.

Rebuild from the official source:

```bash
curl -L \
  https://raw.githubusercontent.com/Vchitect/VBench/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench/VBench_full_info.json \
  -o /tmp/VBench_full_info.json
python build_vbench200.py /tmp/VBench_full_info.json
sha256sum -c SHA256SUMS
```

This 200-prompt subset is not the complete VBench prompt suite and therefore
must not be presented as the official full 16-dimension VBench score. It can
be reported as the `Vbench200` subset with the evaluation dimensions and
aggregation procedure stated explicitly. Reproducible evaluation wrappers are
provided in the sibling `VbenchEvaluation/` directory.

The source prompts come from the
[VBench project](https://github.com/Vchitect/VBench), which is distributed
under the Apache License 2.0. This directory preserves source attribution and
the exact source-file checksum in `selection_manifest.json`.
