#!/usr/bin/env python3
"""Aggregate custom-input VBench dimensions into an explicitly local score."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from evaluate_custom_vbench import CUSTOM_DIMENSIONS


def score(value: Any, dimension: str, path: Path) -> float:
    if not isinstance(value, list) or not value:
        raise ValueError(f"invalid {dimension} result in {path}")
    result = float(value[0])
    if not math.isfinite(result):
        raise ValueError(f"non-finite {dimension} result in {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw: dict[str, float] = {}
    sources: dict[str, str] = {}
    for path in sorted(args.score_dir.resolve(strict=True).glob("*_eval_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for dimension, value in payload.items():
            if dimension not in CUSTOM_DIMENSIONS:
                continue
            if dimension in raw:
                raise ValueError(f"duplicate {dimension} result")
            raw[dimension] = score(value, dimension, path)
            sources[dimension] = str(path)
    missing = sorted(set(CUSTOM_DIMENSIONS) - set(raw))
    if missing:
        raise ValueError(f"missing custom-input VBench dimensions: {missing}")
    vbench_score = sum(raw.values()) / len(CUSTOM_DIMENSIONS)
    payload = {
        "schema_version": 1,
        "protocol": "vbench_custom_input_raw_mean_v1",
        "official_dimension_implementations": True,
        "official_full_vbench_score": False,
        "warning": (
            "VBench defines no official aggregate for arbitrary custom input; "
            "vbench_score is the unweighted arithmetic mean of the ten supported "
            "custom-input raw dimension scores."
        ),
        "dimensions": list(CUSTOM_DIMENSIONS),
        "raw_dimension_scores": raw,
        "source_files": sources,
        "vbench_score": vbench_score,
        "vbench_score_percent": round(vbench_score * 100.0, 4),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"vbench_score": vbench_score}, indent=2))


if __name__ == "__main__":
    main()
