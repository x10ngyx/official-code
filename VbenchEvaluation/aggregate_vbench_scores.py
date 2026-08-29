#!/usr/bin/env python3
"""Aggregate 16 VBench dimension results using the official score formula."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=script_dir / "dimensions.json"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label", default="Vbench200")
    return parser.parse_args()


def extract_score(value: Any, dimension: str, path: Path) -> float:
    if not isinstance(value, list) or not value:
        raise TypeError(f"Expected a non-empty result list for {dimension} in {path}")
    score = float(value[0])
    if not math.isfinite(score):
        raise ValueError(f"Non-finite score for {dimension} in {path}")
    return score


def weighted_group_score(
    dimensions: list[str], weighted_normalized: dict[str, float], config: dict[str, Any]
) -> float:
    denominator = sum(config[dimension]["weight"] for dimension in dimensions)
    return sum(weighted_normalized[dimension] for dimension in dimensions) / denominator


def main() -> None:
    args = parse_args()
    score_dir = args.score_dir.resolve()
    config_path = args.config.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else score_dir / "vbench200_aggregate_scores.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = config["dimensions"]

    raw_scores: dict[str, float] = {}
    source_files: dict[str, str] = {}
    for path in sorted(score_dir.rglob("*_eval_results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"Expected a JSON object in {path}")
        for dimension, value in data.items():
            if dimension not in required:
                continue
            if dimension in raw_scores:
                raise ValueError(
                    f"Duplicate result for {dimension}: {source_files[dimension]} and {path}"
                )
            raw_scores[dimension] = extract_score(value, dimension, path)
            source_files[dimension] = str(path)

    missing = [dimension for dimension in required if dimension not in raw_scores]
    if missing:
        raise ValueError(
            "All 16 dimensions are required for Quality/Semantic/Total aggregation; "
            f"missing: {missing}"
        )

    normalization = config["normalization"]
    normalized = {
        dimension: (raw_scores[dimension] - normalization[dimension]["min"])
        / (normalization[dimension]["max"] - normalization[dimension]["min"])
        for dimension in required
    }
    weighted_normalized = {
        dimension: normalized[dimension] * normalization[dimension]["weight"]
        for dimension in required
    }
    quality_score = weighted_group_score(
        config["quality_dimensions"], weighted_normalized, normalization
    )
    semantic_score = weighted_group_score(
        config["semantic_dimensions"], weighted_normalized, normalization
    )
    quality_weight = config["aggregate_weights"]["quality"]
    semantic_weight = config["aggregate_weights"]["semantic"]
    total_score = (
        quality_score * quality_weight + semantic_score * semantic_weight
    ) / (quality_weight + semantic_weight)

    result = {
        "schema_version": 1,
        "label": args.label,
        "protocol": (
            "Official VBench normalization and aggregation applied to the fixed "
            "Vbench200 subset"
        ),
        "official_full_vbench_score": False,
        "warning": (
            "These aggregate values are Vbench200 subset scores and are not the "
            "official full-suite VBench leaderboard scores."
        ),
        "config": str(config_path),
        "source_files": source_files,
        "raw_dimension_scores": raw_scores,
        "normalized_dimension_scores": normalized,
        "weighted_normalized_dimension_scores": weighted_normalized,
        "aggregate_scores": {
            "quality_score": quality_score,
            "semantic_score": semantic_score,
            "total_score": total_score,
            "quality_score_percent": round(quality_score * 100.0, 4),
            "semantic_score_percent": round(semantic_score * 100.0, 4),
            "total_score_percent": round(total_score * 100.0, 4),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["aggregate_scores"], indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
