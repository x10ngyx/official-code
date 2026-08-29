#!/usr/bin/env python3
"""Validate the local Vbench200 evaluation-resource contract without a GPU."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parent
    dataset_root = root.parent / "Vbench200"
    config = load_json(root / "dimensions.json")
    upstream = load_json(root / "upstream_lock.json")
    resources = load_json(root / "model_resources.json")
    download_sources = load_json(root / "download_sources.json")
    selection = load_json(dataset_root / "selection_manifest.json")
    full_info = load_json(dataset_root / "VBench200_full_info.json")
    prompt_rows = [
        json.loads(line)
        for line in (dataset_root / "prompts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    dimensions = config["dimensions"]
    assert len(dimensions) == len(set(dimensions)) == 16
    quality = config["quality_dimensions"]
    semantic = config["semantic_dimensions"]
    assert not set(quality) & set(semantic)
    assert set(quality) | set(semantic) == set(dimensions)
    assert set(config["normalization"]) == set(dimensions)
    assert config["normalization"]["dynamic_degree"]["weight"] == 0.5
    assert all(
        values["weight"] == 1.0
        for dimension, values in config["normalization"].items()
        if dimension != "dynamic_degree"
    )

    assert len(prompt_rows) == selection["selection"]["sample_size"] == 200
    prompt_set = {row["prompt_en"] for row in prompt_rows}
    assert len(prompt_set) == 200
    assert {record["prompt_en"] for record in full_info} == prompt_set
    assert {dimension for record in full_info for dimension in record["dimension"]} == set(
        dimensions
    )
    assert (
        selection["source"]["sha256"]
        == upstream["files"]["vbench/VBench_full_info.json"]
    )

    resource_dimensions = {
        dimension
        for resource in resources["resources"]
        for dimension in resource["dimensions"]
    }
    resource_dimensions.add(resources["weight_free_dimension"])
    assert resource_dimensions == set(dimensions)

    required_weight_targets = {
        resource["target"] for resource in resources["resources"]
    }
    downloadable_targets = {
        resource["target"] for resource in download_sources["files"]
    }
    downloadable_targets.update(
        str(Path(archive["extract_root"]) / member)
        for archive in download_sources["archives"]
        for member in archive["expected_members"]
    )
    assert required_weight_targets <= downloadable_targets
    assert all(
        item["urls"] and item["urls"][0]["kind"] in {"mirror", "origin"}
        for group in ("files", "archives", "git_repositories")
        for item in download_sources[group]
    )

    required_files = [
        "aggregate_vbench_scores.py",
        "dimensions.json",
        "download_sources.json",
        "download_vbench_weights.py",
        "evaluate_vbench.py",
        "model_resources.json",
        "prepare_videos.py",
        "requirements.txt",
        "run_vbench200.sh",
        "upstream_lock.json",
        "validate_downloaded_weights.py",
    ]
    assert all((root / name).is_file() for name in required_files)

    result = {
        "validation": "pass",
        "dataset": "Vbench200",
        "unique_prompt_count": len(prompt_set),
        "full_info_record_count": len(full_info),
        "dimension_count": len(dimensions),
        "metric_resource_entries": len(resources["resources"]),
        "download_items": sum(
            len(download_sources[group])
            for group in ("files", "archives", "git_repositories")
        ),
        "upstream_commit": upstream["commit"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
