#!/usr/bin/env python3
"""Build a fail-closed VBench standard-mode full-info file for staged prompts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_subset(
    full_info: list[dict[str, Any]],
    staging_manifest: dict[str, Any],
    required_dimensions: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = staging_manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("staging manifest has no items")
    selected_prompts = list(dict.fromkeys(item.get("prompt_en") for item in items))
    if any(not isinstance(prompt, str) or not prompt for prompt in selected_prompts):
        raise ValueError("staging manifest contains an invalid prompt")
    selected_set = set(selected_prompts)
    subset = [row for row in full_info if row.get("prompt_en") in selected_set]
    found = {row.get("prompt_en") for row in subset}
    if found != selected_set:
        raise ValueError(
            f"staged prompts missing from full-info: {sorted(selected_set - found)}"
        )
    covered = {
        dimension
        for row in subset
        for dimension in row.get("dimension", [])
        if isinstance(dimension, str)
    }
    missing_dimensions = sorted(set(required_dimensions) - covered)
    unknown_dimensions = sorted(covered - set(required_dimensions))
    if missing_dimensions or unknown_dimensions:
        raise ValueError(
            "partial VBench standard-mode set must cover exactly the configured "
            f"dimensions; missing={missing_dimensions}, unknown={unknown_dimensions}"
        )
    metadata = {
        "schema_version": 1,
        "selected_prompt_count": len(selected_set),
        "selected_full_info_record_count": len(subset),
        "covered_dimensions": sorted(covered),
        "required_dimension_count": len(required_dimensions),
    }
    return subset, metadata


def write_once_or_match(path: Path, payload: Any) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"existing subset full-info differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-manifest", type=Path, required=True)
    parser.add_argument(
        "--full-info",
        type=Path,
        default=script_dir.parent / "Vbench200" / "VBench200_full_info.json",
    )
    parser.add_argument(
        "--dimension-config", type=Path, default=script_dir / "dimensions.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()
    full_info = load_json(args.full_info.resolve(strict=True))
    staging_manifest = load_json(args.staging_manifest.resolve(strict=True))
    dimension_config = load_json(args.dimension_config.resolve(strict=True))
    if not isinstance(full_info, list):
        raise TypeError("full-info must be a JSON list")
    required_dimensions = dimension_config.get("dimensions")
    if not isinstance(required_dimensions, list) or len(required_dimensions) != 16:
        raise ValueError("dimension config must define the 16 VBench dimensions")
    subset, metadata = build_subset(
        full_info, staging_manifest, required_dimensions
    )
    output = args.output.expanduser().resolve()
    metadata_output = (
        args.metadata_output.expanduser().resolve()
        if args.metadata_output
        else output.with_name(output.stem + "_manifest.json")
    )
    write_once_or_match(output, subset)
    write_once_or_match(
        metadata_output,
        {
            **metadata,
            "source_full_info": str(args.full_info.resolve()),
            "staging_manifest": str(args.staging_manifest.resolve()),
            "subset_full_info": str(output),
        },
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
