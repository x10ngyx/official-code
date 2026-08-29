#!/usr/bin/env python3
"""Build the reproducible Vbench200 subset from VBench_full_info.json."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any


DEFAULT_SEED = 42
DEFAULT_SAMPLE_SIZE = 200
EXPECTED_SOURCE_RECORDS = 946
EXPECTED_UNIQUE_PROMPTS = 944
SOURCE_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/Vchitect/VBench/{SOURCE_COMMIT}/"
    "vbench/VBench_full_info.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_auxiliary_info(
    target: dict[str, Any], source: dict[str, Any], prompt: str
) -> None:
    for key, value in source.items():
        if key in target and target[key] != value:
            raise ValueError(
                f"Conflicting auxiliary_info for prompt {prompt!r}, key {key!r}"
            )
        target[key] = copy.deepcopy(value)


def load_unique_prompts(
    source_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("VBench_full_info.json must contain a JSON list")
    if len(raw) != EXPECTED_SOURCE_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_RECORDS} source records, found {len(raw)}"
        )

    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record_index, record in enumerate(raw, start=1):
        if not isinstance(record, dict) or not isinstance(record.get("prompt_en"), str):
            raise TypeError(f"Invalid source record at index {record_index}")
        prompt = record["prompt_en"]
        if not prompt.strip() or "\n" in prompt or "\r" in prompt:
            raise ValueError(f"Prompt at source record {record_index} is not one line")

        if prompt not in grouped:
            grouped[prompt] = {
                "prompt_en": prompt,
                "source_unique_index_1based": len(grouped) + 1,
                "source_record_indices_1based": [],
                "dimension": [],
                "auxiliary_info": {},
            }
        item = grouped[prompt]
        item["source_record_indices_1based"].append(record_index)

        dimensions = record.get("dimension", [])
        if not isinstance(dimensions, list) or not all(
            isinstance(dimension, str) for dimension in dimensions
        ):
            raise TypeError(f"Invalid dimension list at source record {record_index}")
        for dimension in dimensions:
            if dimension not in item["dimension"]:
                item["dimension"].append(dimension)

        auxiliary_info = record.get("auxiliary_info", {})
        if not isinstance(auxiliary_info, dict):
            raise TypeError(f"Invalid auxiliary_info at source record {record_index}")
        merge_auxiliary_info(item["auxiliary_info"], auxiliary_info, prompt)

    unique = list(grouped.values())
    if len(unique) != EXPECTED_UNIQUE_PROMPTS:
        raise ValueError(
            f"Expected {EXPECTED_UNIQUE_PROMPTS} unique prompts, found {len(unique)}"
        )
    return unique, raw


def select_prompts(
    unique_prompts: list[dict[str, Any]], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    if not 0 < sample_size <= len(unique_prompts):
        raise ValueError("sample_size must be between 1 and the population size")
    rng = random.Random(seed)
    selected_indices = set(rng.sample(range(len(unique_prompts)), sample_size))
    selected = [
        copy.deepcopy(item)
        for index, item in enumerate(unique_prompts)
        if index in selected_indices
    ]
    for output_index, item in enumerate(selected, start=1):
        item["sample_id"] = f"vbench200_{output_index:03d}"
        if not item["auxiliary_info"]:
            item.pop("auxiliary_info")
    return selected


def write_outputs(
    output_dir: Path,
    source_path: Path,
    selected: list[dict[str, Any]],
    source_full_info: list[dict[str, Any]],
    unique_prompt_count: int,
    sample_size: int,
    seed: int,
    source_url: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_txt = output_dir / "prompts.txt"
    prompts_txt.write_text(
        "".join(f"{item['prompt_en']}\n" for item in selected), encoding="utf-8"
    )

    prompts_jsonl = output_dir / "prompts.jsonl"
    prompts_jsonl.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            for item in selected
        ),
        encoding="utf-8",
    )

    selected_prompt_set = {item["prompt_en"] for item in selected}
    selected_full_info = [
        copy.deepcopy(record)
        for record in source_full_info
        if record["prompt_en"] in selected_prompt_set
    ]
    (output_dir / "VBench200_full_info.json").write_text(
        json.dumps(selected_full_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "name": "Vbench200",
        "description": "A reproducible simple-random subset of 200 unique VBench prompts.",
        "source": {
            "name": "VBench_full_info.json",
            "url": source_url,
            "sha256": sha256(source_path),
            "record_count": len(source_full_info),
            "unique_prompt_count": unique_prompt_count,
            "unique_key": "prompt_en",
        },
        "selection": {
            "population_rule": (
                "Deduplicate prompt_en in first-occurrence order; merge metadata from "
                "duplicate source records."
            ),
            "method": "Simple random sample without replacement",
            "implementation": "Python 3 random.Random(seed).sample",
            "seed": seed,
            "sample_size": sample_size,
            "selected_source_record_count": len(selected_full_info),
            "output_order": "Ascending source_unique_index_1based",
        },
        "selected_source_unique_indices_1based": [
            item["source_unique_index_1based"] for item in selected
        ],
        "files": {
            "VBench200_full_info.json": (
                "Selected records in the official VBench full-info schema."
            ),
            "prompts.txt": "One English prompt per line.",
            "prompts.jsonl": "Prompts with source indices and merged VBench metadata.",
        },
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    checksum_names = [
        "VBench200_full_info.json",
        "prompts.jsonl",
        "prompts.txt",
        "selection_manifest.json",
    ]
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(output_dir / name)}  {name}\n" for name in checksum_names),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to VBench_full_info.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory (default: directory containing this script)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--source-url", default=SOURCE_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    unique_prompts, source_full_info = load_unique_prompts(source_path)
    selected = select_prompts(unique_prompts, args.sample_size, args.seed)
    write_outputs(
        output_dir=args.output_dir.resolve(),
        source_path=source_path,
        selected=selected,
        source_full_info=source_full_info,
        unique_prompt_count=len(unique_prompts),
        sample_size=args.sample_size,
        seed=args.seed,
        source_url=args.source_url,
    )


if __name__ == "__main__":
    main()
