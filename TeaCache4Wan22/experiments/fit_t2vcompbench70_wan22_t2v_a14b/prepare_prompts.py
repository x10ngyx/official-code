#!/usr/bin/env python3
"""Create the reproducible 70-prompt calibration manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path


SOURCE_REPOSITORY = "https://github.com/KaiyueSun98/T2V-CompBench"
SOURCE_COMMIT = "4fa8be2c46d49796a16678c245ea16e3f12bc4c1"
CATEGORIES = (
    ("consistent_attribute_binding", "1_consistent_attr.txt"),
    ("dynamic_attribute_binding", "2_dynamic_attr.txt"),
    ("spatial_relationships", "3_spatial_relationship.txt"),
    ("motion_binding", "4_motion_binding.txt"),
    ("action_binding", "5_action_binding.txt"),
    ("object_interactions", "6_interaction.txt"),
    ("generative_numeracy", "7_numeracy.txt"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--per-category", type=int, default=10)
    return parser.parse_args()


def download_prompt_file(filename: str) -> bytes:
    url = (
        "https://raw.githubusercontent.com/KaiyueSun98/T2V-CompBench/"
        f"{SOURCE_COMMIT}/prompts/{filename}"
    )
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "TeaCache4Wan22-calibration"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if not payload:
                raise RuntimeError(f"Downloaded an empty prompt file: {url}")
            return payload
        except Exception as exc:  # pragma: no cover - network-dependent retry path
            last_error = exc
            if attempt < 5:
                time.sleep(attempt * 2)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.output_dir / "source_prompts"
    source_dir.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(CATEGORIES)) as executor:
        futures = {
            filename: executor.submit(download_prompt_file, filename)
            for _, filename in CATEGORIES
        }
        payloads = {filename: future.result() for filename, future in futures.items()}

    rng = random.Random(args.selection_seed)
    manifest: list[dict[str, object]] = []
    source_files: list[dict[str, object]] = []
    for category_index, (category, filename) in enumerate(CATEGORIES, 1):
        payload = payloads[filename]
        target = source_dir / filename
        target.write_bytes(payload)
        prompts = [line.strip() for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
        if len(prompts) != 100:
            raise ValueError(
                f"Historical T2V-CompBench file {filename} has {len(prompts)} non-empty prompts; expected 100"
            )
        if args.per_category > len(prompts):
            raise ValueError(f"Cannot sample {args.per_category} prompts from {filename}")
        selected_indices = sorted(rng.sample(range(len(prompts)), args.per_category))
        source_files.append(
            {
                "category": category,
                "filename": filename,
                "prompt_count": len(prompts),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        for source_index in selected_indices:
            manifest.append(
                {
                    "ordinal": len(manifest),
                    "sample_id": f"c{category_index}_{source_index + 1:03d}",
                    "category_index": category_index,
                    "category": category,
                    "source_file": filename,
                    "source_index_0based": source_index,
                    "source_line_1based": source_index + 1,
                    "prompt": prompts[source_index],
                }
            )

    manifest_path = args.output_dir / "prompts.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
        encoding="utf-8",
    )
    metadata = {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_suite_size": 700,
        "selection_method": "One global random.Random(seed) stream; sample without replacement within each category; sort selected source indices",
        "selection_seed": args.selection_seed,
        "categories": len(CATEGORIES),
        "per_category": args.per_category,
        "selected_prompt_count": len(manifest),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_files": source_files,
    }
    (args.output_dir / "prompt_selection.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
