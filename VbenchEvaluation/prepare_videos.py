#!/usr/bin/env python3
"""Stage ID-named Vbench200 videos for VBench standard-mode evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = (".mp4", ".gif")


def load_prompts(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    sample_ids = [row.get("sample_id") for row in rows]
    prompts = [row.get("prompt_en") for row in rows]
    if not all(isinstance(value, str) and value for value in sample_ids + prompts):
        raise TypeError("Each JSONL row must contain non-empty sample_id and prompt_en")
    if len(set(sample_ids)) != len(rows) or len(set(prompts)) != len(rows):
        raise ValueError("sample_id and prompt_en must both be unique")
    return rows


def find_source_video(videos_dir: Path, sample_id: str, seed_index: int) -> Path | None:
    stems = [f"{sample_id}-{seed_index}"]
    if seed_index == 0:
        stems.insert(0, sample_id)
    matches = [
        videos_dir / f"{stem}{extension}"
        for stem in stems
        for extension in SUPPORTED_EXTENSIONS
        if (videos_dir / f"{stem}{extension}").is_file()
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple source videos match {sample_id} seed {seed_index}: {matches}"
        )
    return matches[0] if matches else None


def stage_link(source: Path, target: Path) -> None:
    if len(target.name.encode("utf-8")) > 255:
        raise ValueError(f"Staged filename exceeds 255 UTF-8 bytes: {target.name!r}")
    relative_source = Path(os.path.relpath(source.resolve(), target.parent.resolve()))
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        raise FileExistsError(f"Symlink points to a different source: {target}")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {target}")
    target.symlink_to(relative_source)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument(
        "--prompts-jsonl",
        type=Path,
        default=script_dir.parent / "Vbench200" / "prompts.jsonl",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        choices=range(1, 6),
        default=1,
        metavar="{1,2,3,4,5}",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Stage a partial set instead of failing on missing sample IDs.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Output manifest path (default: sibling of staging directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos_dir = args.videos_dir.resolve()
    staging_dir = args.staging_dir.resolve()
    prompts_path = args.prompts_jsonl.resolve()
    if not videos_dir.is_dir():
        raise NotADirectoryError(videos_dir)
    rows = load_prompts(prompts_path)

    planned: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    extensions: set[str] = set()
    for row in rows:
        for seed_index in range(args.expected_seeds):
            source = find_source_video(videos_dir, row["sample_id"], seed_index)
            if source is None:
                missing.append(
                    {"sample_id": row["sample_id"], "seed_index": seed_index}
                )
                continue
            extensions.add(source.suffix.lower())
            target_name = f"{row['prompt_en']}-{seed_index}{source.suffix.lower()}"
            planned.append(
                {
                    "sample_id": row["sample_id"],
                    "seed_index": seed_index,
                    "prompt_en": row["prompt_en"],
                    "source_video": str(source),
                    "staged_video": str(staging_dir / target_name),
                }
            )

    if missing and not args.allow_missing:
        preview = ", ".join(
            f"{item['sample_id']}:{item['seed_index']}" for item in missing[:10]
        )
        raise FileNotFoundError(
            f"Missing {len(missing)} expected videos; first entries: {preview}"
        )
    if len(extensions) > 1:
        raise ValueError(
            "VBench standard mode infers one filename extension for the whole folder; "
            f"found mixed extensions: {sorted(extensions)}"
        )
    if not planned:
        raise ValueError("No videos are available to stage")

    staging_dir.mkdir(parents=True, exist_ok=True)
    expected_targets = {Path(item["staged_video"]) for item in planned}
    unexpected = [path for path in staging_dir.iterdir() if path not in expected_targets]
    if unexpected:
        raise FileExistsError(
            f"Staging directory contains unexpected entries: {unexpected[:10]}"
        )
    for item in planned:
        stage_link(Path(item["source_video"]), Path(item["staged_video"]))

    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else staging_dir.parent / f"{staging_dir.name}_manifest.json"
    )
    if manifest_path.parent == staging_dir:
        raise ValueError(
            "The manifest must be outside staging-dir because VBench infers the "
            "video extension from entries in that directory."
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "dataset": "Vbench200",
        "prompts_jsonl": str(prompts_path),
        "videos_dir": str(videos_dir),
        "staging_dir": str(staging_dir),
        "expected_prompts": len(rows),
        "expected_seeds_per_prompt": args.expected_seeds,
        "staged_video_count": len(planned),
        "missing_video_count": len(missing),
        "missing": missing,
        "items": planned,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Staged {len(planned)} videos in {staging_dir}; manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
