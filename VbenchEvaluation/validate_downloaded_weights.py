#!/usr/bin/env python3
"""Validate a downloaded VBench weight bundle without loading any models."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=root / "weights")
    parser.add_argument("--sources", type=Path, default=root / "download_sources.json")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_dir = args.weights_dir.resolve(strict=True)
    sources_path = args.sources.resolve(strict=True)
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []

    for item in sources["files"]:
        path = weights_dir / item["target"]
        assert path.is_file(), f"missing file: {path}"
        assert path.stat().st_size == item["expected_bytes"], f"size mismatch: {path}"
        digest = sha256(path)
        if item.get("sha256"):
            assert digest == item["sha256"], f"SHA256 mismatch: {path}"
        if item.get("sha256_prefix"):
            assert digest.startswith(item["sha256_prefix"]), f"SHA256 mismatch: {path}"
        records.append(
            {
                "name": item["name"],
                "path": item["target"],
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )

    for item in sources["archives"]:
        for member, expected_bytes in item["expected_members"].items():
            relative = str(Path(item["extract_root"]) / member)
            path = weights_dir / relative
            assert path.is_file(), f"missing extracted file: {path}"
            assert path.stat().st_size == expected_bytes, f"size mismatch: {path}"
            records.append(
                {
                    "name": f"{item['name']}: {member}",
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    repositories = []
    for item in sources["git_repositories"]:
        path = weights_dir / item["target"]
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        assert commit == item["commit"], f"commit mismatch: {path}"
        repositories.append(
            {"name": item["name"], "path": item["target"], "commit": commit}
        )

    temporary_files = sorted(
        str(path.relative_to(weights_dir))
        for path in weights_dir.rglob("*")
        if path.is_file() and (path.name.endswith(".part") or path.name.endswith(".aria2"))
    )
    assert not temporary_files, f"incomplete downloads remain: {temporary_files}"

    result = {
        "schema_version": 1,
        "validation": "pass",
        "weights_dir": str(weights_dir),
        "sources": str(sources_path),
        "sources_sha256": sha256(sources_path),
        "validated_files": len(records),
        "validated_file_bytes": sum(record["bytes"] for record in records),
        "files": records,
        "repositories": repositories,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
