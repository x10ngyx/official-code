#!/usr/bin/env python3
"""Validate upstream or prepared SeaCache4Wan22 source trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
LOCK = json.loads((PROJECT / "upstream_lock.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=["upstream", "prepared"])
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if git(source, "rev-parse", "HEAD") != LOCK["wan22"]["commit"]:
        raise ValueError("Wan2.2 commit does not match upstream_lock.json")
    expected = LOCK["wan22"][
        "original_file_sha256" if args.mode == "upstream" else "prepared_file_sha256"
    ]
    observed = {}
    for relative, digest in expected.items():
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[relative] = sha256(path)
        if observed[relative] != digest:
            raise ValueError(
                f"{args.mode} hash mismatch for {relative}: "
                f"expected {digest}, got {observed[relative]}"
            )
        if path.suffix == ".py":
            compile(path.read_bytes(), str(path), "exec")

    integration = LOCK["integration_artifacts"]
    for path_key, hash_key in (
        ("patch", "patch_sha256"),
        ("runtime", "runtime_sha256"),
        ("timing_runtime", "timing_runtime_sha256"),
        ("protocol", "protocol_sha256"),
    ):
        path = PROJECT / integration[path_key]
        if sha256(path) != integration[hash_key]:
            raise ValueError(f"integration artifact hash mismatch: {path}")

    forbidden = ("block_cache", "cfg_cache", "zeustimestep", "teacache")
    found = []
    if args.mode == "prepared":
        active_paths = [source / relative for relative in expected]
        active_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in active_paths
            if path.suffix == ".py"
        ).lower()
        found = [marker for marker in forbidden if marker in active_text]
        if found:
            raise ValueError(f"forbidden cache implementation markers found: {found}")

    payload = {
        "schema": "seacache4wan22_prepared_v1",
        "status": "pass",
        "mode": args.mode,
        "source": str(source),
        "wan22_commit": LOCK["wan22"]["commit"],
        "sha256": observed,
        "forbidden_cache_markers": found,
        "patch_sha256": integration["patch_sha256"],
        "runtime_sha256": integration["runtime_sha256"],
        "timing_runtime_sha256": integration["timing_runtime_sha256"],
        "protocol_sha256": integration["protocol_sha256"],
        "integration_artifacts": {
            hash_key: integration[hash_key]
            for hash_key in (
                "patch_sha256",
                "runtime_sha256",
                "timing_runtime_sha256",
                "protocol_sha256",
            )
        },
    }
    if args.write_manifest:
        if args.mode != "prepared":
            raise ValueError("--write-manifest requires --mode prepared")
        target = source / ".seacache4wan22_prepared.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
