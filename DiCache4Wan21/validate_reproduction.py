#!/usr/bin/env python3
"""Validate DiCache4Wan21 source boundaries and locked upstream checkouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
LOCK = json.loads((PROJECT / "upstream_lock.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def validate_checkout(root: Path, lock_key: str, files_key: str) -> dict[str, object]:
    root = root.expanduser().resolve()
    expected = LOCK[lock_key]
    if git(root, "rev-parse", "HEAD") != expected["commit"]:
        raise ValueError(f"{lock_key} commit does not match upstream_lock.json")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"{lock_key} checkout contains tracked modifications")
    observed = {}
    for relative, digest in expected[files_key].items():
        observed[relative] = sha256(root / relative)
        if observed[relative] != digest:
            raise ValueError(f"{lock_key} hash mismatch: {relative}")
    return {"root": str(root), "commit": expected["commit"], "sha256": observed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan21-root", type=Path)
    parser.add_argument("--dicache-root", type=Path)
    args = parser.parse_args()

    active_files = [
        PROJECT / "generate.py",
        PROJECT / "inference_timing.py",
        PROJECT / "dicache.py",
        PROJECT / "wan21_integration.py",
    ]
    for relative, digest in LOCK["integration_artifact_sha256"].items():
        observed = sha256(PROJECT / relative)
        if observed != digest:
            raise ValueError(
                f"integration artifact hash mismatch for {relative}: "
                f"expected {digest}, got {observed}"
            )
    for path in active_files:
        compile(path.read_bytes(), str(path), "exec")
    active_text = "\n".join(
        path.read_text(encoding="utf-8") for path in active_files
    ).lower()
    forbidden = ["block_cache", "cfg_cache", "zeustimestep", "teacache", "magcache"]
    found = [marker for marker in forbidden if marker in active_text]
    if found:
        raise ValueError(f"forbidden cache implementation markers found: {found}")

    report: dict[str, object] = {
        "status": "ok",
        "active_sha256": {path.name: sha256(path) for path in active_files},
        "forbidden_cache_markers": found,
        "baseline_direct_original_call": (
            "original.generate(args)"
            in (PROJECT / "generate.py").read_text(encoding="utf-8")
        ),
    }
    if args.wan21_root:
        report["wan21"] = validate_checkout(
            args.wan21_root, "wan21", "compatibility_files"
        )
    if args.dicache_root:
        report["dicache"] = validate_checkout(
            args.dicache_root, "dicache_method_reference", "reference_files"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
