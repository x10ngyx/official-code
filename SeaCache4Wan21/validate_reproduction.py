#!/usr/bin/env python3
"""Validate SeaCache4Wan21 source boundaries and optional Wan2.1 checkout."""

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


def validate_wan21(root: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    expected = LOCK["wan21"]
    if git(root, "rev-parse", "HEAD") != expected["commit"]:
        raise ValueError("Wan2.1 commit does not match upstream_lock.json")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Wan2.1 checkout contains tracked modifications")
    observed = {}
    for relative, digest in expected["compatibility_files"].items():
        observed[relative] = sha256(root / relative)
        if observed[relative] != digest:
            raise ValueError(f"Wan2.1 hash mismatch: {relative}")
    return {"root": str(root), "commit": expected["commit"], "sha256": observed}


def validate_seacache(root: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    expected = LOCK["seacache_method_reference"]
    if git(root, "rev-parse", "HEAD") != expected["commit"]:
        raise ValueError("SeaCache commit does not match upstream_lock.json")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("SeaCache checkout contains tracked modifications")
    observed = {}
    for relative, digest in expected["reference_files"].items():
        observed[relative] = sha256(root / relative)
        if observed[relative] != digest:
            raise ValueError(f"SeaCache hash mismatch: {relative}")
    return {"root": str(root), "commit": expected["commit"], "sha256": observed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan21-root", type=Path)
    parser.add_argument("--seacache-root", type=Path)
    args = parser.parse_args()

    active_files = [
        PROJECT / "generate.py",
        PROJECT / "inference_timing.py",
        PROJECT / "seacache.py",
        PROJECT / "wan21_integration.py",
    ]
    locked_artifacts = LOCK["integration_artifact_sha256"]
    for relative, digest in locked_artifacts.items():
        observed = sha256(PROJECT / relative)
        if observed != digest:
            raise ValueError(
                f"integration artifact hash mismatch for {relative}: "
                f"expected {digest}, got {observed}"
            )
    for path in active_files:
        compile(path.read_bytes(), str(path), "exec")
    active_text = "\n".join(path.read_text(encoding="utf-8") for path in active_files).lower()
    forbidden = ["block_cache", "cfg_cache", "zeustimestep", "teacache"]
    found = [marker for marker in forbidden if marker in active_text]
    if found:
        raise ValueError(f"forbidden cache implementation markers found: {found}")

    report: dict[str, object] = {
        "status": "ok",
        "active_sha256": {path.name: sha256(path) for path in active_files},
        "forbidden_cache_markers": found,
        "baseline_direct_original_call": (
            "original.generate(args)" in (PROJECT / "generate.py").read_text(encoding="utf-8")
        ),
    }
    if args.wan21_root:
        report["wan21"] = validate_wan21(args.wan21_root)
    if args.seacache_root:
        report["seacache"] = validate_seacache(args.seacache_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
