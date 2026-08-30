"""Immutable Wan2.1 source compatibility lock shared by runtime and preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DATA_PROJECT = Path(__file__).resolve().parents[2]
UPSTREAM_LOCK = DATA_PROJECT.parent / "upstream_lock.json"


def _wan21_lock() -> tuple[str, dict[str, str]]:
    payload = json.loads(UPSTREAM_LOCK.read_text(encoding="utf-8"))
    wan21 = payload.get("wan21")
    if not isinstance(wan21, dict):
        raise ValueError(f"Wan2.1 lock is missing from {UPSTREAM_LOCK}")
    commit = wan21.get("commit")
    hashes = wan21.get("compatibility_files")
    if not isinstance(commit, str) or not isinstance(hashes, dict):
        raise ValueError(f"Wan2.1 lock is malformed: {UPSTREAM_LOCK}")
    return commit, {str(key): str(value) for key, value in hashes.items()}


WAN21_COMMIT, WAN21_HASHES = _wan21_lock()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_wan21_source(root: Path) -> dict[str, str]:
    root = root.expanduser().resolve(strict=True)
    observed: dict[str, str] = {}
    for relative, expected in WAN21_HASHES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = file_sha256(path)
        if digest != expected:
            raise ValueError(
                f"locked Wan2.1 compatibility mismatch for {relative}: "
                f"expected {expected}, got {digest}"
            )
        observed[relative] = digest
    return observed
