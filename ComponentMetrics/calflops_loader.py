"""Load the repository-locked Calflops implementation without mutating an env."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_COMMIT = "027e89a24daf23ee7ed79ca4abee3fb59b5b23cd"
EXPECTED_VERSION = "0.3.2"


def load_calflops(source: Path | None = None) -> tuple[Any, dict[str, Any]]:
    configured = source or (
        Path(os.environ["CALFLOPS_SOURCE"])
        if os.environ.get("CALFLOPS_SOURCE")
        else None
    )
    metadata: dict[str, Any] = {}
    if configured is not None:
        resolved = configured.expanduser().resolve(strict=True)
        if not (resolved / "calflops" / "__init__.py").is_file():
            raise FileNotFoundError(f"not a Calflops source checkout: {resolved}")
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=resolved,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        if commit != EXPECTED_COMMIT:
            raise ValueError(
                f"Calflops commit mismatch: expected {EXPECTED_COMMIT}, got {commit}"
            )
        sys.path.insert(0, str(resolved))
        metadata = {"source_checkout": str(resolved), "commit": commit}
    try:
        from calflops import calculate_flops
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "calflops is unavailable; install calflops==0.3.2 or pass "
            "--calflops-source /path/to/calculate-flops.pytorch@027e89a"
        ) from exc
    version = (
        EXPECTED_VERSION
        if metadata
        else importlib.metadata.version("calflops")
    )
    if version != EXPECTED_VERSION:
        raise ValueError(
            f"Calflops version mismatch: expected {EXPECTED_VERSION}, got {version}"
        )
    return calculate_flops, {"version": version, **metadata}


__all__ = ["EXPECTED_COMMIT", "EXPECTED_VERSION", "load_calflops"]
