"""Portable result-root policy for the data-collection package."""

from __future__ import annotations

import os
from pathlib import Path


RESULT_BASE_ENV = "OURS4WAN21_EXP_BASE"
DEFAULT_RESULT_BASE = Path("/all/yiran07-disk3/huteng_data/exp")


def result_base() -> Path:
    """Return the configured external experiment root.

    The project default preserves the local repository convention.  A remote
    deployment may set ``OURS4WAN21_EXP_BASE`` (the launcher maps ``EXP_BASE``
    to it) without changing source code.
    """

    configured = os.environ.get(RESULT_BASE_ENV)
    return Path(configured).expanduser().resolve() if configured else DEFAULT_RESULT_BASE.resolve()


def require_result_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    base = result_base()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(
            f"output must be below {base}; set {RESULT_BASE_ENV} for a different remote root: "
            f"{resolved}"
        ) from exc
    return resolved
