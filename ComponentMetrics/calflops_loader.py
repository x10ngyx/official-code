"""Load the repository-locked Calflops implementation without mutating an env."""

from __future__ import annotations

import importlib.metadata
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_COMMIT = "027e89a24daf23ee7ed79ca4abee3fb59b5b23cd"
EXPECTED_VERSION = "0.3.2"


def _upsample_flops_compute_compat(*args: Any, **kwargs: Any) -> tuple[int, int]:
    """Count one interpolation FLOP per output element.

    Calflops 0.3.2 assumes that a tuple ``scale_factor`` can be raised to a
    power and also compares its length with ``len(input)`` (the batch size).
    Wan's 3D VAE uses spatial tuples such as ``(1, 2, 2)``, so the upstream
    hook raises ``TypeError`` before the decode profile can finish.
    """

    if not args:
        raise TypeError("interpolate profiling requires an input tensor")
    input_tensor = args[0]
    spatial_dims = int(input_tensor.dim()) - 2
    if spatial_dims < 1:
        raise ValueError("interpolate profiling requires at least one spatial dimension")

    size = kwargs.get("size")
    if size is None and len(args) > 1:
        size = args[1]
    if size is not None:
        output_spatial = (
            (int(size),) * spatial_dims
            if isinstance(size, (int, float))
            else tuple(int(value) for value in size)
        )
        if len(output_spatial) != spatial_dims:
            raise ValueError("interpolate size rank does not match the input spatial rank")
    else:
        scale_factor = kwargs.get("scale_factor")
        if scale_factor is None and len(args) > 2:
            scale_factor = args[2]
        if scale_factor is None:
            raise ValueError("either size or scale_factor must be defined")
        factors = (
            (float(scale_factor),) * spatial_dims
            if isinstance(scale_factor, (int, float))
            else tuple(float(value) for value in scale_factor)
        )
        if len(factors) != spatial_dims:
            raise ValueError(
                "interpolate scale_factor rank does not match the input spatial rank"
            )
        output_spatial = tuple(
            math.floor(int(input_tensor.shape[index + 2]) * factor)
            for index, factor in enumerate(factors)
        )

    if any(value < 1 for value in output_spatial):
        raise ValueError("interpolate output dimensions must be positive")
    output_elements = (
        int(input_tensor.shape[0])
        * int(input_tensor.shape[1])
        * math.prod(output_spatial)
    )
    return int(output_elements), 0


def _install_calflops_upsample_compat() -> None:
    from calflops import pytorch_ops

    pytorch_ops._upsample_flops_compute = _upsample_flops_compute_compat


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
    _install_calflops_upsample_compat()
    return calculate_flops, {
        "version": version,
        **metadata,
        "compatibility_patches": [
            "calflops_0.3.2_tuple_scale_factor_upsample_output_elements"
        ],
    }


__all__ = ["EXPECTED_COMMIT", "EXPECTED_VERSION", "load_calflops"]
