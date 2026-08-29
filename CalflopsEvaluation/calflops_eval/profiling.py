from __future__ import annotations

import importlib.metadata
import math
from datetime import datetime, timezone
from typing import Any, Iterable

from .models import ManualComponent, ProfileCase


def _number(value: Any, label: str) -> int | float:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric, got {type(value).__name__}")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return value


def _calflops_version() -> str:
    try:
        return importlib.metadata.version("calflops")
    except importlib.metadata.PackageNotFoundError:
        return "source-tree-or-unknown"


def profile_items(
    items: Iterable[ProfileCase | ManualComponent],
    *,
    print_detailed: bool = False,
) -> dict[str, Any]:
    try:
        from calflops import calculate_flops
    except ImportError as exc:
        raise RuntimeError(
            "calflops is not installed; install this project or calflops==0.3.2"
        ) from exc

    components: dict[str, dict[str, Any]] = {}
    for item in items:
        if isinstance(item, ManualComponent):
            flops = _number(item.flops, f"{item.name}.flops")
            macs = None if item.macs is None else _number(item.macs, f"{item.name}.macs")
            params = _number(item.params, f"{item.name}.params")
            components[item.name] = {
                "source": "manual_formula",
                "flops": flops,
                "tflops": flops / 1_000_000_000_000,
                "macs": macs,
                "tmacs": None if macs is None else macs / 1_000_000_000_000,
                "params": params,
                "formula": item.formula,
                "metadata": dict(item.metadata),
            }
            continue

        flops, macs, params = calculate_flops(
            model=item.model,
            args=list(item.args),
            kwargs=dict(item.kwargs),
            include_backPropagation=False,
            print_results=print_detailed,
            print_detailed=print_detailed,
            output_as_string=False,
        )
        flops = _number(flops, f"{item.name}.flops")
        macs = _number(macs, f"{item.name}.macs")
        params = _number(params, f"{item.name}.params")
        warnings = []
        if flops == 0:
            warnings.append("zero FLOPs: inspect unsupported/custom operators")
        if flops < 2 * macs:
            warnings.append("FLOPs < 2*MACs: counting convention or operator coverage needs audit")
        components[item.name] = {
            "source": "calflops",
            "flops": flops,
            "tflops": flops / 1_000_000_000_000,
            "macs": macs,
            "tmacs": macs / 1_000_000_000_000,
            "params": params,
            "metadata": dict(item.metadata),
            "warnings": warnings,
        }

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": {"name": "calflops", "version": _calflops_version()},
        "scope": "forward-only component cost per call",
        "counting_convention": {
            "mac_to_flop": 2,
            "tflop_divisor": 1_000_000_000_000,
            "tflops_is_operation_count_not_rate": True,
        },
        "components": components,
    }
