"""Strict component timing/FLOPs extraction for formal reports."""

from __future__ import annotations

import math
from typing import Any

from component_flops import validate_component_profiles


def _nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def extract_component_latency(
    timing: dict[str, Any],
    *,
    expected_t5_calls: int = 2,
    expected_vae_decode_calls: int = 1,
) -> dict[str, float]:
    if int(timing.get("schema_version", 0)) < 2:
        raise ValueError("timing trace predates required component timing schema v2")
    components = timing.get("component_latency")
    if not isinstance(components, dict):
        raise ValueError("timing trace has no component_latency object")
    expected = {
        "t5": expected_t5_calls,
        "vae_decode": expected_vae_decode_calls,
    }
    result: dict[str, float] = {}
    for component, expected_calls in expected.items():
        row = components.get(component)
        if not isinstance(row, dict) or row.get("call_count") != expected_calls:
            raise ValueError(
                f"{component} call count must be {expected_calls}, got "
                f"{None if not isinstance(row, dict) else row.get('call_count')}"
            )
        result[f"{component}_cuda_seconds"] = _nonnegative_number(
            row.get("cuda_seconds"), f"component_latency.{component}.cuda_seconds"
        )
        result[f"{component}_host_span_seconds"] = _nonnegative_number(
            row.get("host_span_seconds"),
            f"component_latency.{component}.host_span_seconds",
        )
    dit = components.get("dit")
    if not isinstance(dit, dict):
        raise ValueError("timing trace has no component_latency.dit object")
    result["dit_cuda_seconds"] = _nonnegative_number(
        dit.get("cuda_seconds"), "component_latency.dit.cuda_seconds"
    )
    recorded = _nonnegative_number(
        timing.get("model_forward_cuda_seconds"), "model_forward_cuda_seconds"
    )
    if not math.isclose(
        result["dit_cuda_seconds"], recorded, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise ValueError("component DiT CUDA time disagrees with call-trace total")
    return result


def extract_component_tflops(profile: dict[str, Any]) -> dict[str, float]:
    components = validate_component_profiles(profile.get("component_profiles"))
    return {
        "estimated_t5_tflops_per_video": _nonnegative_number(
            components["t5"]["estimated_tflops_per_video"],
            "component_profiles.t5.estimated_tflops_per_video",
        ),
        "estimated_vae_decode_tflops_per_video": _nonnegative_number(
            components["vae_decode"]["estimated_tflops_per_video"],
            "component_profiles.vae_decode.estimated_tflops_per_video",
        ),
    }


__all__ = ["extract_component_latency", "extract_component_tflops"]
