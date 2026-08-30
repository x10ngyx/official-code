"""Trace-weighted Wan2.1 DiT TFLOPs and matched inference speedup."""

from __future__ import annotations

import json
import hashlib
import math
import sys
from pathlib import Path
from typing import Any


TFLOP_DIVISOR = 1_000_000_000_000.0

REPOSITORY_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_DIR / "ComponentMetrics"))
from reporting import extract_component_latency, extract_component_tflops  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def summarize_timing(timing: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    if timing.get("status") != "success":
        raise ValueError("timing trace is not successful")
    calls = timing.get("calls")
    per_forward = profile.get("per_model_forward")
    if not isinstance(calls, list) or len(calls) != 100 or not isinstance(per_forward, dict):
        raise ValueError("timing/profile lacks calls or per_model_forward")
    block_count = int(per_forward["transformer_blocks"])
    full_flops = float(per_forward["estimated_full_flops"])
    always_flops = float(per_forward["estimated_always_on_flops"])
    if block_count <= 0 or not 0.0 <= always_flops <= full_flops:
        raise ValueError("invalid Calflops profile constants")
    total_flops = 0.0
    for call in calls:
        executed = int(call["blocks_executed"])
        if not 0 <= executed <= block_count:
            raise ValueError(f"invalid blocks_executed={executed}")
        total_flops += always_flops + (full_flops - always_flops) * executed / block_count
    pipeline_seconds = float(timing["pipeline_generate_wall_seconds"])
    cuda_seconds = timing.get("model_forward_cuda_seconds")
    if not math.isfinite(pipeline_seconds) or pipeline_seconds <= 0.0:
        raise ValueError("invalid pipeline inference time")
    throughput = (
        total_flops / float(cuda_seconds) / TFLOP_DIVISOR
        if isinstance(cuda_seconds, (int, float)) and float(cuda_seconds) > 0.0
        else None
    )
    component_latency = extract_component_latency(timing)
    component_tflops = extract_component_tflops(profile)
    return {
        "schema": "ours4wan21_per_video_performance_v2",
        "latency_scope": "pipeline_generate_wall_seconds",
        "pipeline_generate_wall_seconds": pipeline_seconds,
        "model_forward_cuda_seconds": cuda_seconds,
        **component_latency,
        "model_forward_call_count": len(calls),
        "full_compute_forward_calls": sum(int(bool(call["full_compute"])) for call in calls),
        "reuse_forward_calls": sum(int(bool(call["reuse"])) for call in calls),
        "estimated_dit_flops": total_flops,
        "estimated_dit_tflops_per_video": total_flops / TFLOP_DIVISOR,
        **component_tflops,
        "estimated_achieved_tflops_per_second": throughput,
        "flops_scope": profile.get("scope"),
        "flops_profile_declared_provenance": profile.get("profile_provenance"),
    }


def compare_matched(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    baseline_seconds = float(baseline["pipeline_generate_wall_seconds"])
    candidate_seconds = float(candidate["pipeline_generate_wall_seconds"])
    baseline_tflops = float(baseline["estimated_dit_tflops_per_video"])
    candidate_tflops = float(candidate["estimated_dit_tflops_per_video"])
    if candidate_seconds <= 0.0 or candidate_tflops <= 0.0:
        raise ValueError("candidate time/TFLOPs must be positive")
    return {
        "inference_latency_speedup": baseline_seconds / candidate_seconds,
        "dit_flops_speedup": baseline_tflops / candidate_tflops,
        "speedup_definition": "matched baseline/candidate pipeline_generate_wall_seconds",
        "flops_speedup_definition": "matched baseline/candidate estimated DiT FLOPs ratio",
    }


def write_performance(timing_path: Path, profile_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    timing = read_json(timing_path)
    profile = read_json(profile_path)
    result = summarize_timing(timing, profile)
    result["timing_json"] = str(timing_path.resolve())
    result["flops_profile_json"] = str(profile_path.resolve())
    result["flops_profile_sha256"] = sha256(profile_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = ["compare_matched", "read_json", "summarize_timing", "write_performance"]
