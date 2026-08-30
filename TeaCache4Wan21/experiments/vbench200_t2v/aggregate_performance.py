#!/usr/bin/env python3
"""Aggregate pure inference latency and trace-weighted Calflops TFLOPs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


EXP_ROOT = Path("/all/yiran07-disk3/huteng_data/exp").resolve()
TFLOP_DIVISOR = 1_000_000_000_000
REPOSITORY_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_DIR / "ComponentMetrics"))
from reporting import extract_component_latency, extract_component_tflops  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summary_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    return {
        "count": len(values),
        "total": sum(values),
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def require_external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as exc:
        raise ValueError(f"path must be below {EXP_ROOT}: {resolved}") from exc
    return resolved


def timing_paths(condition_dir: Path) -> dict[str, Path]:
    timing_dir = condition_dir / "timings"
    if not timing_dir.is_dir():
        raise NotADirectoryError(timing_dir)
    paths = {path.stem: path for path in sorted(timing_dir.glob("*.json"))}
    if not paths:
        raise ValueError(f"no timing traces found under {timing_dir}")
    return paths


def summarize_condition(
    *,
    label: str,
    paths: dict[str, Path],
    expected_implementation: str,
    expected_count: int,
    full_forward_flops: float,
    always_on_flops: float,
    block_count: int,
    component_tflops: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(paths) != expected_count:
        raise ValueError(
            f"{label} requires {expected_count} timing traces, found {len(paths)}"
        )
    rows: list[dict[str, Any]] = []
    for sample_id, path in sorted(paths.items()):
        timing = read_json(path)
        if timing.get("status") != "success":
            raise ValueError(f"unsuccessful timing trace: {path}")
        if timing.get("implementation") != expected_implementation:
            raise ValueError(
                f"implementation mismatch in {path}: {timing.get('implementation')!r}"
            )
        if int(timing.get("transformer_block_count", -1)) != block_count:
            raise ValueError(f"transformer block count mismatch in {path}")
        calls = timing.get("calls")
        if not isinstance(calls, list) or len(calls) != 100:
            raise ValueError(
                f"fixed 50-step CFG protocol requires 100 DiT calls in {path}"
            )

        estimated_flops = 0.0
        block_executions = 0
        for call_index, call in enumerate(calls):
            if not isinstance(call, dict):
                raise TypeError(f"call {call_index} in {path} is not an object")
            executed = int(call.get("blocks_executed", -1))
            if executed < 0 or executed > block_count:
                raise ValueError(
                    f"invalid blocks_executed={executed} in {path}, call {call_index}"
                )
            block_executions += executed
            estimated_flops += always_on_flops + (
                full_forward_flops - always_on_flops
            ) * (executed / block_count)
        if expected_implementation == "wan21" and block_executions != 100 * block_count:
            raise ValueError(f"baseline trace contains a non-full DiT call: {path}")

        inference_seconds = finite_nonnegative(
            timing.get("pipeline_generate_wall_seconds"),
            f"{path}.pipeline_generate_wall_seconds",
        )
        dit_cuda_seconds = finite_nonnegative(
            timing.get("model_forward_cuda_seconds"),
            f"{path}.model_forward_cuda_seconds",
        )
        estimated_tflops = estimated_flops / TFLOP_DIVISOR
        component_latency = extract_component_latency(timing)
        rows.append(
            {
                "condition": label,
                "sample_id": sample_id,
                "timing_path": str(path),
                "inference_latency_seconds": inference_seconds,
                "dit_cuda_seconds": dit_cuda_seconds,
                **component_latency,
                "model_forward_calls": len(calls),
                "transformer_block_executions": block_executions,
                "full_compute_forward_calls": int(
                    timing.get("full_compute_forward_calls", 0)
                ),
                "reuse_forward_calls": int(timing.get("reuse_forward_calls", 0)),
                "estimated_dit_flops": estimated_flops,
                "estimated_dit_tflops": estimated_tflops,
                **component_tflops,
                "estimated_achieved_dit_tflops_per_second": (
                    estimated_tflops / dit_cuda_seconds
                    if dit_cuda_seconds > 0
                    else None
                ),
            }
        )

    inference_values = [row["inference_latency_seconds"] for row in rows]
    cuda_values = [row["dit_cuda_seconds"] for row in rows]
    tflops_values = [row["estimated_dit_tflops"] for row in rows]
    t5_cuda_values = [row["t5_cuda_seconds"] for row in rows]
    vae_cuda_values = [row["vae_decode_cuda_seconds"] for row in rows]
    total_tflops = sum(tflops_values)
    total_cuda_seconds = sum(cuda_values)
    summary = {
        "condition": label,
        "video_count": len(rows),
        "end_to_end_inference_latency_seconds": summary_stats(inference_values),
        "dit_forward_cuda_seconds": summary_stats(cuda_values),
        "t5_cuda_seconds": summary_stats(t5_cuda_values),
        "vae_decode_cuda_seconds": summary_stats(vae_cuda_values),
        "estimated_dit_tflops_per_video": summary_stats(tflops_values),
        "estimated_t5_tflops_per_video": component_tflops[
            "estimated_t5_tflops_per_video"
        ],
        "estimated_vae_decode_tflops_per_video": component_tflops[
            "estimated_vae_decode_tflops_per_video"
        ],
        "estimated_dit_total_tflops": total_tflops,
        "estimated_achieved_dit_tflops_per_second_ratio_of_sums": (
            total_tflops / total_cuda_seconds if total_cuda_seconds > 0 else None
        ),
        "total_full_compute_forward_calls": sum(
            row["full_compute_forward_calls"] for row in rows
        ),
        "total_reuse_forward_calls": sum(row["reuse_forward_calls"] for row in rows),
        "total_transformer_block_executions": sum(
            row["transformer_block_executions"] for row in rows
        ),
    }
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--teacache-dir", type=Path, required=True)
    parser.add_argument("--calflops-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-videos", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.baseline_dir = require_external(args.baseline_dir)
    args.teacache_dir = require_external(args.teacache_dir)
    args.calflops_profile = require_external(args.calflops_profile)
    args.output_dir = require_external(args.output_dir)
    if args.expected_videos < 1:
        raise ValueError("--expected-videos must be positive")
    summary_path = args.output_dir / "summary.json"
    per_video_path = args.output_dir / "per_video.jsonl"
    if summary_path.exists() or per_video_path.exists():
        raise FileExistsError(f"refusing to overwrite performance output: {args.output_dir}")

    profile = read_json(args.calflops_profile)
    if profile.get("input", {}).get("video_shape_fhw") != [81, 480, 832]:
        raise ValueError("Calflops profile does not match the locked 81x480x832 shape")
    forward = profile.get("per_model_forward")
    if not isinstance(forward, dict):
        raise ValueError("Calflops profile has no per_model_forward object")
    full_forward_flops = finite_nonnegative(
        forward.get("estimated_full_flops"), "estimated_full_flops"
    )
    always_on_flops = finite_nonnegative(
        forward.get("estimated_always_on_flops"), "estimated_always_on_flops"
    )
    if always_on_flops > full_forward_flops:
        raise ValueError("always-on FLOPs cannot exceed full-forward FLOPs")
    block_count = int(profile.get("input", {}).get("transformer_blocks", 0))
    if block_count < 1:
        raise ValueError("invalid transformer block count in Calflops profile")
    component_tflops = extract_component_tflops(profile)

    baseline_paths = timing_paths(args.baseline_dir)
    teacache_paths = timing_paths(args.teacache_dir)
    if set(baseline_paths) != set(teacache_paths):
        missing_candidate = sorted(set(baseline_paths) - set(teacache_paths))
        missing_baseline = sorted(set(teacache_paths) - set(baseline_paths))
        raise ValueError(
            "baseline/TeaCache timing IDs differ: "
            f"missing_candidate={missing_candidate}, missing_baseline={missing_baseline}"
        )

    baseline_summary, baseline_rows = summarize_condition(
        label="baseline",
        paths=baseline_paths,
        expected_implementation="wan21",
        expected_count=args.expected_videos,
        full_forward_flops=full_forward_flops,
        always_on_flops=always_on_flops,
        block_count=block_count,
        component_tflops=component_tflops,
    )
    teacache_summary, teacache_rows = summarize_condition(
        label="teacache",
        paths=teacache_paths,
        expected_implementation="teacache",
        expected_count=args.expected_videos,
        full_forward_flops=full_forward_flops,
        always_on_flops=always_on_flops,
        block_count=block_count,
        component_tflops=component_tflops,
    )
    baseline_latency = baseline_summary["end_to_end_inference_latency_seconds"]["mean"]
    teacache_latency = teacache_summary["end_to_end_inference_latency_seconds"]["mean"]
    baseline_tflops = baseline_summary["estimated_dit_total_tflops"]
    teacache_tflops = teacache_summary["estimated_dit_total_tflops"]
    payload = {
        "schema_version": 2,
        "protocol": {
            "model": "Wan2.1-T2V-1.3B",
            "video": {"width": 832, "height": 480, "frames": 81, "fps": 16},
            "sampling": {
                "steps": 50,
                "solver": "unipc",
                "shift": 5.0,
                "cfg": 5.0,
                "seed": 42,
            },
            "precision": "DiT bfloat16",
            "memory": "single 48GB GPU; model and T5 remain resident; no offload",
        },
        "latency_definition": {
            "headline_field": "pipeline_generate_wall_seconds",
            "includes": ["text_encoding", "denoising", "vae_decode"],
            "excludes": ["model_loading", "mp4_export", "metric_evaluation"],
            "aggregation": "per-video mean over matched Vbench200 samples",
        },
        "flops_definition": profile.get("scope"),
        "calflops_profile": str(args.calflops_profile),
        "conditions": {
            "baseline": baseline_summary,
            "teacache": teacache_summary,
        },
        "comparison": {
            "latency_speedup_baseline_over_teacache": (
                baseline_latency / teacache_latency
                if teacache_latency > 0
                else None
            ),
            "latency_reduction_fraction": (
                1.0 - teacache_latency / baseline_latency
                if baseline_latency > 0
                else None
            ),
            "dit_flops_speedup_ratio_of_sums": (
                baseline_tflops / teacache_tflops if teacache_tflops > 0 else None
            ),
            "dit_flops_reduction_fraction": (
                1.0 - teacache_tflops / baseline_tflops
                if baseline_tflops > 0
                else None
            ),
        },
        "warnings": [
            "TFLOPs denotes 10^12 floating-point operations, while TFLOP/s denotes achieved estimated DiT throughput.",
            "DiT remains the headline TFLOPs metric; T5 and VAE decode TFLOPs are recorded separately in every condition.",
            "The four generation workers improve dataset makespan but do not change the per-video latency definition.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with per_video_path.open("x", encoding="utf-8") as handle:
        for row in baseline_rows + teacache_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "README.md").write_text(
        "# TeaCache4Wan21 performance result\n\n"
        "`summary.json` reports inference-only latency and Calflops-based DiT "
        "TFLOPs/TFLOP/s. `per_video.jsonl` retains every matched sample. Model "
        "loading, MP4 saving, and metric evaluation are excluded from inference "
        "latency.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
