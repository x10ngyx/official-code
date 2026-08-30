#!/usr/bin/env python3
"""Aggregate inference time and trace-weighted estimated DiT TFLOPs."""

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
BRANCHES = ("cond", "uncond")
REPOSITORY_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_DIR / "ComponentMetrics"))
from reporting import extract_component_latency, extract_component_tflops  # noqa: E402


def external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as exc:
        raise ValueError(f"path must be below {EXP_ROOT}: {resolved}") from exc
    return resolved


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected object: {path}")
    return payload


def number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"invalid {label}: {result}")
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize no values")
    return {
        "count": len(values), "total": sum(values), "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values), "min": min(values),
        "p50": percentile(values, 0.5), "p90": percentile(values, 0.9), "max": max(values),
    }


def timing_map(condition_dir: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted((condition_dir / "timings").glob("*.json"))}


def validate_calls(timing: dict[str, Any], timing_path: Path, actions: list[str] | None) -> tuple[int, int, float | None]:
    block_count = timing.get("transformer_block_count")
    calls = timing.get("calls")
    if not isinstance(block_count, int) or block_count < 1 or not isinstance(calls, list) or len(calls) != 100:
        raise ValueError(f"fixed protocol requires a 100-call DiT trace: {timing_path}")
    full_count = reuse_count = 0
    cuda_values: list[float] = []
    for index, call in enumerate(calls):
        if call.get("call_index") != index:
            raise ValueError(f"DiT call order mismatch at {index}: {timing_path}")
        expected_branch = "condition" if index % 2 == 0 else "uncondition"
        if call.get("cfg_branch") != expected_branch:
            raise ValueError(f"CFG branch mismatch at {index}: {timing_path}")
        executed = call.get("blocks_executed")
        if executed not in {0, block_count}:
            raise ValueError(f"partial Transformer block execution at {index}: {timing_path}")
        expected_full = actions is None or actions[index] == "recompute"
        if (executed == block_count) != expected_full:
            raise ValueError(f"DiT call disagrees with SeaCache decision at {index}: {timing_path}")
        full_count += int(expected_full)
        reuse_count += int(not expected_full)
        cuda_seconds = call.get("cuda_seconds")
        if cuda_seconds is not None:
            cuda_values.append(number(cuda_seconds, f"{timing_path}.calls[{index}].cuda_seconds"))
    if timing.get("full_compute_forward_calls") != full_count or timing.get("reuse_forward_calls") != reuse_count:
        raise ValueError(f"DiT call summary mismatch: {timing_path}")
    recorded_cuda = timing.get("model_forward_cuda_seconds")
    if recorded_cuda is None:
        if cuda_values:
            raise ValueError(f"missing model_forward_cuda_seconds: {timing_path}")
        return full_count, reuse_count, None
    recorded = number(recorded_cuda, f"{timing_path}.model_forward_cuda_seconds")
    if len(cuda_values) != 100 or not math.isclose(sum(cuda_values), recorded, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"DiT CUDA timing sum mismatch: {timing_path}")
    return full_count, reuse_count, recorded


def summarize_baseline(
    sample_id: str,
    timing_path: Path,
    full: float,
    component_tflops: dict[str, float],
) -> dict[str, Any]:
    timing = load(timing_path)
    if timing.get("status") != "success" or timing.get("implementation") != "wan21":
        raise ValueError(f"not a Wan2.1 baseline timing: {timing_path}")
    seconds = number(timing.get("pipeline_generate_wall_seconds"), str(timing_path), positive=True)
    full_count, reuse_count, cuda_seconds = validate_calls(timing, timing_path, None)
    calls = full_count + reuse_count
    estimated = calls * full
    component_latency = extract_component_latency(timing)
    return {
        "condition": "baseline", "sample_id": sample_id,
        "timing_path": str(timing_path), "trace_path": None,
        "pipeline_generate_wall_seconds": seconds,
        "model_forward_calls": calls, "full_compute_forward_calls": full_count,
        "reuse_forward_calls": reuse_count, "model_forward_cuda_seconds": cuda_seconds,
        **component_latency,
        "estimated_dit_flops": estimated, "estimated_dit_tflops": estimated / TFLOP_DIVISOR,
        **component_tflops,
        "estimated_achieved_dit_tflops_per_second": estimated / TFLOP_DIVISOR / cuda_seconds if cuda_seconds else None,
    }


def summarize_candidate(
    sample_id: str,
    timing_path: Path,
    trace_path: Path,
    full: float,
    always: float,
    component_tflops: dict[str, float],
) -> tuple[dict[str, Any], float]:
    timing, trace = load(timing_path), load(trace_path)
    if timing.get("status") != "success" or timing.get("implementation") != "seacache":
        raise ValueError(f"not a SeaCache timing: {timing_path}")
    if trace.get("schema") != "seacache4wan21_trace_v3" or trace.get("total_steps") != 50:
        raise ValueError(f"invalid SeaCache trace protocol: {trace_path}")
    decisions = trace.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 100:
        raise ValueError(f"expected 100 branch decisions: {trace_path}")
    recompute = reuse = 0
    estimated = 0.0
    for index, decision in enumerate(decisions):
        expected = (index, index // 2, BRANCHES[index % 2])
        observed = (decision.get("call_index"), decision.get("step_index"), decision.get("branch"))
        if observed != expected:
            raise ValueError(f"decision order mismatch in {trace_path}: {observed} != {expected}")
        action = decision.get("action")
        if action not in {"recompute", "reuse"} or decision.get("execution") != action:
            raise ValueError(f"invalid decision action in {trace_path}: {decision}")
        if action == "recompute":
            recompute += 1
            estimated += full
        else:
            reuse += 1
            estimated += always
    if trace.get("recompute") != recompute or trace.get("reuse") != reuse:
        raise ValueError(f"trace summary mismatch: {trace_path}")
    full_count, reuse_count, cuda_seconds = validate_calls(
        timing, timing_path, [str(decision["action"]) for decision in decisions]
    )
    if (full_count, reuse_count) != (recompute, reuse):
        raise ValueError(f"timing/SeaCache count mismatch: {timing_path}")
    threshold = number(trace.get("threshold"), f"{trace_path}.threshold", positive=True)
    seconds = number(timing.get("pipeline_generate_wall_seconds"), str(timing_path), positive=True)
    component_latency = extract_component_latency(timing)
    return ({
        "condition": "seacache", "sample_id": sample_id,
        "timing_path": str(timing_path), "trace_path": str(trace_path),
        "pipeline_generate_wall_seconds": seconds,
        "model_forward_calls": 100, "full_compute_forward_calls": recompute,
        "reuse_forward_calls": reuse, "model_forward_cuda_seconds": cuda_seconds,
        **component_latency,
        "estimated_dit_flops": estimated,
        "estimated_dit_tflops": estimated / TFLOP_DIVISOR,
        **component_tflops,
        "estimated_achieved_dit_tflops_per_second": estimated / TFLOP_DIVISOR / cuda_seconds if cuda_seconds else None,
    }, threshold)


def summarize_condition(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["pipeline_generate_wall_seconds"]) for row in rows]
    tflops = [float(row["estimated_dit_tflops"]) for row in rows]
    cuda = [float(row["model_forward_cuda_seconds"]) for row in rows if row["model_forward_cuda_seconds"] is not None]
    t5_cuda = [float(row["t5_cuda_seconds"]) for row in rows]
    vae_cuda = [float(row["vae_decode_cuda_seconds"]) for row in rows]
    return {
        "condition": label, "video_count": len(rows),
        "pipeline_generate_wall_seconds": stats(latencies),
        "estimated_dit_tflops_per_video": stats(tflops),
        "estimated_dit_total_tflops": sum(tflops),
        "model_forward_cuda_seconds": stats(cuda) if len(cuda) == len(rows) else None,
        "t5_cuda_seconds": stats(t5_cuda),
        "vae_decode_cuda_seconds": stats(vae_cuda),
        "estimated_t5_tflops_per_video": rows[0]["estimated_t5_tflops_per_video"],
        "estimated_vae_decode_tflops_per_video": rows[0]["estimated_vae_decode_tflops_per_video"],
        "estimated_achieved_dit_tflops_per_second_ratio_of_sums": sum(tflops) / sum(cuda) if cuda and len(cuda) == len(rows) and sum(cuda) > 0 else None,
        "total_full_compute_forward_calls": sum(int(row["full_compute_forward_calls"]) for row in rows),
        "total_reuse_forward_calls": sum(int(row["reuse_forward_calls"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--seacache-dir", type=Path, required=True)
    parser.add_argument("--calflops-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-videos", type=int, default=200)
    args = parser.parse_args()
    baseline_dir, candidate_dir = external(args.baseline_dir), external(args.seacache_dir)
    profile_path, output_dir = external(args.calflops_profile), external(args.output_dir)
    if args.expected_videos < 1:
        raise ValueError("--expected-videos must be positive")
    outputs = (output_dir / "summary.json", output_dir / "per_video.jsonl", output_dir / "README.md")
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"refusing to overwrite performance artifacts under {output_dir}")
    profile = load(profile_path)
    if profile.get("input", {}).get("video_shape_fhw") != [81, 480, 832]:
        raise ValueError("invalid SeaCache4Wan21 Calflops profile")
    forward = profile.get("per_model_forward", profile.get("per_forward"))
    if not isinstance(forward, dict):
        raise ValueError("Calflops profile is missing per-forward costs")
    full = number(forward.get("estimated_full_flops"), "full FLOPs", positive=True)
    always = number(forward.get("estimated_always_on_flops"), "always-on FLOPs", positive=True)
    if always > full:
        raise ValueError("always-on FLOPs exceed full forward FLOPs")
    component_tflops = extract_component_tflops(profile)

    baseline_paths, candidate_paths = timing_map(baseline_dir), timing_map(candidate_dir)
    if set(baseline_paths) != set(candidate_paths) or len(baseline_paths) != args.expected_videos:
        raise ValueError(
            f"expected {args.expected_videos} matched timings; baseline={len(baseline_paths)}, candidate={len(candidate_paths)}"
        )
    baseline_rows, candidate_rows, thresholds = [], [], set()
    for sample_id in sorted(baseline_paths):
        baseline_rows.append(
            summarize_baseline(
                sample_id, baseline_paths[sample_id], full, component_tflops
            )
        )
        row, threshold = summarize_candidate(
            sample_id,
            candidate_paths[sample_id],
            candidate_dir / "traces" / f"{sample_id}.json",
            full,
            always,
            component_tflops,
        )
        candidate_rows.append(row)
        thresholds.add(threshold)
    if len(thresholds) != 1:
        raise ValueError(f"candidate traces contain multiple thresholds: {thresholds}")
    baseline = summarize_condition("baseline", baseline_rows)
    candidate = summarize_condition("seacache", candidate_rows)
    base_latency = float(baseline["pipeline_generate_wall_seconds"]["total"])
    sea_latency = float(candidate["pipeline_generate_wall_seconds"]["total"])
    base_flops = float(baseline["estimated_dit_total_tflops"])
    sea_flops = float(candidate["estimated_dit_total_tflops"])
    payload = {
        "schema": "seacache4wan21_vbench200_performance_v2",
        "threshold": next(iter(thresholds)),
        "protocol": {
            "model": "Wan2.1-T2V-1.3B", "video_shape_fhw": [81, 480, 832], "fps": 16,
            "sampling_steps": 50, "solver": "unipc", "shift": 5.0, "cfg": 5.0, "seed": 42,
        },
        "latency_definition": {
            "field": "pipeline_generate_wall_seconds",
            "includes": ["text_encoding", "denoising_cache_cfg_scheduler", "in_generate_transfers", "vae_decode"],
            "excludes": ["pipeline_construction", "model_loading", "mp4_export", "evaluation"],
        },
        "flops_definition": profile["scope"], "calflops_profile": str(profile_path),
        "conditions": {"baseline": baseline, "seacache": candidate},
        "comparison": {
            "latency_speedup_ratio_of_sums": base_latency / sea_latency,
            "latency_reduction_fraction": 1.0 - sea_latency / base_latency,
            "dit_flops_speedup_ratio_of_sums": base_flops / sea_flops,
            "dit_flops_reduction_fraction": 1.0 - sea_flops / base_flops,
        },
        "warnings": [
            "TFLOPs is an estimated DiT operation count (10^12 FLOPs), not measured hardware throughput.",
            "T5 and VAE decode TFLOPs and CUDA times are reported separately from the DiT headline.",
            "SeaCache FFT/gate and residual-add operations are outside the declared DiT profile scope.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per_video.jsonl").open("x", encoding="utf-8") as handle:
        for row in baseline_rows + candidate_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "README.md").write_text(
        "# SeaCache4Wan21 performance artifacts\n\n`summary.json` reports inference latency and trace-weighted estimated DiT TFLOPs; `per_video.jsonl` retains every sample.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
