#!/usr/bin/env python3
"""Aggregate Wan2.2 inference latency and trace-weighted Calflops TFLOPs."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from compare_runs import load_json, validate_manifest
REPOSITORY_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPOSITORY_DIR / "ComponentMetrics"))
from reporting import extract_component_latency, extract_component_tflops  # noqa: E402


TFLOP_DIVISOR = 1_000_000_000_000
STAGES = ("high", "low")
BRANCHES = ("cond", "uncond")


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


def load_profile(path: Path) -> dict[str, Any]:
    profile = load_json(path)
    if profile.get("schema") != "teacache4wan22_calflops_profile_v2":
        raise ValueError("invalid TeaCache4Wan22 Calflops profile schema")
    profile_input = profile.get("input", {})
    if profile_input.get("video_shape_fhw") != [45, 480, 832]:
        raise ValueError("Calflops profile does not match the locked 45x480x832 shape")
    if profile_input.get("stage_steps") != {"high": 32, "low": 18}:
        raise ValueError("Calflops profile does not match the 32/18 stage split")
    stages = profile.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(STAGES):
        raise ValueError("Calflops profile must contain high and low stages")
    for stage in STAGES:
        branches = stages[stage].get("branches")
        if not isinstance(branches, dict) or set(branches) != set(BRANCHES):
            raise ValueError(f"Calflops profile must contain {stage} cond/uncond paths")
        for branch in BRANCHES:
            forward = branches[branch]
            full = finite_nonnegative(
                forward.get("estimated_full_flops"),
                f"profile.{stage}.{branch}.estimated_full_flops",
            )
            always_on = finite_nonnegative(
                forward.get("estimated_always_on_flops"),
                f"profile.{stage}.{branch}.estimated_always_on_flops",
            )
            if always_on > full:
                raise ValueError(
                    f"{stage}/{branch} always-on FLOPs exceed full-forward FLOPs"
                )
        for key in ("estimated_full_flops", "estimated_always_on_flops"):
            if branches["cond"][key] != branches["uncond"][key]:
                raise ValueError(f"{stage} cond/uncond profile mismatch for {key}")
        block_count = profile_input.get("transformer_blocks")
        stage_blocks = stages[stage].get("model", {}).get("transformer_blocks")
        if (
            isinstance(block_count, bool)
            or not isinstance(block_count, int)
            or block_count < 1
            or stage_blocks != block_count
        ):
            raise ValueError(f"invalid Transformer block count for {stage}")
    extract_component_tflops(profile)
    return profile


def expected_call_identity(call_index: int) -> tuple[int, str, str]:
    step_index = call_index // 2
    branch = BRANCHES[call_index % 2]
    stage = "high" if step_index < 32 else "low"
    return step_index, stage, branch


def summarize_manifest(
    *,
    manifest_path: Path,
    expected_method: str,
    profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, inference_seconds = validate_manifest(manifest_path, expected_method)
    timing_payload = manifest["timing"]["payload"]
    calls = timing_payload.get("calls")
    if not isinstance(calls, list) or len(calls) != 100:
        raise ValueError(
            f"fixed 50-step CFG protocol requires 100 DiT calls: {manifest_path}"
        )

    expected_block_count = int(profile["input"]["transformer_blocks"])
    if timing_payload.get("transformer_block_count_by_stage") != {
        "high": expected_block_count,
        "low": expected_block_count,
    }:
        raise ValueError(f"timing/profile block-count mismatch: {manifest_path}")

    trace_decisions = None
    if expected_method == "teacache":
        trace_decisions = load_json(Path(manifest["trace"]["path"])).get("decisions")
        if not isinstance(trace_decisions, list) or len(trace_decisions) != 50:
            raise ValueError(f"invalid TeaCache decisions: {manifest_path}")

    estimated_flops = 0.0
    transformer_block_executions = 0
    cuda_seconds_sum = 0.0
    for call_index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise TypeError(f"DiT call {call_index} is not an object: {manifest_path}")
        step_index, stage, branch = expected_call_identity(call_index)
        observed_identity = (
            call.get("step_index"),
            call.get("model_stage"),
            call.get("cfg_branch"),
        )
        if observed_identity != (step_index, stage, branch):
            raise ValueError(
                f"DiT call ordering mismatch at {call_index}: {observed_identity}"
            )
        blocks_executed = call.get("blocks_executed")
        if blocks_executed not in (0, expected_block_count):
            raise ValueError(
                f"TeaCache4Wan22 supports only all-block or zero-block calls: {call}"
            )
        if expected_method == "none" and blocks_executed != expected_block_count:
            raise ValueError(f"baseline contains a reuse call: {manifest_path}")
        if trace_decisions is not None:
            action = trace_decisions[step_index].get("action")
            if action not in {"recompute", "reuse"}:
                raise ValueError(
                    f"invalid TeaCache action at step {step_index}: {action!r}"
                )
            expected_blocks = 0 if action == "reuse" else expected_block_count
            if blocks_executed != expected_blocks:
                raise ValueError(
                    f"DiT block trace disagrees with TeaCache decision at step {step_index}"
                )
        full_flag = blocks_executed == expected_block_count
        reuse_flag = blocks_executed == 0
        if call.get("full_compute") is not full_flag or call.get("reuse") is not reuse_flag:
            raise ValueError(f"invalid full/reuse flags at DiT call {call_index}")
        cuda_seconds_sum += finite_nonnegative(
            call.get("cuda_seconds"), f"call[{call_index}].cuda_seconds"
        )
        transformer_block_executions += int(blocks_executed)
        forward = profile["stages"][stage]["branches"][branch]
        estimated_flops += finite_nonnegative(
            (
                forward["estimated_full_flops"]
                if full_flag
                else forward["estimated_always_on_flops"]
            ),
            f"{stage} call FLOPs",
        )

    recorded_cuda_seconds = finite_nonnegative(
        timing_payload.get("model_forward_cuda_seconds"),
        f"{manifest_path}.model_forward_cuda_seconds",
    )
    if not math.isclose(
        cuda_seconds_sum,
        recorded_cuda_seconds,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise ValueError(f"DiT CUDA call sum mismatch: {manifest_path}")
    full_calls = sum(int(call["full_compute"]) for call in calls)
    reuse_calls = sum(int(call["reuse"]) for call in calls)
    if (
        timing_payload.get("full_compute_forward_calls") != full_calls
        or timing_payload.get("reuse_forward_calls") != reuse_calls
    ):
        raise ValueError(f"DiT call summary mismatch: {manifest_path}")

    estimated_tflops = estimated_flops / TFLOP_DIVISOR
    component_latency = extract_component_latency(timing_payload)
    component_tflops = extract_component_tflops(profile)
    row = {
        "condition": "baseline" if expected_method == "none" else "teacache",
        "prompt": manifest["prompt"],
        "manifest_path": str(manifest_path.resolve()),
        "pipeline_generate_wall_seconds": inference_seconds,
        "dit_forward_cuda_seconds": recorded_cuda_seconds,
        **component_latency,
        "model_forward_calls": len(calls),
        "transformer_block_executions": transformer_block_executions,
        "full_compute_forward_calls": full_calls,
        "reuse_forward_calls": reuse_calls,
        "estimated_dit_flops": estimated_flops,
        "estimated_dit_tflops": estimated_tflops,
        **component_tflops,
        "estimated_achieved_dit_tflops_per_second": (
            estimated_tflops / recorded_cuda_seconds
            if recorded_cuda_seconds > 0
            else None
        ),
    }
    return manifest, row


def summarize_condition(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    inference = [float(row["pipeline_generate_wall_seconds"]) for row in rows]
    cuda = [float(row["dit_forward_cuda_seconds"]) for row in rows]
    tflops = [float(row["estimated_dit_tflops"]) for row in rows]
    t5_cuda = [float(row["t5_cuda_seconds"]) for row in rows]
    vae_cuda = [float(row["vae_decode_cuda_seconds"]) for row in rows]
    total_tflops = sum(tflops)
    total_cuda = sum(cuda)
    return {
        "condition": label,
        "video_count": len(rows),
        "pipeline_generate_wall_seconds": summary_stats(inference),
        "dit_forward_cuda_seconds": summary_stats(cuda),
        "t5_cuda_seconds": summary_stats(t5_cuda),
        "vae_decode_cuda_seconds": summary_stats(vae_cuda),
        "estimated_dit_tflops_per_video": summary_stats(tflops),
        "estimated_t5_tflops_per_video": rows[0]["estimated_t5_tflops_per_video"],
        "estimated_vae_decode_tflops_per_video": rows[0]["estimated_vae_decode_tflops_per_video"],
        "estimated_dit_total_tflops": total_tflops,
        "estimated_achieved_dit_tflops_per_second_ratio_of_sums": (
            total_tflops / total_cuda if total_cuda > 0 else None
        ),
        "total_full_compute_forward_calls": sum(
            int(row["full_compute_forward_calls"]) for row in rows
        ),
        "total_reuse_forward_calls": sum(
            int(row["reuse_forward_calls"]) for row in rows
        ),
        "total_transformer_block_executions": sum(
            int(row["transformer_block_executions"]) for row in rows
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-manifest", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--teacache-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--calflops-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.baseline_manifest) != len(args.teacache_manifest):
        raise ValueError("baseline and TeaCache manifest counts must match")
    if not args.baseline_manifest:
        raise ValueError("at least one matched manifest pair is required")
    summary_path = args.output_dir / "summary.json"
    per_video_path = args.output_dir / "per_video.jsonl"
    readme_path = args.output_dir / "README.md"
    if summary_path.exists() or per_video_path.exists() or readme_path.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")

    profile = load_profile(args.calflops_profile)
    baseline_rows = []
    teacache_rows = []
    thresholds = set()
    for baseline_path, teacache_path in zip(
        args.baseline_manifest, args.teacache_manifest
    ):
        baseline, baseline_row = summarize_manifest(
            manifest_path=baseline_path,
            expected_method="none",
            profile=profile,
        )
        teacache, teacache_row = summarize_manifest(
            manifest_path=teacache_path,
            expected_method="teacache",
            profile=profile,
        )
        if baseline["prompt"] != teacache["prompt"]:
            raise ValueError("matched manifests use different prompts")
        if baseline["protocol"] != teacache["protocol"]:
            raise ValueError("matched manifests use different protocols")
        if baseline["checkpoint"] != teacache["checkpoint"]:
            raise ValueError("matched manifests use different checkpoints")
        if baseline["prepared_source_manifest"]["sha256"] != teacache[
            "prepared_source_manifest"
        ]["sha256"]:
            raise ValueError("matched manifests use different prepared source trees")
        profile_checkpoint = Path(profile["source"]["checkpoint_dir"]).resolve()
        if Path(baseline["checkpoint"]).resolve() != profile_checkpoint:
            raise ValueError("Calflops profile and run manifests use different checkpoints")
        profile_prepared = profile["source"].get("prepared_manifest", {})
        run_prepared = baseline["prepared_source_manifest"].get("payload", {})
        for key in (
            "wan22_commit",
            "patch_sha256",
            "runtime_sha256",
            "timing_runtime_sha256",
            "protocol_sha256",
        ):
            if profile_prepared.get(key) != run_prepared.get(key):
                raise ValueError(
                    f"Calflops profile and run manifest source lock differ for {key}"
                )
        thresholds.add(float(teacache["threshold"]))
        baseline_rows.append(baseline_row)
        teacache_rows.append(teacache_row)

    if len(thresholds) != 1:
        raise ValueError("all TeaCache runs in one aggregate must use one threshold")
    baseline_summary = summarize_condition("baseline", baseline_rows)
    teacache_summary = summarize_condition("teacache", teacache_rows)
    baseline_latency = baseline_summary["pipeline_generate_wall_seconds"]["total"]
    teacache_latency = teacache_summary["pipeline_generate_wall_seconds"]["total"]
    baseline_tflops = baseline_summary["estimated_dit_total_tflops"]
    teacache_tflops = teacache_summary["estimated_dit_total_tflops"]
    payload = {
        "schema_version": 2,
        "protocol": {
            "model": "Wan2.2-T2V-A14B",
            "video": {"width": 832, "height": 480, "frames": 45, "fps": 16},
            "sampling": {
                "steps": 50,
                "solver": "dpm++",
                "shift": 12.0,
                "guide_scale_low_high": [3.0, 4.0],
                "boundary": 0.875,
                "seed": 42,
            },
            "precision": "DiT bfloat16",
            "memory": "single GPU, model offload enabled, T5 on GPU",
        },
        "latency_definition": {
            "headline_field": "pipeline_generate_wall_seconds",
            "includes": [
                "text_encoding",
                "denoising_cache_cfg_scheduler",
                "cache_state_release",
                "model_weight_transfer_and_offload",
                "vae_decode",
            ],
            "excludes": [
                "model_loading",
                "mp4_export",
                "file_io",
                "metric_evaluation",
            ],
            "aggregation": "ratio of sums over matched prompts",
        },
        "flops_definition": profile.get("scope"),
        "calflops_profile": str(args.calflops_profile.resolve()),
        "threshold": next(iter(thresholds)),
        "conditions": {
            "baseline": baseline_summary,
            "teacache": teacache_summary,
        },
        "comparison": {
            "latency_speedup_ratio_of_sums": baseline_latency / teacache_latency,
            "latency_reduction_fraction": 1.0 - teacache_latency / baseline_latency,
            "dit_flops_speedup_ratio_of_sums": baseline_tflops / teacache_tflops,
            "dit_flops_reduction_fraction": 1.0 - teacache_tflops / baseline_tflops,
        },
        "warnings": [
            "TFLOPs denotes 10^12 floating-point operations; TFLOP/s denotes achieved estimated DiT throughput.",
            "DiT remains the headline TFLOPs metric; T5 and VAE decode CUDA time and TFLOPs are retained as separate component fields.",
            "TeaCache controller and residual-add FLOPs are excluded as negligible and must not be described as complete end-to-end FLOPs.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(summary_path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    with per_video_path.open("x", encoding="utf-8") as handle:
        for row in baseline_rows + teacache_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    readme_path.write_text(
        "# TeaCache4Wan22 performance result\n\n"
        "`summary.json` reports inference-only latency and Calflops-based DiT "
        "TFLOPs/TFLOP/s. `per_video.jsonl` retains each matched sample. Model "
        "loading, MP4 export, file I/O, and metric evaluation are excluded from "
        "headline latency; transfers inside `generate()` remain included.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
