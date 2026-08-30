#!/usr/bin/env python3
"""Read-only integrity audit for a Wan2.1 SeaCache collection archive."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from .collector import (
    all_baselines_complete,
    baseline_paths,
    candidate_paths,
    latent_bundle_complete,
    unique_prompt_rows,
    validate_flops_profile,
)
from .manifest import (
    NUM_STEPS,
    manifest_contract,
    read_jsonl,
    validate_candidate_manifest,
)
from .metrics import load_metric_artifacts
from .paths import require_result_path
from .performance import compare_matched, read_json, summarize_timing
from .publisher import load_completion, sha256


def close(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def optional_close(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    return (left is None and right is None) or close(left, right, tolerance)


def assert_inside(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact escapes archive root: {resolved}") from exc
    return resolved


def validate_ffprobe(path: Path) -> None:
    payload = read_json(path)
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError(f"invalid FFprobe stream list: {path}")
    stream = streams[0]
    if (int(stream.get("width", 0)), int(stream.get("height", 0)), int(stream.get("nb_read_frames", 0))) != (832, 480, 81):
        raise ValueError(f"invalid FFprobe geometry: {path}")
    numerator, denominator = str(stream.get("r_frame_rate", "0/1")).split("/", 1)
    if float(numerator) / float(denominator) != 16.0:
        raise ValueError(f"invalid FFprobe FPS: {path}")


def validate_timing_and_performance(
    timing_path: Path, performance_path: Path, profile: dict[str, Any], *, baseline: bool
) -> dict[str, Any]:
    timing = read_json(timing_path)
    performance = read_json(performance_path)
    expected = summarize_timing(timing, profile)
    if timing.get("status") != "success" or len(timing.get("calls", [])) != 2 * NUM_STEPS:
        raise ValueError(f"timing must contain 100 successful DiT calls: {timing_path}")
    block_count = int(profile["per_model_forward"]["transformer_blocks"])
    for call in timing["calls"]:
        executed = int(call.get("blocks_executed", -1))
        if executed not in {0, block_count}:
            raise ValueError(f"partial block execution is forbidden: {timing_path}")
        if baseline and executed != block_count:
            raise ValueError(f"baseline contains a reused DiT call: {timing_path}")
    for key in (
        "pipeline_generate_wall_seconds", "estimated_dit_flops",
        "estimated_dit_tflops_per_video", "t5_cuda_seconds",
        "vae_decode_cuda_seconds", "estimated_t5_tflops_per_video",
        "estimated_vae_decode_tflops_per_video",
    ):
        if not close(performance.get(key), expected.get(key)):
            raise ValueError(f"performance formula mismatch for {key}: {performance_path}")
    return performance


def validate_latents(trace: dict[str, Any], latent_dir: Path, *, deep: bool) -> None:
    step_records = trace.get("step_records")
    if not isinstance(step_records, list) or len(step_records) != NUM_STEPS:
        raise ValueError("trace lacks 50 step records")
    if not latent_bundle_complete(latent_dir):
        raise ValueError(f"latent bundle is incomplete: {latent_dir}")
    if not deep:
        return
    import torch

    for index, step in enumerate(step_records):
        path = latent_dir / f"step_{index:03d}_input.pt"
        try:
            tensor = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            tensor = torch.load(path, map_location="cpu")
        if not torch.is_tensor(tensor) or str(tensor.dtype) != "torch.float16":
            raise ValueError(f"latent tensor/dtype mismatch: {path}")
        if list(tensor.shape) != step.get("latent_shape") or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"latent shape/finite mismatch: {path}")


def validate_baseline(
    parent: Path, row: dict[str, Any], profile: dict[str, Any], *, deep_latents: bool
) -> dict[str, Any]:
    paths = baseline_paths(parent, str(row["sample_id"]))
    for key in ("video", "timing", "performance", "trace", "latents", "ffprobe", "complete"):
        assert_inside(paths[key], parent)
    marker = read_json(paths["complete"])
    if marker.get("schema") != "ours4wan21_baseline_complete_v1" or marker.get("sample_id") != row["sample_id"]:
        raise ValueError(f"baseline marker identity mismatch: {paths['complete']}")
    trace = read_json(paths["trace"])
    if trace.get("schema") != "ours4wan21_full_compute_trace_v1" or trace.get("recompute") != NUM_STEPS:
        raise ValueError(f"baseline trace mismatch: {paths['trace']}")
    if [int(item.get("step_index", -1)) for item in trace.get("step_records", [])] != list(range(NUM_STEPS)):
        raise ValueError(f"baseline step order mismatch: {paths['trace']}")
    validate_latents(trace, paths["latents"], deep=deep_latents)
    validate_ffprobe(paths["ffprobe"])
    return validate_timing_and_performance(paths["timing"], paths["performance"], profile, baseline=True)


def validate_candidate(
    parent: Path,
    row: dict[str, Any],
    profile: dict[str, Any],
    baseline_performance: dict[str, Any],
    *,
    deep_latents: bool,
) -> None:
    paths = candidate_paths(parent, int(row["shard_index"]), str(row["trajectory_id"]))
    for key in (
        "video", "timing", "performance", "trace", "latents", "ffprobe",
        "metrics_root", "metrics", "metrics_per_frame", "metrics_per_video",
        "metrics_summary", "complete",
    ):
        assert_inside(paths[key], parent)
    completion = load_completion(parent, row)
    if completion is None:
        raise FileNotFoundError(paths["complete"])
    trace = read_json(paths["trace"])
    fixed_seacache = row.get("policy_family") == "fixed_seacache_threshold"
    expected_trace_schema = (
        "ours4wan21_seacache_fixed_threshold_trace_v1"
        if fixed_seacache
        else "ours4wan21_random_threshold_trace_v2"
    )
    expected_gate_mode = (
        "seacache_aligned_independent_cfg_branches_filtered_boundary_fixed_threshold"
        if fixed_seacache
        else "seacache_aligned_independent_cfg_branches_filtered_boundary_dynamic_threshold"
    )
    if trace.get("schema") != expected_trace_schema:
        raise ValueError(f"candidate trace schema mismatch: {paths['trace']}")
    if trace.get("gate_mode") != expected_gate_mode:
        raise ValueError(f"candidate gate mode mismatch: {paths['trace']}")
    if trace.get("policy_family") != row.get("policy_family"):
        raise ValueError(f"candidate trace policy-family mismatch: {paths['trace']}")
    if fixed_seacache and not close(trace.get("fixed_threshold"), row.get("fixed_threshold")):
        raise ValueError(f"candidate fixed-threshold provenance mismatch: {paths['trace']}")
    if trace.get("count_unit") != "cfg_branch_call":
        raise ValueError(f"candidate action count unit mismatch: {paths['trace']}")
    expected_distance_contract = {
        "feature": "sea_filtered_first_block_modulated_input",
        "reference": "previous_step_same_cfg_branch",
        "metric": "relative_l1_mean",
        "accumulation": "sum_since_last_recompute_excluding_forced_boundary",
        "threshold_operand": "accumulated_distance_with_current",
        "reset": "zero_after_recompute",
    }
    if trace.get("distance_contract") != expected_distance_contract:
        raise ValueError(f"candidate distance contract mismatch: {paths['trace']}")
    expected_path = [float(value) for value in row["threshold_path"]]
    observed_path = [float(value) for value in trace.get("threshold_path", [])]
    if len(observed_path) != NUM_STEPS or any(not close(a, b) for a, b in zip(observed_path, expected_path)):
        raise ValueError(f"candidate threshold path mismatch: {paths['trace']}")
    decisions = trace.get("decisions", [])
    if len(decisions) != 2 * NUM_STEPS:
        raise ValueError(f"candidate decision count mismatch: {paths['trace']}")
    previous_after = {"cond": 0.0, "uncond": 0.0}
    for call_index, decision in enumerate(decisions):
        step_index = call_index // 2
        branch = ("cond", "uncond")[call_index % 2]
        if (
            int(decision.get("call_index", -1)) != call_index
            or int(decision.get("step_index", -1)) != step_index
            or decision.get("branch") != branch
            or not close(decision.get("requested_threshold"), expected_path[step_index])
        ):
            raise ValueError(f"candidate decision/threshold mismatch: {paths['trace']}")
        action = decision.get("action")
        if action not in {"reuse", "recompute"} or decision.get("execution") != action:
            raise ValueError(f"CFG branch action/execution mismatch: {paths['trace']}")
        if (
            decision.get("stored_feature") != "sea_filtered"
            or decision.get("distance_reference") != "previous_step_same_cfg_branch"
            or decision.get("distance_feature") != "sea_filtered_first_block_modulated_input"
            or decision.get("distance_metric") != "relative_l1_mean"
        ):
            raise ValueError(f"filtered distance provenance mismatch: {paths['trace']}")
        before = decision.get("accumulated_distance_before")
        distance = decision.get("filtered_relative_l1")
        with_current = decision.get("accumulated_distance_with_current")
        after = decision.get("accumulated_distance_after")
        if not close(before, previous_after[branch]) or float(before) < 0.0:
            raise ValueError(f"accumulator branch chain mismatch: {paths['trace']}")
        for explicit, alias in (
            (distance, decision.get("relative_l1")),
            (before, decision.get("accumulator_before")),
            (with_current, decision.get("accumulator_with_current")),
            (after, decision.get("accumulator_after")),
        ):
            if not optional_close(explicit, alias):
                raise ValueError(f"distance field alias mismatch: {paths['trace']}")
        forced = step_index in {0, NUM_STEPS - 1}
        if bool(decision.get("native_forced_recompute")) != forced:
            raise ValueError(f"native boundary flag mismatch: {paths['trace']}")
        if forced:
            if (
                action != "recompute"
                or distance is not None
                or with_current is not None
                or not close(after, 0.0)
                or decision.get("reason") != "forced_boundary"
            ):
                raise ValueError(f"native boundary recompute mismatch: {paths['trace']}")
        else:
            if (
                not isinstance(distance, (int, float))
                or not math.isfinite(float(distance))
                or float(distance) < 0.0
                or not close(with_current, float(before) + float(distance))
            ):
                raise ValueError(f"filtered distance accumulation mismatch: {paths['trace']}")
            if action == "reuse":
                if (
                    not float(with_current) < expected_path[step_index]
                    or not close(after, with_current)
                    or decision.get("reason") != "accumulator_below_requested_threshold"
                ):
                    raise ValueError(f"reuse threshold transition mismatch: {paths['trace']}")
            elif (
                float(with_current) < expected_path[step_index]
                or not close(after, 0.0)
                or decision.get("reason") != "requested_threshold_reached"
            ):
                raise ValueError(f"recompute threshold transition mismatch: {paths['trace']}")
        previous_after[branch] = float(after)
    if int(trace.get("total_steps", -1)) != NUM_STEPS or int(trace.get("total_branch_calls", -1)) != 2 * NUM_STEPS:
        raise ValueError(f"candidate trace totals mismatch: {paths['trace']}")
    observed_reuse = sum(decision["action"] == "reuse" for decision in decisions)
    observed_recompute = 2 * NUM_STEPS - observed_reuse
    if int(trace.get("reuse", -1)) != observed_reuse or int(trace.get("recompute", -1)) != observed_recompute:
        raise ValueError(f"candidate trace action counts mismatch: {paths['trace']}")
    for branch in ("cond", "uncond"):
        branch_rows = [decision for decision in decisions if decision["branch"] == branch]
        expected_summary = {
            "reuse": sum(decision["action"] == "reuse" for decision in branch_rows),
            "recompute": sum(decision["action"] == "recompute" for decision in branch_rows),
            "reuse_path": [decision["step_index"] for decision in branch_rows if decision["action"] == "reuse"],
            "recompute_path": [decision["step_index"] for decision in branch_rows if decision["action"] == "recompute"],
        }
        if trace.get("per_branch", {}).get(branch) != expected_summary:
            raise ValueError(f"candidate per-branch summary mismatch: {paths['trace']}")
    step_records = trace.get("step_records", [])
    if len(step_records) != NUM_STEPS:
        raise ValueError(f"candidate step record count mismatch: {paths['trace']}")
    for step_index, step in enumerate(step_records):
        branch_decisions = {
            "cond": decisions[2 * step_index],
            "uncond": decisions[2 * step_index + 1],
        }
        actions = {branch: decision["action"] for branch, decision in branch_decisions.items()}
        aggregate = actions["cond"] if actions["cond"] == actions["uncond"] else "mixed"
        if (
            int(step.get("step_index", -1)) != step_index
            or step.get("branch_decisions") != branch_decisions
            or step.get("branches") != actions
            or step.get("action") != aggregate
            or step.get("filtered_relative_l1") != {
                branch: decision["filtered_relative_l1"]
                for branch, decision in branch_decisions.items()
            }
            or step.get("accumulated_distance_before") != {
                branch: decision["accumulated_distance_before"]
                for branch, decision in branch_decisions.items()
            }
            or step.get("accumulated_distance_with_current") != {
                branch: decision["accumulated_distance_with_current"]
                for branch, decision in branch_decisions.items()
            }
            or step.get("accumulated_distance_after") != {
                branch: decision["accumulated_distance_after"]
                for branch, decision in branch_decisions.items()
            }
        ):
            raise ValueError(f"candidate step aggregation mismatch: {paths['trace']}")
        for branch in ("cond", "uncond"):
            decision = branch_decisions[branch]
            checks = {
                f"{branch}_action": decision["action"],
                f"{branch}_filtered_relative_l1": decision["filtered_relative_l1"],
                f"{branch}_accumulated_distance_before": decision["accumulated_distance_before"],
                f"{branch}_accumulated_distance_with_current": decision["accumulated_distance_with_current"],
                f"{branch}_accumulated_distance_after": decision["accumulated_distance_after"],
            }
            for key, expected in checks.items():
                observed = step.get(key)
                if not (
                    observed == expected
                    if isinstance(expected, str)
                    else optional_close(observed, expected)
                ):
                    raise ValueError(f"candidate step field mismatch for {key}: {paths['trace']}")
    validate_latents(trace, paths["latents"], deep=deep_latents)
    validate_ffprobe(paths["ffprobe"])
    candidate_performance = validate_timing_and_performance(
        paths["timing"], paths["performance"], profile, baseline=False
    )
    timing = read_json(paths["timing"])
    block_count = int(profile["per_model_forward"]["transformer_blocks"])
    for decision, call in zip(decisions, timing["calls"], strict=True):
        expected_cfg_branch = "condition" if decision["branch"] == "cond" else "uncondition"
        expected_blocks = 0 if decision["action"] == "reuse" else block_count
        if (
            int(call.get("call_index", -1)) != int(decision["call_index"])
            or int(call.get("step_index", -1)) != int(decision["step_index"])
            or call.get("cfg_branch") != expected_cfg_branch
            or int(call.get("blocks_executed", -1)) != expected_blocks
        ):
            raise ValueError(f"trace/timing branch execution mismatch: {paths['timing']}")
    if (
        int(candidate_performance.get("reuse_forward_calls", -1)) != observed_reuse
        or int(candidate_performance.get("full_compute_forward_calls", -1)) != observed_recompute
    ):
        raise ValueError(f"trace/performance action count mismatch: {paths['performance']}")
    comparison = compare_matched(baseline_performance, candidate_performance)
    metric_result = load_metric_artifacts(paths["root"])
    psnr = metric_result["metrics"]["psnr_rgb_db"]
    ssim = metric_result["metrics"]["ssim_rgb"]
    lpips = metric_result["metrics"]["lpips_alex_v0_1_spatial"]
    trajectory = completion["trajectory_row"]
    if trajectory.get("policy_family") != row.get("policy_family"):
        raise ValueError(f"completion policy-family mismatch: {paths['complete']}")
    if fixed_seacache and not close(
        trajectory.get("fixed_threshold"), row.get("fixed_threshold")
    ):
        raise ValueError(f"completion fixed-threshold mismatch: {paths['complete']}")
    branch_reuse = {
        branch: sum(
            decision["branch"] == branch and decision["action"] == "reuse"
            for decision in decisions
        )
        for branch in ("cond", "uncond")
    }
    mixed_steps = sum(step["action"] == "mixed" for step in step_records)
    both_reuse_steps = sum(step["action"] == "reuse" for step in step_records)
    both_recompute_steps = sum(step["action"] == "recompute" for step in step_records)
    if trajectory.get("action_count_unit") != "cfg_branch_call":
        raise ValueError(f"completion action count unit mismatch: {paths['complete']}")
    if trajectory.get("video_metrics_protocol") != metric_result["protocol_id"]:
        raise ValueError(f"completion metric protocol mismatch: {paths['complete']}")
    checks = {
        "baseline_inference_seconds": baseline_performance["pipeline_generate_wall_seconds"],
        "candidate_inference_seconds": candidate_performance["pipeline_generate_wall_seconds"],
        "inference_latency_speedup": comparison["inference_latency_speedup"],
        "baseline_estimated_dit_tflops": baseline_performance["estimated_dit_tflops_per_video"],
        "candidate_estimated_dit_tflops": candidate_performance["estimated_dit_tflops_per_video"],
        "dit_flops_speedup": comparison["dit_flops_speedup"],
        "baseline_dit_cuda_seconds": baseline_performance["dit_cuda_seconds"],
        "candidate_dit_cuda_seconds": candidate_performance["dit_cuda_seconds"],
        "baseline_t5_cuda_seconds": baseline_performance["t5_cuda_seconds"],
        "candidate_t5_cuda_seconds": candidate_performance["t5_cuda_seconds"],
        "baseline_vae_decode_cuda_seconds": baseline_performance[
            "vae_decode_cuda_seconds"
        ],
        "candidate_vae_decode_cuda_seconds": candidate_performance[
            "vae_decode_cuda_seconds"
        ],
        "baseline_estimated_t5_tflops_per_video": baseline_performance[
            "estimated_t5_tflops_per_video"
        ],
        "candidate_estimated_t5_tflops_per_video": candidate_performance[
            "estimated_t5_tflops_per_video"
        ],
        "baseline_estimated_vae_decode_tflops_per_video": baseline_performance[
            "estimated_vae_decode_tflops_per_video"
        ],
        "candidate_estimated_vae_decode_tflops_per_video": candidate_performance[
            "estimated_vae_decode_tflops_per_video"
        ],
        "mean_psnr": psnr["mean"],
        "std_psnr": psnr["std_population"],
        "min_psnr": psnr["min"],
        "max_psnr": psnr["max"],
        "mean_ssim": ssim["mean"],
        "std_ssim": ssim["std_population"],
        "min_ssim": ssim["min"],
        "max_ssim": ssim["max"],
        "mean_lpips": lpips["mean"],
        "std_lpips": lpips["std_population"],
        "min_lpips": lpips["min"],
        "max_lpips": lpips["max"],
        "metric_frames": metric_result["frames"],
        "actual_reuse": observed_reuse,
        "actual_recompute": observed_recompute,
        "actual_reuse_branch_calls": observed_reuse,
        "actual_recompute_branch_calls": observed_recompute,
        "cond_reuse_branch_calls": branch_reuse["cond"],
        "cond_recompute_branch_calls": NUM_STEPS - branch_reuse["cond"],
        "uncond_reuse_branch_calls": branch_reuse["uncond"],
        "uncond_recompute_branch_calls": NUM_STEPS - branch_reuse["uncond"],
        "actual_both_reuse_steps": both_reuse_steps,
        "actual_both_recompute_steps": both_recompute_steps,
        "actual_mixed_steps": mixed_steps,
    }
    for key, expected in checks.items():
        if not close(trajectory.get(key), expected):
            raise ValueError(f"completion metric mismatch for {key}: {paths['complete']}")
    baseline_latent_dir = baseline_paths(parent, str(row["sample_id"]))["latents"]
    for index, step in enumerate(completion["step_rows"]):
        expected = str((baseline_latent_dir / f"step_{index:03d}_input.pt").resolve())
        if step.get("baseline_latent_path") != expected:
            raise ValueError(f"baseline latent pairing mismatch: {paths['complete']}")
        for key, metric_value in (
            ("final_mean_psnr", psnr["mean"]),
            ("final_mean_ssim", ssim["mean"]),
            ("final_mean_lpips", lpips["mean"]),
        ):
            if not close(step.get(key), metric_value):
                raise ValueError(f"step metric mismatch for {key}: {paths['complete']}")
        if step.get("branch_decisions") != step_records[index].get("branch_decisions"):
            raise ValueError(f"completion/trace step decision mismatch: {paths['complete']}")
    for decision, branch_row in zip(decisions, completion["branch_rows"], strict=True):
        for key in (
            "call_index", "step_index", "branch", "requested_threshold", "action", "reason",
            "filtered_relative_l1", "accumulated_distance_before",
            "accumulated_distance_with_current", "accumulated_distance_after",
            "distance_reference", "distance_feature", "distance_metric", "stored_feature",
            "native_forced_recompute", "execution",
        ):
            observed = branch_row.get(key)
            expected = decision.get(key)
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                matches = optional_close(observed, expected)
            else:
                matches = observed == expected
            if not matches:
                raise ValueError(f"completion/trace branch field mismatch for {key}: {paths['complete']}")


def audit(
    manifest: Path,
    parent: Path,
    profile_path: Path,
    output: Path,
    *,
    require_complete: bool,
    deep_latents: bool,
    max_candidates: int | None,
) -> dict[str, Any]:
    manifest = manifest.expanduser().resolve(strict=True)
    parent = require_result_path(parent)
    profile_path = profile_path.expanduser().resolve(strict=True)
    output = require_result_path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {output}")
    rows = read_jsonl(manifest)
    validate_candidate_manifest(rows)
    contract = manifest_contract(rows)
    expected_candidate_count = int(contract["candidate_count"])
    profile = validate_flops_profile(profile_path)
    ready, missing = all_baselines_complete(rows, parent)
    if not ready:
        raise RuntimeError(f"archive has {len(missing)} incomplete baselines")
    baseline_performance: dict[str, dict[str, Any]] = {}
    for row in unique_prompt_rows(rows):
        baseline_performance[str(row["sample_id"])] = validate_baseline(
            parent, row, profile, deep_latents=deep_latents
        )
    available: list[dict[str, Any]] = []
    for row in rows:
        if load_completion(parent, row) is None:
            break
        available.append(row)
    if require_complete and len(available) != expected_candidate_count:
        raise RuntimeError(
            "complete audit requires "
            f"{expected_candidate_count} candidates; prefix={len(available)}"
        )
    vbench_summary = None
    if require_complete:
        vbench_path = parent / "quality" / "vbench_summary.json"
        vbench_summary = read_json(vbench_path)
        if (
            vbench_summary.get("schema") != "ours4wan21_vbench_summary_v1"
            or vbench_summary.get("protocol") != "vbench_custom_input_raw_mean_v1"
            or int(vbench_summary.get("baseline_video_count", -1)) != len(baseline_performance)
            or int(vbench_summary.get("candidate_video_count", -1))
            != expected_candidate_count
        ):
            raise ValueError(f"invalid archive VBench summary: {vbench_path}")
        for condition in ("baseline", "candidate"):
            record = vbench_summary.get(condition, {})
            score_path = Path(record.get("path", "")).resolve(strict=True)
            if sha256(score_path) != record.get("sha256"):
                raise ValueError(f"{condition} VBench score checksum mismatch")
            score_payload = read_json(score_path)
            if (
                score_payload.get("protocol") != "vbench_custom_input_raw_mean_v1"
                or not close(score_payload.get("vbench_score"), record.get("vbench_score"))
            ):
                raise ValueError(f"{condition} VBench score mismatch")
    selected = available[:max_candidates] if max_candidates is not None else available
    for row in selected:
        validate_candidate(
            parent,
            row,
            profile,
            baseline_performance[str(row["sample_id"])],
            deep_latents=deep_latents,
        )
    payload = {
        "schema": "ours4wan21_archive_audit_v2",
        "status": "ok",
        "manifest": str(manifest),
        "archive_root": str(parent),
        "flops_profile": str(profile_path),
        "baseline_count_audited": len(baseline_performance),
        "contiguous_candidate_prefix": len(available),
        "candidate_count_audited": len(selected),
        "deep_latents": deep_latents,
        "require_complete": require_complete,
        "vbench_summary": vbench_summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--flops-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--deep-latents", action="store_true")
    parser.add_argument("--max-candidates", type=int)
    args = parser.parse_args()
    if args.max_candidates is not None and args.max_candidates < 0:
        raise ValueError("--max-candidates must be nonnegative")
    result = audit(
        args.manifest, args.parent_root, args.flops_profile, args.output,
        require_complete=args.require_complete,
        deep_latents=args.deep_latents,
        max_candidates=args.max_candidates,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
