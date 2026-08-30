#!/usr/bin/env python3
"""Validate matched baseline/TeaCache runs and report inference-only speedup."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_DIR / "ComponentMetrics"))
from reporting import extract_component_latency  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_artifact(record: dict[str, Any], label: str) -> Path:
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = sha256(path)
    if observed != record["sha256"]:
        raise ValueError(f"{label} SHA256 mismatch: {path}")
    return path


def validate_timing_payload(payload: dict[str, Any], expected_method: str) -> float:
    if payload.get("schema_version") != 2:
        raise ValueError("invalid timing schema version")
    if payload.get("status") != "success" or payload.get("error") is not None:
        raise ValueError("timing trace does not describe a successful generation")
    expected_implementation = "teacache" if expected_method == "teacache" else "wan22"
    if payload.get("implementation") != expected_implementation:
        raise ValueError("timing implementation mismatch")
    for key in ("pipeline_init_wall_seconds", "pipeline_generate_wall_seconds"):
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"invalid {key} in timing trace")

    block_counts = payload.get("transformer_block_count_by_stage")
    if not isinstance(block_counts, dict) or set(block_counts) != {"high", "low"}:
        raise ValueError("invalid high/low Transformer block counts")
    for stage, value in block_counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid Transformer block count for {stage}")
    calls = payload.get("calls")
    if not isinstance(calls, list) or len(calls) != 100:
        raise ValueError("fixed 50-step CFG timing requires 100 DiT calls")
    cuda_sum = 0.0
    full_calls = 0
    reuse_calls = 0
    for call_index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise TypeError(f"DiT call {call_index} is not an object")
        step_index = call_index // 2
        expected = (
            call_index,
            step_index,
            "high" if step_index < 32 else "low",
            "cond" if call_index % 2 == 0 else "uncond",
        )
        observed = (
            call.get("call_index"),
            call.get("step_index"),
            call.get("model_stage"),
            call.get("cfg_branch"),
        )
        if observed != expected:
            raise ValueError(
                f"DiT call identity mismatch: expected={expected}, observed={observed}"
            )
        block_count = block_counts[call["model_stage"]]
        executed = call.get("blocks_executed")
        if executed not in (0, block_count):
            raise ValueError("TeaCache4Wan22 DiT calls must execute all or zero blocks")
        full = executed == block_count
        reuse = executed == 0
        if call.get("full_compute") is not full or call.get("reuse") is not reuse:
            raise ValueError(f"invalid full/reuse flags at DiT call {call_index}")
        full_calls += int(full)
        reuse_calls += int(reuse)
        for key in ("host_span_seconds", "cuda_seconds"):
            value = call.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {key} at DiT call {call_index}")
        cuda_sum += float(call["cuda_seconds"])
    if payload.get("model_forward_call_count") != len(calls):
        raise ValueError("DiT model-forward call count mismatch")
    if payload.get("full_compute_forward_calls") != full_calls:
        raise ValueError("DiT full-compute call count mismatch")
    if payload.get("reuse_forward_calls") != reuse_calls:
        raise ValueError("DiT reuse call count mismatch")
    if expected_method == "none" and (full_calls != len(calls) or reuse_calls != 0):
        raise ValueError("baseline timing trace contains a reuse call")
    total_cuda = payload.get("model_forward_cuda_seconds")
    if (
        not isinstance(total_cuda, (int, float))
        or not math.isfinite(total_cuda)
        or total_cuda < 0
        or not math.isclose(cuda_sum, float(total_cuda), rel_tol=1e-12, abs_tol=1e-9)
    ):
        raise ValueError("DiT model-forward CUDA total mismatch")
    extract_component_latency(payload)
    return float(payload["pipeline_generate_wall_seconds"])


def validate_timing(manifest: dict[str, Any], expected_method: str) -> float:
    timing_path = validate_artifact(manifest["timing"], "timing trace")
    external = load_json(timing_path)
    if external != manifest["timing"]["payload"]:
        raise ValueError("embedded and external timing payloads differ")
    return validate_timing_payload(external, expected_method)


def validate_teacache_trace(manifest: dict[str, Any]) -> None:
    trace_path = validate_artifact(manifest["trace"], "TeaCache trace")
    trace = load_json(trace_path)
    coefficients = manifest["coefficients"]
    coefficient_path = validate_artifact(coefficients, "coefficient file")
    coefficient_payload = load_json(coefficient_path)
    if coefficient_payload.get("schema") != "teacache4wan22_coefficients_v1":
        raise ValueError("invalid coefficient schema")
    if trace.get("schema") != "teacache4wan22_trace_v1":
        raise ValueError("invalid TeaCache trace schema")
    if trace.get("coefficients_sha256") != sha256(coefficient_path):
        raise ValueError("TeaCache trace coefficient SHA256 mismatch")
    if trace.get("coefficient_protocol") != coefficient_payload.get("protocol"):
        raise ValueError("TeaCache trace coefficient protocol mismatch")
    if trace.get("threshold") != manifest["threshold"]:
        raise ValueError("TeaCache trace threshold mismatch")
    decisions = trace.get("decisions", [])
    if len(decisions) != 50:
        raise ValueError(f"expected 50 TeaCache decisions, found {len(decisions)}")
    if [row.get("step_index") for row in decisions] != list(range(50)):
        raise ValueError("TeaCache decision indices are incomplete")
    expected_stages = ["high"] * 32 + ["low"] * 18
    if [row.get("stage") for row in decisions] != expected_stages:
        raise ValueError("TeaCache high/low stage path is not 32/18")
    for row in decisions:
        branches = row.get("branches", {})
        if set(branches) != {"cond", "uncond"}:
            raise ValueError(f"missing CFG branch at step {row['step_index']}")
        if any(action != row.get("action") for action in branches.values()):
            raise ValueError(f"CFG branch action mismatch at step {row['step_index']}")
    forced = {
        0: "global_first",
        32: "stage_first",
        49: "global_final",
    }
    for step, reason in forced.items():
        row = decisions[step]
        if row.get("action") != "recompute" or row.get("forced_reason") != reason:
            raise ValueError(f"invalid forced recompute boundary at step {step}")
    timing = manifest["timing"]["payload"]
    block_counts = timing["transformer_block_count_by_stage"]
    for call in timing["calls"]:
        decision = decisions[call["step_index"]]
        expected_blocks = (
            0
            if decision["action"] == "reuse"
            else block_counts[call["model_stage"]]
        )
        if call["blocks_executed"] != expected_blocks:
            raise ValueError(
                "TeaCache decision trace and measured block execution disagree at "
                f"call {call['call_index']}"
            )


def validate_manifest(path: Path, expected_method: str) -> tuple[dict[str, Any], float]:
    manifest = load_json(path)
    if manifest.get("schema") != "teacache4wan22_run_manifest_v1":
        raise ValueError(f"invalid run manifest schema: {path}")
    threshold = manifest.get("threshold")
    if expected_method == "none" and threshold != 0:
        raise ValueError(f"baseline manifest has nonzero threshold: {path}")
    if expected_method == "teacache" and not isinstance(threshold, (int, float)):
        raise ValueError(f"TeaCache manifest has invalid threshold: {path}")
    if expected_method == "teacache" and (not math.isfinite(threshold) or threshold <= 0):
        raise ValueError(f"TeaCache manifest threshold must be positive: {path}")
    protocol_path = validate_artifact(manifest["protocol"], "protocol lock")
    if load_json(protocol_path) != manifest["protocol"]["payload"]:
        raise ValueError("embedded and external protocol locks differ")
    prepared_path = validate_artifact(
        manifest["prepared_source_manifest"], "prepared-source manifest"
    )
    if load_json(prepared_path) != manifest["prepared_source_manifest"]["payload"]:
        raise ValueError("embedded and external prepared-source manifests differ")
    validate_artifact(manifest["video"], "video")
    if Path(manifest["video"]["path"]).stat().st_size == 0:
        raise ValueError("generated video is empty")
    validate_artifact(manifest["log"], "run log")
    compute_seconds = validate_timing(manifest, expected_method)
    if expected_method == "teacache":
        validate_teacache_trace(manifest)
    elif manifest.get("trace") is not None or manifest.get("coefficients") is not None:
        raise ValueError("baseline manifest unexpectedly contains TeaCache artifacts")
    return manifest, compute_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-manifest", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--teacache-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if len(args.baseline_manifest) != len(args.teacache_manifest):
        raise ValueError("baseline and TeaCache manifest counts must match")

    pairs = []
    baseline_sum = 0.0
    teacache_sum = 0.0
    thresholds = set()
    for baseline_path, teacache_path in zip(
        args.baseline_manifest, args.teacache_manifest
    ):
        baseline, baseline_seconds = validate_manifest(baseline_path, "none")
        teacache, teacache_seconds = validate_manifest(teacache_path, "teacache")
        if baseline["prompt"] != teacache["prompt"]:
            raise ValueError("matched manifests use different prompts")
        if baseline["protocol"] != teacache["protocol"]:
            raise ValueError("matched manifests use different protocols")
        if baseline["checkpoint"] != teacache["checkpoint"]:
            raise ValueError("matched manifests use different checkpoints")
        if (
            baseline["prepared_source_manifest"]["sha256"]
            != teacache["prepared_source_manifest"]["sha256"]
        ):
            raise ValueError("matched manifests use different prepared source trees")
        thresholds.add(float(teacache["threshold"]))
        baseline_sum += baseline_seconds
        teacache_sum += teacache_seconds
        pairs.append(
            {
                "prompt": baseline["prompt"],
                "baseline_manifest": str(baseline_path.resolve()),
                "teacache_manifest": str(teacache_path.resolve()),
                "baseline_pipeline_generate_wall_seconds": baseline_seconds,
                "teacache_pipeline_generate_wall_seconds": teacache_seconds,
                "speedup": baseline_seconds / teacache_seconds,
            }
        )

    if len(thresholds) != 1:
        raise ValueError("all TeaCache runs in one aggregate must use one threshold")
    payload = {
        "schema_version": 1,
        "timing_boundary": (
            "CUDA-synchronized WanT2V.generate wall time: text encoding + denoising/"
            "cache/CFG/scheduler + model offload + VAE decode; excludes pipeline "
            "construction, MP4 export, file I/O, and evaluation"
        ),
        "threshold": next(iter(thresholds)),
        "pair_count": len(pairs),
        "baseline_pipeline_generate_wall_seconds_sum": baseline_sum,
        "teacache_pipeline_generate_wall_seconds_sum": teacache_sum,
        "inference_only_speedup_ratio_of_sums": baseline_sum / teacache_sum,
        "pairs": pairs,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
