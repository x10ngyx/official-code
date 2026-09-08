#!/usr/bin/env python3
"""Validate measured calls and aggregate matching baseline/DiCache components."""
from __future__ import annotations
import argparse
import math
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "runtime"))
sys.path.insert(0, str(PROJECT.parent / "ComponentMetrics"))
from common import artifact, check_environment, external_output, index_result, read_json, sha256, write_json
from reporting import extract_component_latency, extract_component_tflops


def load_artifact(record):
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError(f"artifact SHA256 mismatch: {path}")
    return path


def summarize_manifest(path, mode, profile):
    validate_profile(profile)
    manifest = read_json(path)
    if manifest.get("schema") != "dicache4wan22_run_v1" or manifest.get("status") != "complete" or manifest["mode"] != mode:
        raise ValueError("invalid completed run manifest")
    protocol = read_json(PROJECT / "configs/wan22_t2v_a14b_50step_dpmpp.json")
    if manifest["protocol"] != protocol:
        raise ValueError("run protocol mismatch")
    load_artifact(manifest["video"])
    run = read_json(load_artifact(manifest["run"]))
    for key in ("schema", "mode", "prompt", "sample_id", "checkpoint", "protocol", "method", "source"):
        if run[key] != manifest[key]:
            raise ValueError(f"run/manifest metadata mismatch: {key}")
    timing = read_json(load_artifact(manifest["timing"]))
    if timing.get("status") != "success" or timing.get("implementation") != mode or timing.get("error") is not None:
        raise ValueError("timing is not a successful matching method run")
    calls = timing["calls"]
    if len(calls) != 100 or timing["model_forward_call_count"] != 100:
        raise ValueError("expected 100 CFG forwards")
    latency = float(timing["pipeline_generate_wall_seconds"])
    if not math.isfinite(latency) or latency <= 0:
        raise ValueError("invalid generate latency")
    components = extract_component_latency(timing)
    trace = None
    if mode == "dicache":
        trace = read_json(load_artifact(manifest["trace"]))
        if trace.get("schema") != "dicache4wan22_trace_v1" or trace["method"] != manifest["method"] or len(trace["calls"]) != 100:
            raise ValueError("invalid DiCache decision trace")
    dit_flops = cuda = 0.0
    reuse = 0
    for i, call in enumerate(calls):
        stage, branch = ("high" if i < 64 else "low"), ("cond" if i % 2 == 0 else "uncond")
        if (call["call_index"], call["step_index"], call["model_stage"], call["cfg_branch"]) != (i, i // 2, stage, branch):
            raise ValueError("call identity or 32/18 stage path mismatch")
        count = timing["transformer_block_count_by_stage"][stage]
        if count != profile["stages"][stage]["model"]["transformer_blocks"]:
            raise ValueError("profile/run block count mismatch")
        executed = call["blocks_executed"]
        if executed not in (1, count) or (mode == "baseline" and executed != count):
            raise ValueError("unexpected partial/full block path")
        if call["reuse"] != (executed == 1) or call["full_compute"] != (executed == count):
            raise ValueError("incorrect full/reuse flags")
        if trace:
            decision = trace["calls"][i]
            for field in ("call_index", "step_index", "model_stage", "cfg_branch", "blocks_executed"):
                if decision[field] != call[field]:
                    raise ValueError("trace and measured execution disagree")
            if decision["action"] != ("reuse" if executed == 1 else "recompute"):
                raise ValueError("trace action mismatch")
        branch_profile = profile["stages"][stage]["branches"][branch]
        flops = float(branch_profile["estimated_full_flops" if executed == count else "estimated_probe_flops"])
        if not math.isfinite(flops) or flops <= 0:
            raise ValueError("invalid profile FLOPs")
        dit_flops += flops
        seconds = call["cuda_seconds"]
        if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid call CUDA seconds")
        cuda += seconds
        reuse += int(executed == 1)
    if reuse != timing["reuse_forward_calls"] or 100 - reuse != timing["full_compute_forward_calls"]:
        raise ValueError("timing full/reuse totals mismatch")
    if not math.isclose(cuda, components["dit_cuda_seconds"], rel_tol=1e-10):
        raise ValueError("CUDA total mismatch")
    if manifest["source"] != profile["source"]["prepared_manifest"]:
        raise ValueError("profile/run source lock mismatch")
    if Path(manifest["checkpoint"]).resolve() != Path(profile["source"]["checkpoint_dir"]).resolve():
        raise ValueError("profile/run checkpoint mismatch")
    return manifest, dict(sample_id=manifest["sample_id"], generate_seconds=latency,
                          **components, **extract_component_tflops(profile),
                          estimated_dit_tflops=dit_flops / 1e12, reuse_calls=reuse,
                          full_calls=100 - reuse)


def validate_profile(profile):
    if profile.get("schema") != "dicache4wan22_calflops_profile_v2":
        raise ValueError("profile schema mismatch")
    expected_inputs = dict(video_shape_fhw=[45, 480, 832], output_fps=16,
                           sampling_steps=50, solver="dpm++", shift=12.0,
                           guide_scale_low_high=[3.0, 4.0], boundary=.875, seed=42,
                           parameter_dtype="bfloat16", stage_steps={"high": 32, "low": 18})
    for key, expected in expected_inputs.items():
        if profile.get("input", {}).get(key) != expected:
            raise ValueError(f"profile input mismatch: {key}")
    for stage in ("high", "low"):
        if profile["stages"][stage]["model"]["transformer_blocks"] != 40:
            raise ValueError("expected 40 blocks per expert")
        for branch in ("cond", "uncond"):
            p = profile["stages"][stage]["branches"][branch]
            values = [p[k] for k in ("estimated_always_on_flops", "estimated_probe_flops", "estimated_full_flops")]
            if p.get("probe_depth") != 1 or not all(math.isfinite(v) and v > 0 for v in values) or not values[0] < values[1] < values[2]:
                raise ValueError("invalid full/probe/always-on FLOPs profile")
    for key, expected_calls in (("t5", 2), ("vae_decode", 1)):
        component = profile["component_profiles"][key]
        tflops, flops = component["estimated_tflops_per_video"], component["estimated_flops_per_video"]
        if component["calls_per_video"] != expected_calls or not math.isfinite(tflops) or not math.isfinite(flops) or not math.isclose(tflops, flops / 1e12):
            raise ValueError(f"invalid {key} component profile")


def aggregate(baseline_paths, candidate_paths, profile):
    validate_profile(profile)
    if not baseline_paths or len(baseline_paths) != len(candidate_paths):
        raise ValueError("expected equally sized nonempty matched run lists")
    rows = []
    seen = set()
    method = None
    for baseline, candidate in zip(baseline_paths, candidate_paths):
        b, br = summarize_manifest(baseline, "baseline", profile)
        c, cr = summarize_manifest(candidate, "dicache", profile)
        for key in ("prompt", "sample_id", "protocol", "checkpoint", "source"):
            if b[key] != c[key]:
                raise ValueError(f"pair mismatch: {key}")
        if b["sample_id"] in seen:
            raise ValueError("duplicate sample ID")
        seen.add(b["sample_id"])
        if method is not None and c["method"] != method:
            raise ValueError("mixed DiCache settings")
        method = c["method"]
        rows.append(dict(baseline=br, dicache=cr))
    sums = {mode: {key: sum(row[mode][key] for row in rows)
                   for key in rows[0][mode] if key != "sample_id"}
            for mode in ("baseline", "dicache")}
    return dict(schema="dicache4wan22_performance_v1", pair_count=len(rows), method=method,
                speedup=sums["baseline"]["generate_seconds"] / sums["dicache"]["generate_seconds"],
                dit_flops_speedup=sums["baseline"]["estimated_dit_tflops"] / sums["dicache"]["estimated_dit_tflops"],
                sums=sums, per_video=rows,
                scope="Complete generate latency; DiT TFLOPs headline and separate T5/VAE counts. Probe block included in every reuse call; DiCache gate/DCTA, residual addition, CFG/scheduler/export excluded from FLOPs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, action="append", required=True)
    parser.add_argument("--dicache-manifest", type=Path, action="append", required=True)
    parser.add_argument("--calflops-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_environment()
    payload = aggregate(args.baseline_manifest, args.dicache_manifest, read_json(args.calflops_profile))
    payload["profile"] = artifact(args.calflops_profile)
    destination = external_output(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    index_result(destination.parent)
    if not (destination.parent / "README.md").exists():
        (destination.parent / "README.md").write_text("# DiCache performance aggregate\n\nValidated paired time and component FLOPs with per-video records.\n")
    write_json(destination, payload)

if __name__ == "__main__":
    main()
