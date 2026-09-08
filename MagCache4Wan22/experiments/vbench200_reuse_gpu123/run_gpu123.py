#!/usr/bin/env python3
"""Run the three fixed MagCache VBench200 candidates with one reused baseline."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parents[1]
REPOSITORY = PROJECT.parent
WORKSPACE = REPOSITORY.parents[1]
sys.path.insert(0, str(PROJECT / "runtime"))
sys.path.insert(0, str(PROJECT / "experiments/performance_t2v_a14b"))

from aggregate_performance import summarize_manifest, validate_profile
from batch import atomic_json, job_directory, job_plan, launch_workers, validate_completed
from common import (THREAD_VARS, artifact, check_environment, external_output, index_result,
                    read_json, sha256, write_json)
from suite import build_plan, ensure_profile, evaluate, link_video


DEFAULT_SOURCE = PROJECT / "build/Wan2.2-42bf4cf"
DEFAULT_CHECKPOINT = WORKSPACE / "models/Wan2.2-T2V-A14B"
DEFAULT_PROMPTS = REPOSITORY / "Vbench200/prompts.jsonl"
DEFAULT_PROFILE = Path(
    "/all/yiran07-disk3/huteng_data/exp/"
    "magcache4wan22_targeted_e_scan_gpu123_20260905_165530/"
    "shared_profile/calflops.json"
)
DEFAULT_BASELINE_AUDIT = Path(
    "/all/yiran07-disk3/huteng_data/exp/"
    "magcache4wan22_baseline_reuse_audit_20260906/"
    "baseline_reuse_manifest.json"
)
TARGETS = (
    dict(label="1p8x", target_speedup=1.8, gpu="1", threshold=.072, K=2, retention_ratio=.2),
    dict(label="2p4x", target_speedup=2.4, gpu="2", threshold=.199, K=4, retention_ratio=.2),
    dict(label="3p0x", target_speedup=3.0, gpu="3", threshold=.198, K=5, retention_ratio=.1),
)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_baseline(audit_path: Path, prompts: list[dict], profile: dict) -> tuple[dict, dict[str, dict]]:
    """Validate the audited bytes and corrected reporting rows used by every target."""
    audit_path = audit_path.expanduser().resolve(strict=True)
    audit = read_json(audit_path)
    if audit.get("schema") != "wan22_shared_baseline_reuse_index_v1" or audit.get("status") != "validated":
        raise ValueError("baseline audit is not a validated reuse index")
    if audit.get("video_count") != len(prompts) or len(audit.get("rows", [])) != len(prompts):
        raise ValueError("baseline audit does not contain exactly the requested prompts")
    expected_protocol = {
        "task": "t2v-A14B", "size_wh": [832, 480], "frame_num": 45, "fps": 16,
        "sampling_steps": 50, "sample_solver": "dpm++", "shift": 12.0,
        "guide_scale_low_high": [3.0, 4.0], "boundary": .875, "seed": 42,
        "param_dtype": "torch.bfloat16", "offload_model": True, "t5_cpu": False,
    }
    if audit.get("protocol") != expected_protocol:
        raise ValueError("reused baseline protocol mismatch")
    audit_rows = {row["sample_id"]: row for row in audit["rows"]}
    if len(audit_rows) != len(prompts):
        raise ValueError("duplicate baseline sample IDs")
    baseline_dir = Path(audit["baseline_directory"]).resolve(strict=True)
    reported_path = (baseline_dir / "reported_timings.jsonl").resolve(strict=True)
    reported_rows = {row["sample_id"]: row for row in jsonl(reported_path)}
    if len(reported_rows) != len(prompts):
        raise ValueError("corrected baseline timing table is incomplete or duplicated")

    # Validate every reusable video and raw timing byte against the committed audit.
    for prompt in prompts:
        sample_id = prompt["sample_id"]
        row = audit_rows.get(sample_id)
        timing = reported_rows.get(sample_id)
        if row is None or timing is None or row["prompt_en"] != prompt["prompt_en"] or row["seed"] != 42:
            raise ValueError(f"baseline prompt mismatch: {sample_id}")
        for name in ("video", "timing"):
            record = row[name]
            path = Path(record["path"]).resolve(strict=True)
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise ValueError(f"reused baseline {name} changed: {sample_id}")
        if Path(timing["timing_path"]).resolve() != Path(row["timing"]["path"]).resolve():
            raise ValueError(f"baseline timing provenance mismatch: {sample_id}")
        if not math.isclose(float(timing["pipeline_generate_wall_seconds"]),
                            float(row["generate_seconds"]), rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"baseline corrected timing mismatch: {sample_id}")
        if timing["model_forward_calls"] != 100 or timing["full_compute_forward_calls"] != 100 or timing["reuse_forward_calls"] != 0:
            raise ValueError(f"baseline call count mismatch: {sample_id}")

    # The reused component FLOPs must agree with the MagCache Calflops profile.
    first = next(iter(reported_rows.values()))
    for field, component in (("estimated_t5_tflops_per_video", "t5"),
                             ("estimated_vae_decode_tflops_per_video", "vae_decode")):
        expected = profile["component_profiles"][component]["estimated_tflops_per_video"]
        if not math.isclose(float(first[field]), float(expected), rel_tol=1e-12):
            raise ValueError(f"baseline/profile {component} FLOPs mismatch")
    provenance = {
        "audit": artifact(audit_path),
        "baseline_directory": str(baseline_dir),
        "reported_timings": artifact(reported_path),
        "timing_policy": audit.get("timing_policy"),
        "timing_correction": audit.get("timing_correction"),
        "video_count": len(prompts),
        "generation": "reused; zero baseline videos generated by this suite",
    }
    return provenance, reported_rows


def baseline_metric_row(row: dict) -> dict:
    return {
        "sample_id": row["sample_id"],
        "generate_seconds": float(row["pipeline_generate_wall_seconds"]),
        "t5_cuda_seconds": float(row["t5_cuda_seconds"]),
        "t5_host_span_seconds": float(row["t5_host_span_seconds"]),
        "vae_decode_cuda_seconds": float(row["vae_decode_cuda_seconds"]),
        "vae_decode_host_span_seconds": float(row["vae_decode_host_span_seconds"]),
        "dit_cuda_seconds": float(row["dit_cuda_seconds"]),
        "estimated_t5_tflops_per_video": float(row["estimated_t5_tflops_per_video"]),
        "estimated_vae_decode_tflops_per_video": float(row["estimated_vae_decode_tflops_per_video"]),
        "estimated_dit_tflops": float(row["estimated_dit_tflops"]),
        "reuse_calls": int(row["reuse_forward_calls"]),
        "full_calls": int(row["full_compute_forward_calls"]),
    }


def aggregate_reused(root: Path, plan: dict, profile: dict, reported: dict[str, dict]) -> dict:
    condition = plan["conditions"][0]
    rows = []
    method = None
    for prompt in plan["prompts"]:
        sample_id = prompt["sample_id"]
        path = job_directory(root, prompt, condition) / "manifest.json"
        manifest = validate_completed(path, job_plan(plan, prompt, condition), profile,
                                      plan["gpu_ids"][0], plan["warmup_videos"])
        actions = [row["action"] for row in read_json(manifest["trace"]["path"])["calls"]]
        if actions != condition["schedule"]:
            raise ValueError(f"candidate schedule mismatch: {sample_id}")
        _, candidate = summarize_manifest(path, "magcache", profile)
        if manifest["sample_id"] != sample_id or manifest["prompt"] != prompt["prompt_en"]:
            raise ValueError(f"candidate prompt mismatch: {sample_id}")
        if method is not None and method != manifest["method"]:
            raise ValueError("mixed candidate settings")
        method = manifest["method"]
        baseline = baseline_metric_row(reported[sample_id])
        rows.append({"baseline": baseline, "magcache": candidate})
        source_video = Path(plan["baseline_reuse"]["baseline_directory"]) / "videos" / f"{sample_id}.mp4"
        link_video(root / "videos/baseline" / f"{sample_id}.mp4", source_video)
        link_video(root / "videos" / condition["id"] / f"{sample_id}.mp4", manifest["video"]["path"])
    sums = {
        mode: {key: sum(row[mode][key] for row in rows) for key in rows[0][mode] if key != "sample_id"}
        for mode in ("baseline", "magcache")
    }
    result = {
        "schema": "magcache4wan22_performance_v1",
        "pair_count": len(rows),
        "method": method,
        "speedup": sums["baseline"]["generate_seconds"] / sums["magcache"]["generate_seconds"],
        "dit_flops_speedup": sums["baseline"]["estimated_dit_tflops"] / sums["magcache"]["estimated_dit_tflops"],
        "sums": sums,
        "per_video": rows,
        "profile": read_json(root / "profile_artifact.json"),
        "equivalent_parameters": condition["equivalent_parameters"],
        "baseline_reuse": plan["baseline_reuse"],
        "scope": ("Complete generate latency; DiT TFLOPs headline and separate T5/VAE counts. "
                  "Baseline video bytes are reused and baseline reporting time applies the committed GPU0 correction policy."),
    }
    atomic_json(root / "performance" / f"{condition['id']}.json", result)
    atomic_json(root / "performance.json", {condition["id"]: result})
    return result


def make_plan(args: argparse.Namespace, target: dict) -> tuple[dict, dict[str, dict]]:
    namespace = argparse.Namespace(
        source=args.source, checkpoint=args.checkpoint, prompts=args.prompts,
        vbench_mode="standard", gpu_ids=[target["gpu"]], worker_launch_wave_size=1,
        worker_ready_timeout=args.worker_ready_timeout, warmup_videos=1,
        calflops_profile=args.calflops_profile, threshold=target["threshold"],
        magcache_k=target["K"], retention_ratio=target["retention_ratio"],
        target_speedups=None, target_tolerance=None, target_presets=None,
        thresholds=None, ks=None, retention_ratios=None, no_deduplicate=False,
    )
    plan = build_plan(namespace, False)
    plan["conditions"] = plan["conditions"][1:]
    plan.update(
        schema="magcache4wan22_reused_baseline_batch_plan_v1",
        kind="vbench200_candidate_with_reused_baseline",
        requested_target_speedup=target["target_speedup"],
        generated_videos=len(plan["prompts"]),
        runner="one persistent pipeline on the assigned GPU; candidate only; audited baseline reused",
    )
    profile = read_json(args.calflops_profile)
    validate_profile(profile)
    provenance, reported = validate_baseline(args.baseline_audit, plan["prompts"], profile)
    plan["baseline_reuse"] = provenance
    return plan, reported


def run_target(args: argparse.Namespace, target: dict) -> None:
    root = external_output(args.output_dir / f"target_{target['label']}_gpu{target['gpu']}")
    plan, reported = make_plan(args, target)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "plan.json").exists():
            if not args.resume or read_json(root / "plan.json") != plan:
                raise ValueError(f"existing target plan mismatch: {root}")
        else:
            write_json(root / "plan.json", plan)
        (root / "README.md").write_text(
            "# MagCache VBench200 target\n\n"
            "This directory contains 200 candidate videos from one persistent GPU worker. "
            "videos/baseline contains symlinks to the audited shared baseline; performance.json "
            "uses its corrected reported timings. Quality evaluation includes PSNR, SSIM, LPIPS, and candidate VBench.\n"
        )
        try:
            profile = ensure_profile(root, plan)
            for prompt in plan["prompts"]:
                sample_id = prompt["sample_id"]
                source_video = Path(plan["baseline_reuse"]["baseline_directory"]) / "videos" / f"{sample_id}.mp4"
                link_video(root / "videos/baseline" / f"{sample_id}.mp4", source_video)
            if not (root / "baseline_reuse.json").exists():
                write_json(root / "baseline_reuse.json", plan["baseline_reuse"])
            if not (root / "GENERATION_COMPLETE.json").exists():
                atomic_json(root / "status.json", {"status": "running", "phase": "generation"})
                launch_workers(root, plan, args.resume, 1, args.worker_ready_timeout)
                result = aggregate_reused(root, plan, profile, reported)
                atomic_json(root / "GENERATION_COMPLETE.json", {
                    "status": "complete", "performance": artifact(root / "performance.json"),
                    "baseline_generated_videos": 0, "candidate_generated_videos": len(plan["prompts"]),
                })
            else:
                result = read_json(root / "performance.json")[plan["conditions"][0]["id"]]
            atomic_json(root / "status.json", {"status": "running", "phase": "quality_evaluation"})
            quality = evaluate(root, plan)
            report = {
                "schema": "magcache4wan22_reused_baseline_batch_report_v1",
                "status": "complete", "plan": plan,
                "performance": {plan["conditions"][0]["id"]: result}, **quality,
            }
            atomic_json(root / "report.json", report)
            atomic_json(root / "COMPLETE.json", {"status": "complete", "report": artifact(root / "report.json")})
            atomic_json(root / "status.json", {"status": "complete", "phase": "complete"})
        except BaseException as exc:
            atomic_json(root / "status.json", {"status": "failed", "error": repr(exc)})
            raise


def gpu_is_idle(gpu: str) -> bool:
    output = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
    ], text=True)
    return not output.strip()


def parent(args: argparse.Namespace) -> None:
    plans = {target["label"]: make_plan(args, target)[0] for target in TARGETS}
    if args.dry_run:
        print(json.dumps({"output_dir": str(args.output_dir), "plans": plans}, indent=2, ensure_ascii=False))
        return
    if any(not gpu_is_idle(target["gpu"]) for target in TARGETS):
        raise RuntimeError("GPU1/2/3 must be idle at launch")
    root = external_output(args.output_dir)
    if root.exists() and not args.resume:
        raise FileExistsError(f"use --resume for existing output: {root}")
    root.mkdir(parents=True, exist_ok=True)
    index_result(root)
    (root / "README.md").write_text(
        "# MagCache4Wan22 VBench200 three-target run\n\n"
        "target_1p8x_gpu1, target_2p4x_gpu2, and target_3p0x_gpu3 hold the fixed target suites. "
        "All candidates use one persistent pipeline and the same audited 200-video baseline without regeneration.\n"
    )
    launch = {
        "schema": "magcache4wan22_vbench200_gpu123_launch_v1",
        "status": "running", "output_dir": str(root), "targets": list(TARGETS),
        "plans": plans, "baseline_generated_videos": 0, "candidate_generated_videos": 600,
    }
    if (root / "launch.json").exists():
        if read_json(root / "launch.json") != launch:
            raise ValueError("resume launch mismatch")
    else:
        write_json(root / "launch.json", launch)
    atomic_json(root / "status.json", {"status": "running", "phase": "generation_and_evaluation"})
    processes = []
    try:
        for target in TARGETS:
            log_path = root / f"controller_{target['label']}_gpu{target['gpu']}.log"
            log = log_path.open("a")
            command = [sys.executable, __file__, "--output-dir", str(root), "--target-only", target["label"],
                       "--source", str(args.source), "--checkpoint", str(args.checkpoint),
                       "--prompts", str(args.prompts), "--calflops-profile", str(args.calflops_profile),
                       "--baseline-audit", str(args.baseline_audit),
                       "--worker-ready-timeout", str(args.worker_ready_timeout)]
            if args.resume:
                command.append("--resume")
            env = dict(os.environ, PYTHONUNBUFFERED="1", **{name: "1" for name in THREAD_VARS})
            processes.append((target, subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log, log_path))
        while any(process.poll() is None for _, process, _, _ in processes):
            failures = [
                {"target": target["label"], "returncode": process.returncode, "log": str(log_path)}
                for target, process, _, log_path in processes
                if process.poll() not in (None, 0)
            ]
            if failures:
                for _, process, _, _ in processes:
                    if process.poll() is None:
                        process.terminate()
                raise RuntimeError(f"target controllers failed: {failures}")
            import time
            time.sleep(5)
        failures = [
            {"target": target["label"], "returncode": process.returncode, "log": str(log_path)}
            for target, process, _, log_path in processes if process.returncode
        ]
        for _, _, log, _ in processes:
            log.close()
        if failures:
            raise RuntimeError(f"target controllers failed: {failures}")
        atomic_json(root / "COMPLETE.json", {
            "status": "complete",
            "reports": {target["label"]: artifact(root / f"target_{target['label']}_gpu{target['gpu']}" / "report.json")
                        for target in TARGETS},
        })
        atomic_json(root / "status.json", {"status": "complete", "phase": "complete"})
    except BaseException as exc:
        atomic_json(root / "status.json", {"status": "failed", "error": repr(exc)})
        for _, process, log, _ in processes:
            if process.poll() is None:
                process.terminate()
            if not log.closed:
                log.close()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--calflops-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--baseline-audit", type=Path, default=DEFAULT_BASELINE_AUDIT)
    parser.add_argument("--worker-ready-timeout", type=float, default=1800)
    parser.add_argument("--target-only", choices=[target["label"] for target in TARGETS])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("source", "checkpoint", "prompts", "calflops_profile", "baseline_audit"):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.worker_ready_timeout <= 0:
        raise ValueError("worker timeout must be positive")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    check_environment()
    if parsed.target_only:
        run_target(parsed, next(target for target in TARGETS if target["label"] == parsed.target_only))
    else:
        parent(parsed)
