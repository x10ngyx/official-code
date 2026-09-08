#!/usr/bin/env python3
"""Resume the three VBench200 targets with balanced cross-target GPU work."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT / "runtime"))
sys.path.insert(0, str(PROJECT / "experiments/performance_t2v_a14b"))
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_performance import summarize_manifest
from batch import atomic_json, job_directory, job_plan
from common import (THREAD_VARS, artifact, check_environment, external_output,
                    read_json, sha256, validate_source, write_json)
from generation import create_pipeline, generate_native, generate_run
from official import load_official, method_args
from run_gpu123 import baseline_metric_row, validate_baseline
from suite import ensure_profile, evaluate, link_video


GPUS = ("1", "2", "3")
TARGET_DIRS = ("target_1p8x_gpu1", "target_2p4x_gpu2", "target_3p0x_gpu3")


def target_context(root: Path) -> dict[str, dict]:
    contexts = {}
    for name in TARGET_DIRS:
        target = root / name
        plan = read_json(target / "plan.json")
        if len(plan["conditions"]) != 1 or plan["conditions"][0]["mode"] != "magcache":
            raise ValueError(f"target plan is not candidate-only: {target}")
        profile_record = read_json(target / "profile_artifact.json")
        if sha256(profile_record["path"]) != profile_record["sha256"]:
            raise ValueError(f"profile changed: {target}")
        contexts[name] = {
            "root": target, "plan": plan, "condition": plan["conditions"][0],
            "profile_record": profile_record, "profile": read_json(profile_record["path"]),
            "prompts": {row["sample_id"]: row for row in plan["prompts"]},
        }
    sources = {json.dumps(value["plan"]["source"], sort_keys=True) for value in contexts.values()}
    checkpoints = {str(Path(value["plan"]["checkpoint"]).resolve()) for value in contexts.values()}
    profiles = {json.dumps(value["profile_record"], sort_keys=True) for value in contexts.values()}
    if len(sources) != 1 or len(checkpoints) != 1 or len(profiles) != 1:
        raise ValueError("targets do not share one source, checkpoint, and profile")
    first = next(iter(contexts.values()))
    if validate_source(first["plan"]["source"]["source"]) != first["plan"]["source"]:
        raise ValueError("source lock changed")
    return contexts


def validate_any_gpu(path: Path, context: dict) -> tuple[dict, dict]:
    plan, condition, profile = context["plan"], context["condition"], context["profile"]
    expected = job_plan(plan, context["prompts"][path.parent.name], condition)
    manifest, row = summarize_manifest(path, "magcache", profile)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"run input mismatch: {path}")
    timing = read_json(manifest["timing"]["path"])
    lifecycle = timing.get("pipeline_lifecycle", {})
    if (lifecycle.get("persistent_pipeline") is not True
            or str(lifecycle.get("physical_gpu")) not in GPUS
            or lifecycle.get("pipeline_init_accounted_in_this_sample") is not False
            or lifecycle.get("profiler_freshly_installed_for_sample") is not True
            or lifecycle.get("warmup_videos") != 1
            or timing.get("pipeline_init_wall_seconds") != 0):
        raise ValueError(f"invalid persistent lifecycle: {path}")
    actions = [value["action"] for value in read_json(manifest["trace"]["path"])["calls"]]
    if actions != condition["schedule"]:
        raise ValueError(f"official action schedule mismatch: {path}")
    return manifest, row


def completed_and_means(contexts: dict[str, dict]) -> tuple[dict[str, set[str]], dict[str, float]]:
    completed, means = {}, {}
    for name, context in contexts.items():
        paths = sorted((context["root"] / "runs" / context["condition"]["id"]).glob("*/manifest.json"))
        values, ids = [], set()
        for path in paths:
            manifest, row = validate_any_gpu(path, context)
            if manifest["sample_id"] in ids:
                raise ValueError(f"duplicate sample: {manifest['sample_id']}")
            ids.add(manifest["sample_id"])
            values.append(float(row["generate_seconds"]))
        if not values:
            raise ValueError(f"cannot estimate target cost without completed samples: {name}")
        completed[name], means[name] = ids, sum(values) / len(values)
    return completed, means


def build_assignment(root: Path, contexts: dict[str, dict]) -> dict:
    completed, means = completed_and_means(contexts)
    jobs = []
    for name, context in contexts.items():
        for index, row in enumerate(context["plan"]["prompts"]):
            if row["sample_id"] not in completed[name]:
                jobs.append({
                    "target": name, "sample_id": row["sample_id"], "prompt_index": index,
                    "estimated_seconds": means[name],
                })
    # Longest-processing-time greedy assignment gives a deterministic near-equal
    # predicted makespan while distributing every target across all three GPUs.
    jobs.sort(key=lambda row: (-row["estimated_seconds"], row["target"], row["prompt_index"]))
    loads = {gpu: 0.0 for gpu in GPUS}
    assignments = {gpu: [] for gpu in GPUS}
    for job in jobs:
        gpu = min(GPUS, key=lambda value: (loads[value], int(value)))
        assignments[gpu].append(job)
        loads[gpu] += job["estimated_seconds"]
    return {
        "schema": "magcache4wan22_balanced_resume_assignment_v1",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "reason": "user requested balanced GPU load",
        "algorithm": "deterministic longest-processing-time greedy over remaining videos",
        "gpus": list(GPUS),
        "completed_before_rebalance": {name: len(rows) for name, rows in completed.items()},
        "mean_generate_seconds_at_rebalance": means,
        "remaining_video_count": len(jobs),
        "estimated_load_seconds": loads,
        "assignments": assignments,
        "warmup_videos_per_new_worker": 1,
        "baseline_generation_count": 0,
    }


def archive_incomplete(directory: Path, target: Path, condition: dict, sample_id: str) -> None:
    if not directory.exists() or (directory / "manifest.json").exists():
        return
    archive = target / "incomplete_attempts" / "balanced_resume" / condition["id"] / sample_id
    archive.mkdir(parents=True, exist_ok=True)
    directory.rename(archive / f"attempt_{time.time_ns()}")


def worker(root: Path, gpu: str, resume: bool) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise ValueError("balanced worker must expose exactly its assigned physical GPU")
    assignment = read_json(root / "balanced_assignment.json")
    contexts = target_context(root)
    jobs = assignment["assignments"][gpu]
    pending = []
    for job in jobs:
        context = contexts[job["target"]]
        row = context["prompts"][job["sample_id"]]
        directory = job_directory(context["root"], row, context["condition"])
        if (directory / "manifest.json").exists():
            if not resume:
                raise FileExistsError(directory)
            validate_any_gpu(directory / "manifest.json", context)
            continue
        archive_incomplete(directory, context["root"], context["condition"], row["sample_id"])
        pending.append((job, context, row, directory))
    state_path = root / "balanced_worker_status" / f"gpu_{gpu}.json"
    state = {
        "pid": os.getpid(), "physical_gpu": gpu, "persistent_pipeline": True,
        "pipeline_init_count": 0, "completed_this_process": 0,
        "assigned_count": len(jobs), "pending_count": len(pending),
        "estimated_assigned_seconds": assignment["estimated_load_seconds"][gpu],
        "status": "initializing",
    }
    atomic_json(state_path, state)
    if not pending:
        atomic_json(state_path, dict(state, status="complete"))
        return
    first = pending[0][1]
    sys.path.insert(0, first["plan"]["source"]["source"])
    import torch
    try:
        pipeline, init_seconds = create_pipeline(first["plan"]["checkpoint"])
        core = load_official()
        state.update(status="ready", pipeline_init_count=1,
                     pipeline_init_wall_seconds_once=init_seconds)
        atomic_json(state_path, state)
        torch.cuda.synchronize()
        started = time.perf_counter()
        video = generate_native(pipeline, pending[0][2]["prompt_en"])
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - started
        del video
        gc.collect()
        torch.cuda.empty_cache()
        for job, context, row, directory in pending:
            condition = context["condition"]
            state.update(status="running", target=job["target"], sample_id=row["sample_id"])
            atomic_json(state_path, state)
            lifecycle = {
                "runner": "persistent_balanced_batch_worker", "persistent_pipeline": True,
                "physical_gpu": gpu, "worker_pid": os.getpid(),
                "pipeline_init_wall_seconds_once": init_seconds,
                "pipeline_init_accounted_in_this_sample": False,
                "profiler_freshly_installed_for_sample": True,
                "warmup_videos": 1, "warmup_wall_seconds": warmup_seconds,
                "balanced_assignment": artifact(root / "balanced_assignment.json"),
            }
            expected = job_plan(context["plan"], row, condition)
            path = generate_run(
                pipeline, core,
                method_args(condition["threshold"], condition["K"], condition["retention_ratio"]),
                expected, directory, lifecycle=lifecycle,
            )
            validate_any_gpu(path, context)
            state["completed_this_process"] += 1
            atomic_json(state_path, state)
            with (root / "balanced_worker_status" / f"gpu_{gpu}.jsonl").open("a") as stream:
                stream.write(json.dumps({"target": job["target"], "sample_id": row["sample_id"],
                                         "manifest": artifact(path)}) + "\n")
            gc.collect()
            torch.cuda.empty_cache()
        atomic_json(state_path, dict(state, status="complete"))
    except BaseException as exc:
        atomic_json(state_path, dict(state, status="failed", error=repr(exc)))
        raise


def aggregate_target(context: dict, reported: dict[str, dict]) -> dict:
    root, plan, condition = context["root"], context["plan"], context["condition"]
    rows, method = [], None
    for prompt in plan["prompts"]:
        path = job_directory(root, prompt, condition) / "manifest.json"
        manifest, candidate = validate_any_gpu(path, context)
        if manifest["sample_id"] != prompt["sample_id"] or manifest["prompt"] != prompt["prompt_en"]:
            raise ValueError(f"candidate prompt mismatch: {path}")
        if method is not None and method != manifest["method"]:
            raise ValueError("mixed MagCache settings")
        method = manifest["method"]
        rows.append({"baseline": baseline_metric_row(reported[prompt["sample_id"]]),
                     "magcache": candidate})
        link_video(root / "videos" / condition["id"] / f"{prompt['sample_id']}.mp4",
                   manifest["video"]["path"])
    sums = {
        mode: {key: sum(row[mode][key] for row in rows) for key in rows[0][mode] if key != "sample_id"}
        for mode in ("baseline", "magcache")
    }
    value = {
        "schema": "magcache4wan22_performance_v1", "pair_count": len(rows), "method": method,
        "speedup": sums["baseline"]["generate_seconds"] / sums["magcache"]["generate_seconds"],
        "dit_flops_speedup": sums["baseline"]["estimated_dit_tflops"] / sums["magcache"]["estimated_dit_tflops"],
        "sums": sums, "per_video": rows, "profile": context["profile_record"],
        "equivalent_parameters": condition["equivalent_parameters"],
        "baseline_reuse": plan["baseline_reuse"],
        "balanced_assignment": artifact(root.parent / "balanced_assignment.json"),
        "scope": ("Complete generate latency; DiT TFLOPs headline and separate T5/VAE counts. "
                  "Candidate rows use validated persistent workers on GPU1/2/3; baseline bytes and corrected reporting times are reused."),
    }
    atomic_json(root / "performance" / f"{condition['id']}.json", value)
    atomic_json(root / "performance.json", {condition["id"]: value})
    return value


def finalize_target(root: Path, name: str, evaluation_gpu: str | None = None) -> None:
    contexts = target_context(root)
    context = contexts[name]
    target, plan = context["root"], context["plan"]
    with (target / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            profile = ensure_profile(target, plan)
            audit_path = Path(plan["baseline_reuse"]["audit"]["path"])
            provenance, reported = validate_baseline(audit_path, plan["prompts"], profile)
            if provenance != plan["baseline_reuse"]:
                raise ValueError("baseline reuse provenance changed")
            result = aggregate_target(context, reported)
            atomic_json(target / "GENERATION_COMPLETE.json", {
                "status": "complete", "performance": artifact(target / "performance.json"),
                "baseline_generated_videos": 0, "candidate_generated_videos": 200,
                "balanced_assignment": artifact(root / "balanced_assignment.json"),
            })
            quality_gpu = evaluation_gpu or str(plan["gpu_ids"][0])
            evaluation_plan = dict(plan, gpu_ids=[quality_gpu])
            atomic_json(target / "status.json", {"status": "running", "phase": "quality_evaluation",
                                                   "evaluation_gpu": quality_gpu})
            quality = evaluate(target, evaluation_plan)
            report = {
                "schema": "magcache4wan22_reused_baseline_batch_report_v1",
                "status": "complete", "plan": plan,
                "performance": {context["condition"]["id"]: result}, **quality,
                "balanced_assignment": artifact(root / "balanced_assignment.json"),
                "quality_evaluation_gpu": quality_gpu,
            }
            atomic_json(target / "report.json", report)
            atomic_json(target / "COMPLETE.json", {"status": "complete", "report": artifact(target / "report.json")})
            atomic_json(target / "status.json", {"status": "complete", "phase": "complete"})
        except BaseException as exc:
            atomic_json(target / "status.json", {"status": "failed", "error": repr(exc)})
            raise


def spawn(root: Path, extra: list[str], log_path: Path, gpu: str | None = None) -> tuple[subprocess.Popen, object]:
    log = log_path.open("a")
    command = [sys.executable, __file__, "--output-dir", str(root), *extra]
    env = dict(os.environ, PYTHONUNBUFFERED="1", **{name: "1" for name in THREAD_VARS})
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log


def wait_processes(records: list[tuple[str, subprocess.Popen, object, Path]]) -> None:
    try:
        while any(process.poll() is None for _, process, _, _ in records):
            failures = [{"name": name, "returncode": process.returncode, "log": str(path)}
                        for name, process, _, path in records if process.poll() not in (None, 0)]
            if failures:
                raise RuntimeError(f"balanced subprocess failed: {failures}")
            time.sleep(5)
        failures = [{"name": name, "returncode": process.returncode, "log": str(path)}
                    for name, process, _, path in records if process.returncode]
        if failures:
            raise RuntimeError(f"balanced subprocess failed: {failures}")
    finally:
        for _, process, _, _ in records:
            if process.poll() is None:
                process.terminate()
        for _, process, log, _ in records:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()


def coordinator(args: argparse.Namespace) -> None:
    root = external_output(args.output_dir)
    contexts = target_context(root)
    assignment_path = root / "balanced_assignment.json"
    if assignment_path.exists():
        if not args.resume:
            raise FileExistsError("balanced assignment exists; use --resume")
        assignment = read_json(assignment_path)
    else:
        assignment = build_assignment(root, contexts)
    if args.dry_run:
        print(json.dumps(assignment, indent=2, ensure_ascii=False))
        return
    if not assignment_path.exists():
        write_json(assignment_path, assignment)
    if any(subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid",
                                    "--format=csv,noheader,nounits"], text=True).strip() for gpu in GPUS):
        raise RuntimeError("GPU1/2/3 must be idle before balanced resume")
    with (root / ".balanced_runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            atomic_json(root / "status.json", {"status": "running", "phase": "balanced_generation",
                                                "assignment": artifact(assignment_path)})
            for context in contexts.values():
                atomic_json(context["root"] / "status.json", {"status": "running", "phase": "balanced_generation",
                                                               "assignment": artifact(assignment_path)})
            records = []
            for gpu in GPUS:
                path = root / f"balanced_worker_gpu{gpu}.log"
                process, log = spawn(root, ["--worker", gpu, "--resume"], path, gpu)
                records.append((f"gpu{gpu}", process, log, path))
            wait_processes(records)
            atomic_json(root / "status.json", {"status": "running", "phase": "parallel_quality_evaluation",
                                                "assignment": artifact(assignment_path)})
            records = []
            for name in TARGET_DIRS:
                gpu = contexts[name]["plan"]["gpu_ids"][0]
                path = root / f"balanced_finalize_{name}.log"
                process, log = spawn(root, ["--finalize-target", name], path, gpu)
                records.append((name, process, log, path))
            wait_processes(records)
            atomic_json(root / "COMPLETE.json", {
                "status": "complete", "balanced_assignment": artifact(assignment_path),
                "reports": {name: artifact(root / name / "report.json") for name in TARGET_DIRS},
            })
            atomic_json(root / "status.json", {"status": "complete", "phase": "complete"})
        except BaseException as exc:
            atomic_json(root / "status.json", {"status": "failed", "error": repr(exc),
                                                "assignment": artifact(assignment_path)})
            raise


def complete_root(root: Path) -> None:
    """Commit the suite marker after independently resumed target finalizers."""
    target_context(root)
    reports = {}
    for name in TARGET_DIRS:
        target = root / name
        complete = read_json(target / "COMPLETE.json")
        report = artifact(target / "report.json")
        if complete.get("status") != "complete" or complete.get("report") != report:
            raise ValueError(f"incomplete or changed target report: {name}")
        if read_json(report["path"]).get("status") != "complete":
            raise ValueError(f"target report is not complete: {name}")
        reports[name] = report
    atomic_json(root / "COMPLETE.json", {
        "status": "complete", "balanced_assignment": artifact(root / "balanced_assignment.json"),
        "reports": reports,
    })
    atomic_json(root / "status.json", {"status": "complete", "phase": "complete"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=GPUS)
    parser.add_argument("--finalize-target", choices=TARGET_DIRS)
    parser.add_argument("--evaluation-gpu", choices=("0", "1", "2", "3"))
    parser.add_argument("--complete-root", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.worker, args.finalize_target, args.complete_root)) > 1:
        raise ValueError("worker, finalizer, and root completion modes are mutually exclusive")
    if args.evaluation_gpu and not args.finalize_target:
        raise ValueError("--evaluation-gpu requires --finalize-target")
    args.output_dir = args.output_dir.expanduser().resolve(strict=True)
    return args


if __name__ == "__main__":
    parsed = parse_args()
    check_environment()
    if parsed.worker:
        worker(external_output(parsed.output_dir), parsed.worker, parsed.resume)
    elif parsed.finalize_target:
        finalize_target(external_output(parsed.output_dir), parsed.finalize_target, parsed.evaluation_gpu)
    elif parsed.complete_root:
        complete_root(external_output(parsed.output_dir))
    else:
        coordinator(parsed)
