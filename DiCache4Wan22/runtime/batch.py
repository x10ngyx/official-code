"""Persistent worker contracts, strict resume, and staggered process launch."""
from __future__ import annotations
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from common import (PROJECT, THREAD_VARS, artifact, read_json, sha256, validate_source,
                    write_json)
from generation import create_pipeline, generate_native, generate_run, make_plan
from official import load_official, method_args

sys.path.insert(0, str(PROJECT / "experiments/performance_t2v_a14b"))
from aggregate_performance import summarize_manifest


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def jobs_for_worker(plan, index):
    if not 0 <= index < len(plan["gpu_ids"]):
        raise ValueError("worker index out of range")
    for i, row in enumerate(plan["prompts"]):
        if i % len(plan["gpu_ids"]) == index:
            for condition in plan["conditions"]:
                yield row, condition


def job_plan(plan, row, condition):
    return make_plan(plan["source"], plan["checkpoint"], row, condition["mode"],
                     method_args(condition["threshold"], condition["retention_ratio"]))


def job_directory(root, row, condition):
    return Path(root) / "runs" / condition["id"] / row["sample_id"]


def validate_completed(path, expected, profile, gpu, warmup_videos=None):
    manifest, _ = summarize_manifest(path, expected["mode"], profile)
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError(f"resume input mismatch: {path}")
    timing = read_json(manifest["timing"]["path"])
    lifecycle = timing.get("pipeline_lifecycle", {})
    if (lifecycle.get("persistent_pipeline") is not True
            or lifecycle.get("physical_gpu") != gpu
            or lifecycle.get("pipeline_init_accounted_in_this_sample") is not False
            or lifecycle.get("profiler_freshly_installed_for_sample") is not True
            or timing.get("pipeline_init_wall_seconds") != 0):
        raise ValueError(f"invalid persistent worker lifecycle: {path}")
    if warmup_videos is not None and lifecycle.get("warmup_videos") != warmup_videos:
        raise ValueError(f"resume warmup mismatch: {path}")
    return manifest


def worker(root, index, resume=False):
    root = Path(root)
    plan = read_json(root / "plan.json")
    gpu = plan["gpu_ids"][index]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise ValueError("worker must expose its one assigned physical GPU")
    if validate_source(plan["source"]["source"]) != plan["source"]:
        raise ValueError("worker source lock changed")
    profile_record = read_json(root / "profile_artifact.json")
    if sha256(profile_record["path"]) != profile_record["sha256"]:
        raise ValueError("profile artifact changed")
    profile = read_json(profile_record["path"])
    pending = []
    for row, condition in jobs_for_worker(plan, index):
        directory = job_directory(root, row, condition)
        expected = job_plan(plan, row, condition)
        if directory.exists():
            if not resume:
                raise FileExistsError(directory)
            if (directory / "manifest.json").exists():
                validate_completed(directory / "manifest.json", expected, profile, gpu, plan["warmup_videos"])
                continue
            # Incomplete attempts are never interpreted as completed or overwritten.
            # Preserve them outside runs/, then regenerate the missing video.
            archive = root / "incomplete_attempts" / condition["id"] / row["sample_id"]
            archive.mkdir(parents=True, exist_ok=True)
            directory.rename(archive / f"attempt_{time.time_ns()}")
        pending.append((row, condition, expected, directory))
    state_path = root / "worker_status" / f"worker_{index:03d}.json"
    state = dict(pid=os.getpid(), worker_index=index, physical_gpu=gpu,
                 persistent_pipeline=True, pipeline_init_count=0, completed_this_process=0,
                 pending_count=len(pending), status="initializing")
    atomic_json(state_path, state)
    if not pending:
        atomic_json(state_path, dict(state, status="complete"))
        return
    sys.path.insert(0, plan["source"]["source"])
    import torch
    try:
        pipeline, init_seconds = create_pipeline(plan["checkpoint"])
        core = load_official()
        state.update(status="ready", pipeline_init_count=1,
                     pipeline_init_wall_seconds_once=init_seconds)
        atomic_json(state_path, state)
        warmup_seconds = 0.
        for _ in range(plan["warmup_videos"]):
            torch.cuda.synchronize()
            started = time.perf_counter()
            video = generate_native(pipeline, pending[0][0]["prompt_en"])
            torch.cuda.synchronize()
            warmup_seconds += time.perf_counter() - started
            del video
            gc.collect()
            torch.cuda.empty_cache()
        for row, condition, expected, directory in pending:
            state.update(status="running", sample_id=row["sample_id"], condition=condition["id"])
            atomic_json(state_path, state)
            lifecycle = dict(runner="persistent_batch_worker", persistent_pipeline=True,
                             worker_index=index, physical_gpu=gpu, worker_pid=os.getpid(),
                             pipeline_init_wall_seconds_once=init_seconds,
                             pipeline_init_accounted_in_this_sample=False,
                             profiler_freshly_installed_for_sample=True,
                             warmup_videos=plan["warmup_videos"], warmup_wall_seconds=warmup_seconds)
            path = generate_run(pipeline, core,
                                method_args(condition["threshold"], condition["retention_ratio"]),
                                expected, directory, lifecycle=lifecycle)
            manifest = validate_completed(path, expected, profile, gpu, plan["warmup_videos"])
            with (root / "worker_status" / f"worker_{index:03d}.jsonl").open("a") as log:
                log.write(json.dumps(dict(condition=condition["id"], sample_id=row["sample_id"],
                                          manifest=artifact(path))) + "\n")
            state["completed_this_process"] += 1
            gc.collect()
            torch.cuda.empty_cache()
        atomic_json(state_path, dict(state, status="complete"))
    except BaseException as exc:
        atomic_json(state_path, dict(state, status="failed", error=repr(exc)))
        raise


def launch_workers(root, plan, resume, wave_size, ready_timeout):
    """Launch waves after previous PIDs report pipeline readiness, as in SeaCache."""
    processes, logs = [], []
    def ready(record):
        process, index = record
        state_path = Path(root) / "worker_status" / f"worker_{index:03d}.json"
        if not state_path.exists():
            return False
        state = read_json(state_path)
        return (state.get("pid") == process.pid and state.get("persistent_pipeline") is True
                and state.get("status") in {"ready", "running", "complete"}
                and (state.get("pipeline_init_count") == 1 or state.get("pending_count") == 0))
    try:
        for start in range(0, len(plan["gpu_ids"]), wave_size):
            wave = []
            for index in range(start, min(start + wave_size, len(plan["gpu_ids"]))):
                command = [sys.executable, str(PROJECT / "experiments/vbench200_t2v/generate_vbench200.py"),
                           "--output-dir", str(root), "--worker-index", str(index)]
                if resume:
                    command.append("--resume")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=plan["gpu_ids"][index],
                           PYTHONUNBUFFERED="1", **{k: "1" for k in THREAD_VARS})
                log = (Path(root) / f"worker_{index:03d}.log").open("a")
                logs.append(log)
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                record = (process, index)
                processes.append(record)
                wave.append(record)
            deadline = time.monotonic() + ready_timeout
            while not all(ready(record) for record in wave):
                if any(p.poll() not in (None, 0) for p, _ in processes):
                    raise RuntimeError("generation worker failed; inspect worker logs")
                if any(p.poll() == 0 and not ready((p, i)) for p, i in wave):
                    raise RuntimeError("worker exited without a valid ready/complete record")
                if time.monotonic() >= deadline:
                    raise TimeoutError("worker pipeline readiness timeout")
                time.sleep(2)
        while any(p.poll() is None for p, _ in processes):
            if any(p.poll() not in (None, 0) for p, _ in processes):
                raise RuntimeError("generation worker failed; inspect worker logs")
            time.sleep(2)
        if any(p.returncode != 0 for p, _ in processes):
            raise RuntimeError("generation worker failed")
    finally:
        for process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, _ in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
