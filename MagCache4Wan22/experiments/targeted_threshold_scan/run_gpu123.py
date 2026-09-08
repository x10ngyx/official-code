#!/usr/bin/env python3
"""Run the three targets on physical GPU1/2/3 with one shared FLOPs profile."""
from __future__ import annotations
import argparse
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import fcntl
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "runtime"))
from common import (artifact, check_environment, checkpoint_path, external_output,
                    index_result, read_json, validate_source, write_json)
from batch import atomic_json
from aggregate_performance import validate_profile


def gpu_idle(gpu):
    processes = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    return not processes.strip()


def run(args):
    check_environment()
    source = validate_source(args.source)
    checkpoint = checkpoint_path(args.checkpoint)
    root = external_output(args.output_dir)
    if root.exists() and not args.resume:
        raise FileExistsError(f"use --resume for existing output: {root}")
    root.mkdir(parents=True, exist_ok=True)
    index_result(root)
    with (root / ".gpu123.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute(args, source, checkpoint, root)


def execute(args, source, checkpoint, root):
    profile_path = root / "shared_profile/calflops.json"
    assignments = [("1", "1.8", "target_1p8_gpu1"), ("2", "2.4", "target_2p4_gpu2"),
                   ("3", "3.0", "target_3p0_gpu3")]
    config = dict(source=source, checkpoint=str(checkpoint),
                  presets=artifact(Path(__file__).with_name("presets.json")),
                  assignments=[dict(physical_gpu=g, target=t, directory=n) for g, t, n in assignments],
                  profile=str(profile_path), baseline_policy="fresh baseline per prompt per assigned GPU",
                  measured_videos_by_target=[44, 77, 66], measured_videos_total=187,
                  quality="VideoMetrics RGB PSNR/SSIM/LPIPS and standard 16-dimension VBench subset")
    if (root / "launch_config.json").exists():
        if read_json(root / "launch_config.json") != config:
            raise ValueError("GPU123 resume configuration changed")
    else:
        write_json(root / "launch_config.json", config)
    (root / "README.md").write_text(
        "# MagCache fixed-target GPU1/2/3 scan\n\nlaunch_config.json records the immutable assignment and sources. "
        "shared_profile/ contains one actual component FLOPs profile; target_1p8_gpu1/, target_2p4_gpu2/, "
        "target_3p0_gpu3/ each run 11 prompts with fresh same-GPU baselines and full quality evaluation. "
        "status.json and console logs describe execution; COMPLETE.json requires all three target suites. "
        "Baseline counts are 11 per GPU, for 187 measured videos plus three warmups.\n")
    state = dict(pid=os.getpid(), phase="waiting_for_profile_gpu", status="running", targets={})
    children, handles = [], []

    def save():
        atomic_json(root / "status.json", dict(state, updated_unix=time.time()))

    def spawn(command, gpu, log_name):
        log = (root / log_name).open("a")
        handles.append(log)
        child = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                                 PYTHONUNBUFFERED="1"), stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
        return child

    try:
        save()
        if not profile_path.exists():
            while not gpu_idle("1"):
                time.sleep(5)
            state["phase"] = "profiling_on_gpu1"
            command = [sys.executable, str(PROJECT / "experiments/performance_t2v_a14b/profile_calflops.py"),
                       "--wan22-root", source["source"], "--checkpoint-dir", str(checkpoint),
                       "--output", str(profile_path)]
            child = spawn(command, "1", "shared_profile.log")
            state["profile_pid"] = child.pid
            save()
            if child.wait() != 0:
                raise RuntimeError("shared FLOPs profile failed; see shared_profile.log")
        profile = read_json(profile_path)
        validate_profile(profile)
        if profile["source"]["prepared_manifest"] != source:
            raise ValueError("shared profile source changed")
        atomic_json(root / "profile_artifact.json", artifact(profile_path))
        pending, active, last_started = list(assignments), {}, None
        state["phase"] = "scanning"
        for gpu, target, name in assignments:
            state["targets"][name] = dict(physical_gpu=gpu, target=target, status="queued")
        save()
        while pending or active:
            for name, child in list(active.items()):
                code = child.poll()
                if code is not None:
                    completed = code == 0 and (root / name / "COMPLETE.json").is_file()
                    state["targets"][name].update(status="complete" if completed else "failed", exit_code=code)
                    del active[name]
            # Stagger model loading, while allowing an occupied GPU to queue independently.
            previous_ready = last_started is None or last_started not in active
            if not previous_ready:
                marker = root / last_started / "worker_status/worker_000.json"
                previous_ready = marker.exists() and read_json(marker).get("status") in {"ready", "running", "complete"}
            if previous_ready:
                for item in pending:
                    gpu, target, name = item
                    if not gpu_idle(gpu):
                        state["targets"][name]["status"] = "waiting_for_gpu"
                        continue
                    command = [sys.executable, str(Path(__file__).with_name("run_scan.py")),
                               "--source", source["source"], "--checkpoint", str(checkpoint),
                               "--output-dir", str(root / name), "--gpu-ids", gpu,
                               "--target-speedups", target, "--calflops-profile", str(profile_path)]
                    if args.resume:
                        command.append("--resume")
                    child = spawn(command, gpu, name + ".console.log")
                    state["targets"][name].update(status="running", controller_pid=child.pid, command=command)
                    active[name], last_started = child, name
                    pending.remove(item)
                    break
            save()
            if pending or active:
                time.sleep(5)
        if any(v["status"] != "complete" for v in state["targets"].values()):
            raise RuntimeError("one or more target scans failed; inspect target status/logs")
        selected = {name: read_json(root / name / "target_selection.json") for _, _, name in assignments}
        atomic_json(root / "COMPLETE.json", dict(status="complete", targets=selected, config=artifact(root / "launch_config.json")))
        state.update(status="complete", phase="complete")
        save()
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        save()
        raise
    finally:
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
        for child in children:
            try:
                child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "build/Wan2.2-42bf4cf")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT.parents[2] / "models/Wan2.2-T2V-A14B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    run(args)
