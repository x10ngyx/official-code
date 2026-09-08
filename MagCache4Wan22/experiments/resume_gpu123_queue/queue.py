#!/usr/bin/env python3
"""Wait for the preceding experiment, then resume the unchanged GPU123 scan."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for key in THREADS:
    os.environ[key] = "1"
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "runtime"))
from common import artifact, check_environment, external_output, read_json, sha256, validate_source


def write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def dependency_complete(root):
    try:
        done = read_json(root / "COORDINATOR_EXIT.json")
        quality = read_json(root / "HELDOUT_COMPLETE.json")
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (done.get("status") == "complete" and done.get("exit_code") == 0
            and quality.get("status") == "complete" and quality.get("candidate_cells") == 50
            and quality.get("checkpoint_epoch") == 304)


def gpu_availability(gpu_ids):
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                        "--format=csv,noheader,nounits"], text=True)
    occupied = {line.split(",")[0].strip() for line in processes.splitlines() if line.strip()}
    result = {}
    for line in gpu_rows.splitlines():
        gpu, uuid, memory, utilization = [v.strip() for v in line.split(",")]
        if gpu in gpu_ids:
            result[gpu] = dict(uuid=uuid, memory_mib=int(memory), utilization=int(utilization),
                               idle=uuid not in occupied and int(memory) <= 1024 and int(utilization) <= 5)
    if set(result) != set(gpu_ids):
        raise RuntimeError("assigned GPUs missing from nvidia-smi")
    return result


def validate_inputs(config):
    for record in config["frozen_inputs"]:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"queued input changed: {record['path']}")
    scan = Path(config["scan_root"])
    if validate_source(config["source"]) != read_json(scan / "launch_config.json")["source"]:
        raise ValueError("scan source changed while queued")


def archive_old_state(scan, queue):
    history = queue / "prior_state"
    history.mkdir(exist_ok=True)
    (history / "README.md").write_text("# Prior execution state\n\nHistorical stop and worker status records; measured videos/manifests remain in the original scan.\n")
    for path in [scan / "STOPPED_BY_USER.json", *scan.glob("target_*/worker_status/worker_000.json")]:
        if path.exists():
            # Stale ready/running records must not bypass staggered resume loading.
            if path.name == "worker_000.json":
                pid = read_json(path).get("pid")
                proc = Path(f"/proc/{pid}/cmdline")
                if proc.exists():
                    command = proc.read_bytes().replace(b"\0", b" ").decode(errors="replace")
                    if str(scan) in command and "MagCache4Wan22" in command:
                        raise RuntimeError("a previous scan worker is still active")
            name = "__".join(path.relative_to(scan).parts)
            destination = history / f"{time.time_ns()}__{name}"
            path.rename(destination)


def main(queue):
    check_environment()
    queue = external_output(queue)
    config = read_json(queue / "queue_config.json")
    scan = external_output(config["scan_root"])
    predecessor = Path(config["predecessor_root"])
    child = None
    state = dict(status="queued", phase="waiting_for_predecessor", pid=os.getpid(),
                 scan_root=str(scan), predecessor_root=str(predecessor))
    with (scan / ".resume_queue.lock").open("a") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            validate_inputs(config)
            idle_checks = 0
            while True:
                ready = dependency_complete(predecessor)
                gpus = gpu_availability(config["gpu_ids"])
                idle_checks = idle_checks + 1 if ready and all(v["idle"] for v in gpus.values()) else 0
                state.update(status="queued", phase="confirming_gpu_idle" if ready else "waiting_for_predecessor",
                             predecessor_complete=ready, gpus=gpus, idle_checks=idle_checks, updated_unix=time.time())
                write(queue / "status.json", state)
                write(scan / "status.json", dict(status="queued", phase=state["phase"],
                                                queue_dir=str(queue), queue_pid=os.getpid()))
                if idle_checks >= 3:
                    break
                time.sleep(10)
            validate_inputs(config)
            with (scan / ".gpu123.lock").open("a") as scan_lock:
                fcntl.flock(scan_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                archive_old_state(scan, queue)
            command = [sys.executable, "-u", str(PROJECT / "experiments/targeted_threshold_scan/run_gpu123.py"),
                       "--output-dir", str(scan), "--resume"]
            with (queue / "resume.console.log").open("a") as log:
                child = subprocess.Popen(command, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(status="running", phase="resuming_scan", child_pid=child.pid,
                             command=command, resumed_unix=time.time())
                write(queue / "status.json", state)
                write(scan / "RESUMED.json", dict(status="resumed", queue_dir=str(queue),
                                                 predecessor_completion=artifact(predecessor / "COORDINATOR_EXIT.json"),
                                                 child_pid=child.pid, resumed_unix=time.time()))
                code = child.wait()
            if code != 0 or not (scan / "COMPLETE.json").exists():
                raise RuntimeError(f"resumed scan did not complete successfully: exit {code}")
            state.update(status="complete", phase="complete", completed_unix=time.time())
            write(queue / "status.json", state)
        except BaseException as exc:
            state.update(status="failed", error=repr(exc), updated_unix=time.time())
            write(queue / "status.json", state)
            raise
        finally:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=50)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, required=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main(parser.parse_args().queue_dir)
