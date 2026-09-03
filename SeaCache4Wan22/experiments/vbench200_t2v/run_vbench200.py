#!/usr/bin/env python3
"""Run the complete SeaCache4Wan22 Vbench200 benchmark."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent
WORKSPACE_ROOT = REPOSITORY_DIR.parents[1]
EXP_ROOT = Path("/all/yiran07-disk3/huteng_data/exp").resolve()
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "models" / "Wan2.2-T2V-A14B"
DEFAULT_VBENCH_CACHE = WORKSPACE_ROOT / "models" / "VBench"
DEFAULT_METRICS_CACHE = WORKSPACE_ROOT / "models" / "torch-cache"
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}


def generation_protocol() -> dict[str, Any]:
    return {
        "task": "t2v-A14B", "size": "832*480", "frame_num": 45, "fps": 16,
        "sample_steps": 50, "sample_solver": "dpm++", "sample_shift": 12.0,
        "guide_scale_low_high": [3.0, 4.0], "boundary": 0.875, "seed": 42,
        "dtype": "bfloat16", "offload_model": True, "t5_cpu": False,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as exc:
        raise ValueError(f"output must be below {EXP_ROOT}: {resolved}") from exc
    return resolved


def check_python(path: Path, modules: list[str], label: str) -> None:
    result = subprocess.run(
        [str(path), "-c", ";".join(f"import {name}" for name in modules)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ, **THREAD_ENV},
    )
    if result.returncode:
        raise RuntimeError(f"{label} Python lacks {modules}: {path}\n{result.stderr}")


def write_locked(path: Path, payload: dict[str, Any], resume: bool) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if not resume:
            raise FileExistsError(f"configuration exists; use --resume: {path}")
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"resume configuration mismatch: {path}")
        return
    path.write_text(rendered, encoding="utf-8")


def _class_method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise ValueError(f"missing {class_name}.{method_name}")


def _baseline_model_refactor_is_equivalent(source_model: Path, current_model: Path) -> bool:
    source_tree = ast.parse(source_model.read_text(encoding="utf-8"))
    current_tree = ast.parse(current_model.read_text(encoding="utf-8"))
    source_forward = _class_method(source_tree, "WanAttentionBlock", "forward")
    source_self_attention = [
        call for call in ast.walk(source_forward)
        if isinstance(call, ast.Call) and ast.unparse(call.func) == "self.self_attn"
    ]
    current_helper = _class_method(current_tree, "WanAttentionBlock", "_modulated_norm1")
    helper_returns = [node for node in current_helper.body if isinstance(node, ast.Return)]
    return (
        len(source_self_attention) == 1
        and bool(source_self_attention[0].args)
        and len(helper_returns) == 1
        and ast.dump(source_self_attention[0].args[0]) == ast.dump(helper_returns[0].value)
    )


def validate_reusable_baseline(
    source: Path, current_prepared_manifest: Path, expected_count: int
) -> dict[str, Any]:
    """Validate a completed no-cache baseline and record exact reuse provenance."""
    config_paths = sorted(source.glob("generation_config.shard_*.json"))
    if not config_paths:
        raise ValueError(f"baseline source has no generation configs: {source}")
    expected_prompt_sha = sha256(REPOSITORY_DIR / "Vbench200" / "prompts.jsonl")
    source_manifest_paths: set[Path] = set()
    source_manifest_hashes: set[str] = set()
    shard_indices: set[int] = set()
    num_shards: set[int] = set()
    for path in config_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        requirements = (
            (payload.get("condition") == "baseline", "condition"),
            (payload.get("selected_prompt_count") == expected_count, "prompt count"),
            (payload.get("prompt_manifest_sha256") == expected_prompt_sha, "prompt manifest"),
            (payload.get("protocol") == generation_protocol(), "generation protocol"),
            (payload.get("thread_env") == THREAD_ENV, "thread environment"),
        )
        for satisfied, label in requirements:
            if not satisfied:
                raise ValueError(f"baseline source {label} mismatch: {path}")
        source_manifest = Path(payload["prepared_manifest"]).expanduser().resolve(strict=True)
        if sha256(source_manifest) != payload.get("prepared_manifest_sha256"):
            raise ValueError(f"baseline prepared manifest hash mismatch: {path}")
        source_manifest_paths.add(source_manifest)
        source_manifest_hashes.add(payload["prepared_manifest_sha256"])
        shard_indices.add(payload["shard_index"])
        num_shards.add(payload["num_shards"])
    if len(source_manifest_paths) != 1 or len(source_manifest_hashes) != 1:
        raise ValueError("baseline shards do not share one prepared manifest")
    if len(num_shards) != 1 or shard_indices != set(range(next(iter(num_shards)))):
        raise ValueError("baseline source shard set is incomplete")

    prompt_rows = [
        json.loads(line)
        for line in (REPOSITORY_DIR / "Vbench200" / "prompts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ][:expected_count]
    expected_ids = {str(row["sample_id"]) for row in prompt_rows}
    video_ids = {path.stem for path in (source / "videos").glob("*.mp4") if path.stat().st_size > 0}
    timing_ids = {path.stem for path in (source / "timings").glob("*.json") if path.stat().st_size > 0}
    if video_ids != expected_ids or timing_ids != expected_ids:
        raise ValueError("baseline source does not contain the exact expected video/timing set")
    for sample_id in expected_ids:
        timing = json.loads((source / "timings" / f"{sample_id}.json").read_text(encoding="utf-8"))
        seconds = timing.get("pipeline_generate_wall_seconds")
        if (
            timing.get("status") != "success"
            or timing.get("implementation") != "wan22"
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError(f"invalid reusable baseline timing: {sample_id}")

    source_prepared_manifest = next(iter(source_manifest_paths))
    source_prepared = json.loads(source_prepared_manifest.read_text(encoding="utf-8"))
    current_prepared = json.loads(current_prepared_manifest.read_text(encoding="utf-8"))
    source_hashes = source_prepared.get("sha256", {})
    current_hashes = current_prepared.get("sha256", {})
    changed_files = sorted(
        key for key in source_hashes.keys() | current_hashes.keys()
        if source_hashes.get(key) != current_hashes.get(key)
    )
    if changed_files != ["wan/modules/model.py"]:
        raise ValueError(f"baseline/current prepared files differ beyond model.py: {changed_files}")
    source_root = Path(source_prepared["source"]).expanduser().resolve(strict=True)
    current_root = Path(current_prepared["source"]).expanduser().resolve(strict=True)
    if not _baseline_model_refactor_is_equivalent(
        source_root / "wan" / "modules" / "model.py",
        current_root / "wan" / "modules" / "model.py",
    ):
        raise ValueError("baseline/current norm1 refactor is not AST-equivalent")
    return {
        "source": str(source),
        "artifact_counts": {"videos": len(video_ids), "timings": len(timing_ids)},
        "generation_config_sha256": {
            path.name: sha256(path) for path in config_paths
        },
        "source_prepared_manifest": str(source_prepared_manifest),
        "source_prepared_manifest_sha256": next(iter(source_manifest_hashes)),
        "current_prepared_manifest_sha256": sha256(current_prepared_manifest),
        "prepared_files_changed": changed_files,
        "compatibility_check": "WanAttentionBlock norm1 expression AST-equivalent",
    }


def link_directory(link: Path, target: Path) -> None:
    if link.is_symlink() and link.resolve() == target:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"refusing existing baseline link path: {link}")
    link.symlink_to(target, target_is_directory=True)


def status_write(root: Path, payload: dict[str, Any]) -> None:
    (root / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_logged(command: list[str], log: Path, *, env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    if dry_run:
        print(shlex.join(command))
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        log = log.with_name(f"{log.stem}.{stamp}{log.suffix}")
    with log.open("x", encoding="utf-8") as handle:
        handle.write(f"command: {shlex.join(command)}\n")
        handle.flush()
        subprocess.run(command, env=env or {**os.environ, **THREAD_ENV}, stdout=handle, stderr=subprocess.STDOUT, text=True, check=True)


def gpu_memory_used_mib(gpu: str) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **THREAD_ENV},
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1 or not rows[0].isdigit():
        raise RuntimeError(f"cannot parse GPU {gpu} memory usage: {result.stdout!r}")
    return int(rows[0])


def generation_command(args: argparse.Namespace, condition: str, shard: int) -> list[str]:
    command = [
        str(args.generation_python), str(SCRIPT_DIR / "generate_vbench200.py"),
        "--condition", condition, "--wan22-root", str(args.wan22_root),
        "--checkpoint-dir", str(args.checkpoint_dir), "--output-dir", str(args.output_dir / condition),
        "--shard-index", str(shard), "--num-shards", str(len(args.gpu_ids)),
        "--physical-gpu", args.gpu_ids[shard],
    ]
    if condition == "seacache":
        command.extend(["--threshold", str(args.threshold)])
        if args.use_ret_steps:
            command.append("--use-ret-steps")
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.resume:
        command.append("--resume")
    return command


def run_generation(args: argparse.Namespace, condition: str, status: dict[str, Any]) -> None:
    commands = [generation_command(args, condition, shard) for shard in range(len(args.gpu_ids))]
    if args.dry_run:
        for gpu, command in zip(args.gpu_ids, commands):
            print(f"CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}")
        return
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records: list[dict[str, Any]] = []
    started = time.monotonic()

    def start_worker(shard: int) -> dict[str, Any]:
        gpu, command = args.gpu_ids[shard], commands[shard]
        log = args.output_dir / "orchestration_logs" / f"generate_{condition}_shard_{shard}.{stamp}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("x", encoding="utf-8")
        handle.write(f"CUDA_VISIBLE_DEVICES={gpu}\ncommand: {shlex.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            env={**os.environ, **THREAD_ENV, "CUDA_VISIBLE_DEVICES": gpu},
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        record = {
            "shard": shard,
            "gpu": gpu,
            "command": command,
            "process": process,
            "handle": handle,
            "log": log,
            "status_path": (
                args.output_dir
                / condition
                / "worker_status"
                / f"worker_{shard:03d}.json"
            ),
        }
        records.append(record)
        return record

    def worker_ready(record: dict[str, Any]) -> bool:
        path = record["status_path"]
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if (
            payload.get("worker_pid") != record["process"].pid
            or payload.get("persistent_pipeline") is not True
            or payload.get("pipeline_initialization_count") != 1
            or payload.get("status") not in {"running", "complete"}
        ):
            return False
        return (
            args.stagger_workers_gpu_memory_mib is None
            or gpu_memory_used_mib(record["gpu"])
            >= args.stagger_workers_gpu_memory_mib
        )

    def wait_until_wave_ready(wave: list[dict[str, Any]]) -> None:
        deadline = time.monotonic() + args.stagger_worker_timeout_seconds
        while True:
            early_failures = [
                record for record in wave
                if record["process"].poll() is not None and not worker_ready(record)
            ]
            if early_failures:
                raise RuntimeError(
                    "worker exited before persistent pipeline readiness: "
                    + ", ".join(str(record["log"]) for record in early_failures)
                )
            if all(worker_ready(record) for record in wave):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "worker wave did not complete one-time pipeline initialization "
                    f"within {args.stagger_worker_timeout_seconds} seconds"
                )
            time.sleep(5)

    launch_waves: list[dict[str, Any]] = []
    try:
        for wave_start in range(0, len(commands), args.worker_launch_wave_size):
            shards = list(
                range(
                    wave_start,
                    min(wave_start + args.worker_launch_wave_size, len(commands)),
                )
            )
            print(
                f"Launching {condition} worker wave on GPUs "
                + ",".join(args.gpu_ids[shard] for shard in shards),
                flush=True,
            )
            wave = [start_worker(shard) for shard in shards]
            wave_payload = {
                "shards": shards,
                "gpus": [args.gpu_ids[shard] for shard in shards],
                "launched_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            launch_waves.append(wave_payload)
            if wave_start + args.worker_launch_wave_size < len(commands):
                wait_until_wave_ready(wave)
                wave_payload["all_workers_pipeline_ready_utc"] = (
                    dt.datetime.now(dt.timezone.utc).isoformat()
                )
                print(
                    f"Worker wave GPUs {','.join(wave_payload['gpus'])} passed "
                    "the one-time pipeline-load peak",
                    flush=True,
                )
    except BaseException:
        for record in records:
            if record["process"].poll() is None:
                record["process"].terminate()
        for record in records:
            try:
                record["process"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                record["process"].kill()
                record["process"].wait()
            record["handle"].close()
        raise

    failures = []
    for record in records:
        code = record["process"].wait()
        record["handle"].close()
        if code:
            failures.append({"returncode": code, "log": str(record["log"])})
    status["phases"][f"generation_{condition}"] = {
        "status": "failed" if failures else "complete",
        "orchestration_wall_seconds": time.monotonic() - started,
        "launch_waves": launch_waves,
        "failures": failures,
    }
    status_write(args.output_dir, status)
    if failures:
        raise RuntimeError(f"{condition} generation failed: {failures}")


def link_result(root: Path) -> None:
    link = PROJECT_DIR / "experiment_results" / root.name
    if link.is_symlink() and link.resolve() == root:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"result link already exists: {link}")
    link.symlink_to(root, target_is_directory=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--wan22-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--baseline-source",
        type=Path,
        help="Reuse a fully validated no-cache baseline with explicit provenance.",
    )
    parser.add_argument("--gpu-ids", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument(
        "--stagger-workers-gpu-memory-mib",
        type=int,
        help=(
            "Require every worker in the current launch wave to reach this GPU "
            "memory usage before the next wave; avoids synchronized A14B "
            "host-memory peaks."
        ),
    )
    parser.add_argument(
        "--worker-launch-wave-size",
        type=int,
        default=1,
        help=(
            "Workers launched concurrently per wave; the next wave waits until "
            "every current worker has completed its one-time pipeline load."
        ),
    )
    parser.add_argument("--stagger-worker-timeout-seconds", type=int, default=1800)
    parser.add_argument("--generation-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--calflops-source", type=Path)
    parser.add_argument("--video-metrics-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--vbench-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--video-metrics-cache-dir", type=Path, default=DEFAULT_METRICS_CACHE)
    parser.add_argument("--vbench-cache-dir", type=Path, default=DEFAULT_VBENCH_CACHE)
    parser.add_argument("--use-ret-steps", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--defer-evaluation",
        action="store_true",
        help=(
            "Run generation and performance aggregation now, retain evaluation "
            "as pending, then resume later without this flag."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        raise ValueError("--threshold must be finite and positive")
    if not args.gpu_ids or len(set(args.gpu_ids)) != len(args.gpu_ids) or any(not gpu.isdigit() for gpu in args.gpu_ids):
        raise ValueError("--gpu-ids must be distinct non-negative integers")
    if args.stagger_workers_gpu_memory_mib is not None and args.stagger_workers_gpu_memory_mib < 1:
        raise ValueError("--stagger-workers-gpu-memory-mib must be positive")
    if not 1 <= args.worker_launch_wave_size <= len(args.gpu_ids):
        raise ValueError("--worker-launch-wave-size must be in [1, number of GPUs]")
    if args.stagger_worker_timeout_seconds < 1:
        raise ValueError("--stagger-worker-timeout-seconds must be positive")
    if args.limit is not None and (not 1 <= args.limit <= 200 or not args.skip_evaluation):
        raise ValueError("--limit must be in [1,200] and requires --skip-evaluation")
    if args.skip_evaluation and args.defer_evaluation:
        raise ValueError("--skip-evaluation and --defer-evaluation are mutually exclusive")
    args.output_dir = external(args.output_dir)
    args.wan22_root = args.wan22_root.expanduser().resolve(strict=True)
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve(strict=not args.dry_run)
    for name in ("generation_python", "video_metrics_python", "vbench_python"):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    args.video_metrics_cache_dir = args.video_metrics_cache_dir.expanduser().resolve(strict=not args.dry_run)
    args.vbench_cache_dir = args.vbench_cache_dir.expanduser().resolve(strict=not args.dry_run)
    if args.calflops_source is not None:
        args.calflops_source = args.calflops_source.expanduser().resolve(strict=True)
    if not args.dry_run:
        check_python(args.generation_python, ["torch"], "generation")
        if args.calflops_source is None:
            check_python(args.generation_python, ["calflops"], "Calflops")
        if not args.skip_evaluation and not args.defer_evaluation:
            check_python(args.video_metrics_python, ["torch", "lpips"], "VideoMetrics")
            check_python(args.vbench_python, ["torch", "vbench"], "VBench")
    prepared = args.wan22_root / ".seacache4wan22_prepared.json"
    if not prepared.is_file():
        raise FileNotFoundError(f"Wan2.2 source must be prepared by SeaCache4Wan22: {args.wan22_root}")
    baseline_reuse = None
    if args.baseline_source is not None:
        args.baseline_source = external(args.baseline_source)
        baseline_reuse = validate_reusable_baseline(
            args.baseline_source, prepared, args.limit or 200
        )
    config = {
        "schema": "seacache4wan22_vbench200_run_v1", "method": "SeaCache4Wan22",
        "threshold": args.threshold, "use_ret_steps": args.use_ret_steps,
        "protocol": {
            "dataset": "Vbench200", "prompt_count": args.limit or 200, "model": "Wan2.2-T2V-A14B",
            "video": {"width": 832, "height": 480, "frames": 45, "fps": 16},
            "sampling": {"steps": 50, "solver": "dpm++", "shift": 12.0, "guide_scale_low_high": [3.0, 4.0], "boundary": 0.875, "seed": 42},
            "precision": "DiT bfloat16", "model_cpu_offload": True, "t5_cpu": False,
        },
        "gpu_ids": args.gpu_ids,
        "worker_start_stagger": {
            "wave_size": args.worker_launch_wave_size,
            "next_wave_readiness": "all current workers completed one-time pipeline initialization",
            "gpu_memory_mib": args.stagger_workers_gpu_memory_mib,
            "timeout_seconds": args.stagger_worker_timeout_seconds,
        },
        "baseline_reuse": baseline_reuse,
        "generation_runner": {
            "candidate": "one persistent WanT2V pipeline per GPU worker",
            "sample_batch_size": 1,
            "profiler": "fresh install/restore per sample",
            "baseline": (
                "reused legacy per-sample-process artifacts"
                if baseline_reuse is not None
                else "one persistent WanT2V pipeline per GPU worker"
            ),
            "timing_scope_unchanged": (
                "pipeline_generate_wall_seconds excludes pipeline initialization "
                "and MP4 export for both baseline and candidate"
            ),
            "lifecycle_caveat": (
                "The reused baseline used a fresh process per sample while the "
                "candidate uses persistent workers; process-level warm state can "
                "slightly affect latency but initialization is never counted."
                if baseline_reuse is not None
                else None
            ),
        },
        "paths": {
            "wan22_root": str(args.wan22_root), "prepared_manifest_sha256": sha256(prepared),
            "checkpoint_dir": str(args.checkpoint_dir), "generation_python": str(args.generation_python),
            "calflops_source": str(args.calflops_source) if args.calflops_source else None,
            "video_metrics_python": str(args.video_metrics_python), "vbench_python": str(args.vbench_python),
            "output_dir": str(args.output_dir),
        },
        "scripts": {path.name: sha256(path) for path in sorted(SCRIPT_DIR.glob("*.py"))},
        "shared_profile_script": {
            "path": str(SCRIPT_DIR.parent / "performance_t2v_a14b" / "profile_calflops.py"),
            "sha256": sha256(SCRIPT_DIR.parent / "performance_t2v_a14b" / "profile_calflops.py"),
        },
        "evaluation_enabled": not args.skip_evaluation, "thread_env": THREAD_ENV,
    }
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
    else:
        if args.output_dir.exists() and not args.resume:
            raise FileExistsError(f"output exists; use --resume: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_locked(args.output_dir / "run_config.json", config, args.resume)
        if not (args.output_dir / "README.md").exists():
            (args.output_dir / "README.md").write_text(
                "# SeaCache4Wan22 Vbench200 result\n\nBaseline/candidate generation, trace-weighted performance, repository-standard quality metrics, and final reports.\n",
                encoding="utf-8",
            )
        link_result(args.output_dir)
    status = {"status": "running", "phases": {}}
    if not args.dry_run:
        status_write(args.output_dir, status)
    if args.baseline_source is None:
        run_generation(args, "baseline", status)
    else:
        if args.dry_run:
            print(f"reuse baseline: {args.baseline_source}")
        else:
            link_directory(args.output_dir / "baseline", args.baseline_source)
            status["phases"]["generation_baseline"] = {
                "status": "reused",
                "provenance": baseline_reuse,
            }
            status_write(args.output_dir, status)
    run_generation(args, "seacache", status)
    profile = args.output_dir / "performance" / "calflops_profile.json"
    if not (args.resume and profile.is_file()):
        profile_command = [
            str(args.generation_python), str(SCRIPT_DIR / "profile_calflops.py"),
            "--wan22-root", str(args.wan22_root), "--checkpoint-dir", str(args.checkpoint_dir), "--output", str(profile),
        ]
        if args.calflops_source is not None:
            profile_command.extend(["--calflops-source", str(args.calflops_source)])
        run_logged(profile_command, args.output_dir / "orchestration_logs" / "calflops_profile.log",
            env={**os.environ, **THREAD_ENV, "CUDA_VISIBLE_DEVICES": args.gpu_ids[0]}, dry_run=args.dry_run)
    performance = args.output_dir / "performance" / "summary.json"
    if not (args.resume and performance.is_file()):
        run_logged([
            str(args.generation_python), str(SCRIPT_DIR / "aggregate_performance.py"),
            "--baseline-dir", str(args.output_dir / "baseline"), "--seacache-dir", str(args.output_dir / "seacache"),
            "--calflops-profile", str(profile), "--output-dir", str(args.output_dir / "performance"),
            "--expected-videos", str(args.limit or 200),
        ], args.output_dir / "orchestration_logs" / "aggregate_performance.log", dry_run=args.dry_run)
    if not args.dry_run:
        status["phases"]["performance"] = {"status": "complete", "summary": str(performance)}
        status_write(args.output_dir, status)
    if args.defer_evaluation:
        if not args.dry_run:
            status["status"] = "inference_complete_evaluation_pending"
            status["phases"]["evaluation_and_report"] = {"status": "pending"}
            status_write(args.output_dir, status)
        return
    if not args.skip_evaluation:
        metrics = args.output_dir / "evaluation" / "video_metrics" / "summary.json"
        candidate_vbench = args.output_dir / "evaluation" / "vbench_candidate" / "vbench200_aggregate_scores.json"
        if not (args.resume and metrics.is_file() and candidate_vbench.is_file()):
            command = [
                str(args.generation_python), str(SCRIPT_DIR / "evaluate_results.py"),
                "--reference-videos", str(args.output_dir / "baseline" / "videos"),
                "--candidate-videos", str(args.output_dir / "seacache" / "videos"),
                "--output-dir", str(args.output_dir / "evaluation"), "--expected-frames", "45",
                "--gpu-id", args.gpu_ids[0], "--video-metrics-python", str(args.video_metrics_python),
                "--vbench-python", str(args.vbench_python), "--video-metrics-cache-dir", str(args.video_metrics_cache_dir),
                "--vbench-cache-dir", str(args.vbench_cache_dir),
            ]
            if args.resume:
                command.append("--resume")
            run_logged(command, args.output_dir / "orchestration_logs" / "evaluation.log", dry_run=args.dry_run)
        report = args.output_dir / "benchmark_report.json"
        if not (args.resume and report.is_file()):
            run_logged([str(args.generation_python), str(SCRIPT_DIR / "build_final_report.py"), "--result-dir", str(args.output_dir)], args.output_dir / "orchestration_logs" / "final_report.log", dry_run=args.dry_run)
        if not args.dry_run:
            status["phases"]["evaluation_and_report"] = {"status": "complete", "report": str(report)}
    if not args.dry_run:
        status["status"] = "complete"
        status_write(args.output_dir, status)


if __name__ == "__main__":
    main()
