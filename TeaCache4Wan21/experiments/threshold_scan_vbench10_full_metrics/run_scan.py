#!/usr/bin/env python3
"""Run the reproducible 10-prompt TeaCache threshold scan and all metrics."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent
GENERATOR = PROJECT_DIR / "experiments" / "vbench200_t2v" / "generate_vbench200.py"
CALFLOPS_PROFILER = PROJECT_DIR / "experiments" / "vbench200_t2v" / "profile_calflops.py"
ANALYZER = SCRIPT_DIR / "analyze_scan.py"
VIDEO_METRICS_ENTRYPOINT = REPOSITORY_DIR / "VideoMetrics" / "evaluate.py"
PROMPTS_PATH = REPOSITORY_DIR / "Vbench200" / "prompts.jsonl"
EXP_ROOT = Path("/mnt/hdd/xiongyuxiang/tmp/exp").resolve()
DEFAULT_PYTHON = Path(
    "/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python"
)
DEFAULT_WAN21_ROOT = Path(
    "/mnt/hdd/xiongyuxiang/tmp/data/source/Wan2.1-65386b2"
)
DEFAULT_CHECKPOINT = Path("/mnt/hdd/xiongyuxiang/tmp/models/Wan2.1-T2V-1.3B")
DEFAULT_TORCH_CACHE = Path("/mnt/hdd/xiongyuxiang/tmp/models/torch-cache")

DEFAULT_SAMPLE_IDS = [
    "vbench200_001",  # temporal flickering / static scene
    "vbench200_016",  # multiple objects
    "vbench200_034",  # human action
    "vbench200_056",  # subject consistency / dynamics / motion smoothness
    "vbench200_085",  # color
    "vbench200_097",  # appearance style
    "vbench200_113",  # temporal style
    "vbench200_135",  # overall consistency / aesthetics / imaging quality
    "vbench200_159",  # scene / background consistency
    "vbench200_177",  # spatial relationship
]

DEFAULT_THRESHOLDS = [
    0.15,
    0.16,
    0.17,
    0.18,
    0.19,
    0.20,
    0.22,
    0.24,
    0.26,
    0.28,
    0.30,
    0.32,
    0.34,
    0.36,
    0.44,
    0.56,
    0.68,
    0.80,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--wan21-root", type=Path, default=DEFAULT_WAN21_ROOT)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--torch-cache", type=Path, default=DEFAULT_TORCH_CACHE)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--sample-ids", nargs="+", default=DEFAULT_SAMPLE_IDS)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lpips-batch-size", type=int, default=8)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("all", "profile", "generate", "metrics", "analyze"),
        default=["all"],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-low-thresholds",
        action="store_true",
        help="allow an explicit threshold grid below 0.15 for low-speed calibration",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as exc:
        raise ValueError(f"output must be below {EXP_ROOT}: {resolved}") from exc
    return resolved


def threshold_label(threshold: float) -> str:
    return f"threshold_{threshold:.4f}".replace(".", "p")


def load_selected_prompts(sample_ids: list[str]) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row["sample_id"]): row for row in rows}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"unknown VBench200 sample ids: {missing}")
    return [by_id[sample_id] for sample_id in sample_ids]


def write_once_or_match(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"existing protocol differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if len(args.sample_ids) != 10 or len(set(args.sample_ids)) != 10:
        raise ValueError("the full scan requires exactly 10 unique prompt IDs")
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4:
        raise ValueError("the locked scan requires exactly four unique GPUs")
    if len(set(args.thresholds)) != len(args.thresholds):
        raise ValueError("thresholds must be unique")
    if args.thresholds != sorted(args.thresholds):
        raise ValueError("thresholds must be sorted in ascending order")
    minimum_threshold = 0.0 if args.allow_low_thresholds else 0.15
    if (
        not args.thresholds
        or args.thresholds[0] < minimum_threshold
        or args.thresholds[-1] > 0.80
    ):
        raise ValueError(
            f"threshold scan must stay within [{minimum_threshold:.2f}, 0.80]"
        )
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.lpips_batch_size < 1:
        raise ValueError("LPIPS batch size must be positive")
    for path in (
        args.python,
        args.wan21_root,
        args.ckpt_dir,
        GENERATOR,
        CALFLOPS_PROFILER,
        ANALYZER,
        VIDEO_METRICS_ENTRYPOINT,
        PROMPTS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(path)


def make_protocol(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> dict[str, Any]:
    covered_dimensions = sorted(
        {str(dimension) for prompt in prompts for dimension in prompt["dimension"]}
    )
    return {
        "schema_version": 1,
        "experiment": "TeaCache Wan2.1 VBench200 10-prompt threshold scan",
        "model": "Wan2.1-T2V-1.3B",
        "source": {
            "wan21_root": str(args.wan21_root.resolve()),
            "checkpoint_dir": str(args.ckpt_dir.resolve()),
            "generator": str(GENERATOR),
            "generator_sha256": sha256(GENERATOR),
            "timing_instrumentation_sha256": sha256(PROJECT_DIR / "inference_timing.py"),
            "prompt_manifest": str(PROMPTS_PATH),
            "prompt_manifest_sha256": sha256(PROMPTS_PATH),
        },
        "generation": {
            "task": "t2v-1.3B",
            "size": "832*480",
            "frame_num": 81,
            "fps": 16,
            "sample_steps": 50,
            "sample_solver": "unipc",
            "sample_shift": 5.0,
            "guide_scale": 5.0,
            "seed": args.seed,
            "parameter_dtype": "bfloat16",
            "offload_model": False,
            "t5_cpu": False,
            "use_ret_steps": False,
        },
        "sample_ids": args.sample_ids,
        "prompts": prompts,
        "covered_dimensions": covered_dimensions,
        "thresholds": args.thresholds,
        "threshold_grid": (
            {
                "explicit_low_speed_extension": args.thresholds,
                "purpose": "bracket approximately 1.8x full-inference speedup",
            }
            if args.allow_low_thresholds
            else {
                "dense_front": "0.15-0.20 step 0.01; 0.22-0.36 step 0.02",
                "sparse_tail": [0.44, 0.56, 0.68, 0.80],
            }
        ),
        "gpus": args.gpus,
        "latency": {
            "headline": "pipeline_generate_wall_seconds",
            "aggregation": "sum(baseline prompt latency) / sum(candidate prompt latency)",
            "includes": ["text_encoding", "denoising", "vae_decode"],
            "excludes": ["model_loading", "mp4_export", "metric_evaluation"],
            "stage_fields": [
                "text_encoding",
                "denoising_core",
                "vae_decode",
                "pipeline_other",
            ],
        },
        "quality": {
            "reference": "matched baseline video with identical prompt and seed",
            "metrics": ["psnr_rgb_db", "ssim_rgb", "lpips_alex_v0_1_spatial"],
            "expected_frames_per_video": 81,
        },
        "flops": {
            "scope": "DiT forward only",
            "method": "Calflops plus analytical dense FlashAttention correction, weighted by actual block trace",
            "reported_units": ["TFLOPs/video", "estimated achieved TFLOP/s"],
        },
        "targets": [1.8, 2.4, 3.0],
    }


def ensure_result_root(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise FileExistsError(f"output is not empty; use --resume: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_once_or_match(args.output_root / "protocol.json", protocol)
    readme = args.output_root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# TeaCache Wan2.1 VBench10 full threshold scan\n\n"
            "This directory contains the immutable protocol, Calflops profile, matched "
            "baseline and TeaCache videos, per-video module timing traces, full-reference "
            "PSNR/SSIM/LPIPS outputs (including per-frame CSVs), and validated aggregate "
            "analysis. Inference latency excludes checkpoint loading and MP4 export.\n",
            encoding="utf-8",
        )


def run_profile(args: argparse.Namespace) -> None:
    output = args.output_root / "profiling" / "calflops.json"
    if output.exists():
        if args.resume:
            print(json.dumps({"phase": "profile", "status": "skipped", "output": str(output)}))
            return
        raise FileExistsError(output)
    command = [
        str(args.python),
        str(CALFLOPS_PROFILER),
        "--wan21-root",
        str(args.wan21_root),
        "--checkpoint-dir",
        str(args.ckpt_dir),
        "--output",
        str(output),
    ]
    if args.dry_run:
        print(json.dumps({"phase": "profile", "command": command}, ensure_ascii=False))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.parent / "calflops.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus[0]
    with log.open("x", encoding="utf-8") as handle:
        subprocess.run(command, cwd=SCRIPT_DIR, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def generation_command(
    args: argparse.Namespace,
    condition_dir: Path,
    implementation: str,
    threshold: float | None,
    shard_index: int,
) -> list[str]:
    command = [
        str(args.python),
        str(GENERATOR),
        "--implementation",
        implementation,
        "--task",
        "t2v-1.3B",
        "--wan21-root",
        str(args.wan21_root),
        "--ckpt-dir",
        str(args.ckpt_dir),
        "--output-dir",
        str(condition_dir),
        "--seeds",
        str(args.seed),
        "--size",
        "832*480",
        "--frame-num",
        "81",
        "--sample-steps",
        "50",
        "--sample-shift",
        "5.0",
        "--guide-scale",
        "5.0",
        "--sample-solver",
        "unipc",
        "--sample-ids",
        *args.sample_ids,
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(len(args.gpus)),
    ]
    if threshold is not None:
        command.extend(("--teacache-thresh", str(threshold)))
    if args.resume:
        command.append("--resume")
    return command


def validate_generation(condition_dir: Path, sample_ids: list[str]) -> None:
    videos = {path.stem for path in (condition_dir / "videos").glob("*.mp4")}
    timings = {path.stem for path in (condition_dir / "timings").glob("*.json")}
    expected = set(sample_ids)
    if videos != expected or timings != expected:
        raise ValueError(
            f"incomplete condition {condition_dir}: videos={sorted(videos)}, timings={sorted(timings)}"
        )
    for sample_id in sample_ids:
        payload = json.loads(
            (condition_dir / "timings" / f"{sample_id}.json").read_text(encoding="utf-8")
        )
        if payload.get("schema_version") != 2 or payload.get("status") != "success":
            raise ValueError(f"invalid staged timing trace: {condition_dir}/{sample_id}")
        if payload.get("model_forward_call_count") != 100:
            raise ValueError(f"expected 100 DiT calls: {condition_dir}/{sample_id}")
        stages = payload.get("stage_wall_seconds")
        if not isinstance(stages, dict):
            raise ValueError(f"missing module timings: {condition_dir}/{sample_id}")


def run_generation_condition(
    args: argparse.Namespace,
    *,
    label: str,
    implementation: str,
    threshold: float | None,
) -> None:
    condition_dir = args.output_root / label
    if condition_dir.exists() and not args.resume:
        raise FileExistsError(condition_dir)
    if condition_dir.exists() and args.resume and not args.dry_run:
        try:
            validate_generation(condition_dir, args.sample_ids)
        except ValueError:
            pass
        else:
            print(
                json.dumps(
                    {"phase": "generate", "condition": label, "status": "skipped"}
                )
            )
            return
    if args.dry_run:
        for shard_index in range(len(args.gpus)):
            print(
                json.dumps(
                    {
                        "phase": "generate",
                        "condition": label,
                        "gpu": args.gpus[shard_index],
                        "command": generation_command(
                            args, condition_dir, implementation, threshold, shard_index
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        return
    condition_dir.mkdir(parents=True, exist_ok=True)
    write_once_or_match(
        condition_dir / "condition.json",
        {
            "schema_version": 1,
            "label": label,
            "implementation": implementation,
            "threshold": threshold,
            "use_ret_steps": False,
        },
    )
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    processes: list[tuple[int, subprocess.Popen[str], Any, Path]] = []
    print(json.dumps({"phase": "generate", "condition": label, "status": "started"}))
    for shard_index, gpu in enumerate(args.gpus):
        command = generation_command(args, condition_dir, implementation, threshold, shard_index)
        log_path = condition_dir / f"controller.shard_{shard_index:03d}.{timestamp}.log"
        handle = log_path.open("x", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        process = subprocess.Popen(
            command,
            cwd=SCRIPT_DIR,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((shard_index, process, handle, log_path))
    failures: list[str] = []
    for shard_index, process, handle, log_path in processes:
        returncode = process.wait()
        handle.close()
        if returncode != 0:
            failures.append(f"shard={shard_index}, returncode={returncode}, log={log_path}")
    if failures:
        raise RuntimeError(f"generation failed for {label}: {'; '.join(failures)}")
    validate_generation(condition_dir, args.sample_ids)
    print(json.dumps({"phase": "generate", "condition": label, "status": "complete"}))


def run_generation(args: argparse.Namespace) -> None:
    run_generation_condition(
        args, label="baseline", implementation="wan21", threshold=None
    )
    for threshold in args.thresholds:
        run_generation_condition(
            args,
            label=threshold_label(threshold),
            implementation="teacache",
            threshold=threshold,
        )


def metrics_complete(output_dir: Path, expected_videos: int) -> bool:
    required = [output_dir / "summary.json", output_dir / "per_video.csv", output_dir / "per_frame.csv"]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    payload = json.loads(required[0].read_text(encoding="utf-8"))
    return (
        payload.get("video_count") == expected_videos
        and payload.get("frame_count_total") == expected_videos * 81
        and payload.get("selected_metrics") == ["psnr", "ssim", "lpips"]
    )


def run_metrics(args: argparse.Namespace) -> None:
    jobs: list[tuple[float, Path]] = []
    for threshold in args.thresholds:
        condition_dir = args.output_root / threshold_label(threshold)
        if not args.dry_run:
            validate_generation(condition_dir, args.sample_ids)
        output_dir = condition_dir / "metrics" / "video_metrics"
        if metrics_complete(output_dir, len(args.sample_ids)):
            if args.resume:
                continue
            raise FileExistsError(output_dir)
        jobs.append((threshold, output_dir))

    for start in range(0, len(jobs), len(args.gpus)):
        batch = jobs[start : start + len(args.gpus)]
        processes: list[tuple[float, subprocess.Popen[str], Any, Path, Path]] = []
        for worker_index, (threshold, output_dir) in enumerate(batch):
            condition_dir = args.output_root / threshold_label(threshold)
            command = [
                str(args.python),
                str(VIDEO_METRICS_ENTRYPOINT),
                "--reference-dir",
                str(args.output_root / "baseline" / "videos"),
                "--candidate-dir",
                str(condition_dir / "videos"),
                "--output-dir",
                str(output_dir),
                "--metrics",
                "psnr",
                "ssim",
                "lpips",
                "--device",
                "cuda:0",
                "--lpips-batch-size",
                str(args.lpips_batch_size),
                "--expected-frames",
                "81",
                "--model-cache",
                str(args.torch_cache),
            ]
            if args.resume and output_dir.exists():
                command.append("--overwrite")
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "phase": "metrics",
                            "threshold": threshold,
                            "gpu": args.gpus[worker_index],
                            "command": command,
                        },
                        ensure_ascii=False,
                    )
                )
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "evaluation.log"
            mode = "a" if args.resume and log_path.exists() else "x"
            handle = log_path.open(mode, encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.gpus[worker_index]
            env["TORCH_HOME"] = str(args.torch_cache)
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_DIR / "VideoMetrics",
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((threshold, process, handle, log_path, output_dir))
        failures: list[str] = []
        for threshold, process, handle, log_path, output_dir in processes:
            returncode = process.wait()
            handle.close()
            if returncode != 0:
                failures.append(f"threshold={threshold}, returncode={returncode}, log={log_path}")
            elif not metrics_complete(output_dir, len(args.sample_ids)):
                failures.append(f"threshold={threshold}, incomplete output={output_dir}")
        if failures:
            raise RuntimeError("metric evaluation failed: " + "; ".join(failures))


def run_analysis(args: argparse.Namespace) -> None:
    command = [str(args.python), str(ANALYZER), "--result-root", str(args.output_root)]
    if args.dry_run:
        print(json.dumps({"phase": "analyze", "command": command}, ensure_ascii=False))
        return
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def main() -> None:
    args = parse_args()
    args.output_root = require_external(args.output_root)
    args.python = args.python.expanduser().resolve()
    args.wan21_root = args.wan21_root.expanduser().resolve()
    args.ckpt_dir = args.ckpt_dir.expanduser().resolve()
    args.torch_cache = args.torch_cache.expanduser().resolve()
    args.sample_ids = [str(value) for value in args.sample_ids]
    args.thresholds = [float(value) for value in args.thresholds]
    validate_args(args)
    prompts = load_selected_prompts(args.sample_ids)
    protocol = make_protocol(args, prompts)
    if not args.dry_run:
        ensure_result_root(args, protocol)

    phases = ["profile", "generate", "metrics", "analyze"] if "all" in args.phases else args.phases
    began = time.monotonic()
    if "profile" in phases:
        run_profile(args)
    if "generate" in phases:
        run_generation(args)
    if "metrics" in phases:
        run_metrics(args)
    if "analyze" in phases:
        run_analysis(args)
    print(
        json.dumps(
            {
                "status": "dry_run" if args.dry_run else "complete",
                "phases": phases,
                "elapsed_seconds": time.monotonic() - began,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
