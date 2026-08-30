#!/usr/bin/env python3
"""Persistent-model worker for the fixed VBench8 TeaCache threshold scan."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--enable-warmup", type=int, choices=(0, 1), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wan22_source = Path(os.environ["WAN22_SOURCE"]).resolve()
    checkpoint = Path(os.environ["WAN22_CKPT"]).resolve()
    coefficients = (
        PROJECT_DIR
        / "coefficients"
        / "wan22_t2v_a14b_50step_dpmpp_nonretention.json"
    ).resolve()
    result_root = args.result_root.resolve()

    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker index must be in [0, worker count)")
    if not (wan22_source / ".teacache4wan22_prepared.json").is_file():
        raise FileNotFoundError("WAN22_SOURCE is not a prepared tree")
    if not checkpoint.is_dir() or not coefficients.is_file():
        raise FileNotFoundError("checkpoint or TeaCache coefficients are missing")

    sys.path.insert(0, str(wan22_source))
    sys.path.insert(0, str(SCRIPTS_DIR))
    import torch
    import wan
    from compare_runs import validate_manifest
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
    from wan.inference_timing import _PipelineProfiler
    from wan.teacache import TeaCacheConfig
    from wan.utils.utils import save_video

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )
    torch.cuda.set_device(0)

    prompts = [
        json.loads(line)
        for line in (EXPERIMENT_DIR / "prompts.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assigned = [
        row
        for index, row in enumerate(prompts)
        if index % args.worker_count == args.worker_index
    ]
    if not assigned:
        raise ValueError(f"worker {args.worker_index} has no assigned prompts")
    thresholds = [
        float(value)
        for value in load_json(EXPERIMENT_DIR / "scan_config.json")["thresholds"]
    ]

    runtime_path = (
        result_root / "worker_status" / f"worker_{args.worker_index}_runtime.json"
    )
    started_utc = datetime.now(timezone.utc).isoformat()
    init_started = time.perf_counter()
    pipeline = wan.WanT2V(
        config=WAN_CONFIGS["t2v-A14B"],
        checkpoint_dir=str(checkpoint),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
    )
    torch.cuda.synchronize()
    pipeline_init_wall_seconds = time.perf_counter() - init_started
    runtime_payload = {
        "schema_version": 1,
        "status": "running",
        "started_utc": started_utc,
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "persistent_pipeline": True,
        "pipeline_init_wall_seconds_once": pipeline_init_wall_seconds,
        "assigned_prompt_ids": [row["sample_id"] for row in assigned],
        "completed_run_count": 0,
    }
    atomic_json(runtime_path, runtime_payload)
    logging.info(
        "Persistent WanT2V pipeline ready in %.3f seconds on physical GPU %d",
        pipeline_init_wall_seconds,
        args.physical_gpu,
    )

    def run_one(sample_root: Path, run_id: str, threshold: float, prompt: str) -> None:
        method = "teacache" if threshold > 0 else "none"
        video_path = sample_root / f"{run_id}.mp4"
        trace_path = sample_root / f"{run_id}.teacache.json"
        timing_path = sample_root / f"{run_id}.timing.json"
        log_path = sample_root / f"{run_id}.log"
        manifest_path = sample_root / f"{run_id}.manifest.json"
        artifacts = [video_path, timing_path, log_path, manifest_path]
        if method == "teacache":
            artifacts.append(trace_path)

        if manifest_path.is_file():
            validate_manifest(manifest_path, method)
            logging.info("Validated existing run: %s", manifest_path)
            return
        partial = [str(path) for path in artifacts if path.exists()]
        if partial:
            raise FileExistsError(
                "refusing to overwrite partial run without a valid manifest: "
                + ", ".join(partial)
            )
        sample_root.mkdir(parents=True, exist_ok=True)

        implementation = "teacache" if method == "teacache" else "wan22"
        profiler = _PipelineProfiler(
            pipeline,
            init_wall_seconds=0.0,
            output_path=timing_path,
            implementation=implementation,
        )
        profiler.install()
        cache_config = (
            TeaCacheConfig(
                threshold=threshold,
                coefficients_path=str(coefficients),
                trace_path=str(trace_path),
                use_ret_steps=False,
            )
            if method == "teacache"
            else None
        )
        run_started_utc = datetime.now(timezone.utc).isoformat()
        run_started = time.perf_counter()
        logging.info(
            "Generating sample=%s run=%s threshold=%.3f prompt=%r",
            sample_root.name,
            run_id,
            threshold,
            prompt,
        )
        video = pipeline.generate(
            prompt,
            size=SIZE_CONFIGS["832*480"],
            frame_num=45,
            shift=12.0,
            sample_solver="dpm++",
            sampling_steps=50,
            guide_scale=(3.0, 4.0),
            seed=42,
            offload_model=True,
            teacache_config=cache_config,
        )
        inference_finished = time.perf_counter()
        export_started = time.perf_counter()
        save_video(
            tensor=video[None],
            save_file=str(video_path),
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        torch.cuda.synchronize()
        export_finished = time.perf_counter()
        del video
        gc.collect()
        torch.cuda.empty_cache()

        timing_payload = load_json(timing_path)
        run_log = {
            "schema_version": 1,
            "status": "success",
            "started_utc": run_started_utc,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "sample_id": sample_root.name,
            "run_id": run_id,
            "prompt": prompt,
            "threshold": threshold,
            "implementation": implementation,
            "seed": 42,
            "persistent_pipeline": True,
            "shared_pipeline_init_wall_seconds_once": pipeline_init_wall_seconds,
            "pipeline_generate_wall_seconds": timing_payload[
                "pipeline_generate_wall_seconds"
            ],
            "generation_call_wall_seconds_observed_by_worker": (
                inference_finished - run_started
            ),
            "video_export_wall_seconds": export_finished - export_started,
            "run_wall_seconds_including_export": export_finished - run_started,
            "video_path": str(video_path.resolve()),
            "timing_path": str(timing_path.resolve()),
            "trace_path": str(trace_path.resolve()) if method == "teacache" else None,
        }
        atomic_json(log_path, run_log)

        command = [
            sys.executable,
            str(SCRIPTS_DIR / "write_run_manifest.py"),
            "--output",
            str(manifest_path),
            "--source",
            str(wan22_source),
            "--checkpoint",
            str(checkpoint),
            "--threshold",
            str(threshold),
            "--prompt",
            prompt,
            "--video",
            str(video_path),
            "--timing",
            str(timing_path),
            "--log",
            str(log_path),
        ]
        if method == "teacache":
            command.extend(
                [
                    "--coefficients",
                    str(coefficients),
                    "--trace",
                    str(trace_path),
                ]
            )
        subprocess.run(command, check=True)
        validate_manifest(manifest_path, method)
        runtime_payload["completed_run_count"] += 1
        runtime_payload["last_completed"] = {
            "sample_id": sample_root.name,
            "run_id": run_id,
            "threshold": threshold,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(runtime_path, runtime_payload)
        logging.info("Validated completed run: %s", manifest_path)

    if args.enable_warmup:
        first = assigned[0]
        run_one(
            result_root / "warmups" / f"gpu_{args.physical_gpu}",
            f"warmup_{first['sample_id']}_seed42",
            0.0,
            first["prompt_en"],
        )

    for row in assigned:
        sample_root = result_root / "runs" / row["sample_id"]
        run_one(sample_root, "baseline", 0.0, row["prompt_en"])
        for threshold in thresholds:
            label = f"{threshold:.3f}".replace(".", "p")
            run_one(
                sample_root,
                f"threshold_{label}",
                threshold,
                row["prompt_en"],
            )

    runtime_payload["status"] = "complete"
    runtime_payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(runtime_path, runtime_payload)
    status_path = (
        result_root / "worker_status" / f"worker_{args.worker_index}.json"
    )
    atomic_json(
        status_path,
        {
            "schema_version": 1,
            "status": "complete",
            "completed_utc": runtime_payload["completed_utc"],
            "worker_index": args.worker_index,
            "worker_count": args.worker_count,
            "physical_gpu": args.physical_gpu,
            "completed_prompts": len(assigned),
            "persistent_pipeline": True,
            "pipeline_init_wall_seconds_once": pipeline_init_wall_seconds,
        },
    )
    logging.info(
        "Worker %d complete on physical GPU %d",
        args.worker_index,
        args.physical_gpu,
    )


if __name__ == "__main__":
    main()
