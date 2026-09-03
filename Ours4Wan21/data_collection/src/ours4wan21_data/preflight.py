"""Fail-closed package and remote-runtime preflight checks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .paths import RESULT_BASE_ENV, require_result_path
from .source_lock import file_sha256, validate_wan21_source


DATA_PROJECT = Path(__file__).resolve().parents[2]
OURS_PROJECT = DATA_PROJECT.parent
OFFICIAL_CODE = OURS_PROJECT.parent
PROMPT_POOL = DATA_PROJECT / "resources/prompts/openvidhd_balanced_5000.upstream.jsonl"
PROMPT_SHA256 = "fb5d5d73f86b84d10d8e55154b789ac8549c74e90f33c1d4d2a02d67a5cde3e5"
PROMPT_ROWS = 5000
THREAD_KEYS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_package() -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    digest = file_sha256(PROMPT_POOL) if PROMPT_POOL.is_file() else None
    if digest:
        with PROMPT_POOL.open(encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
    else:
        rows = 0
    if digest != PROMPT_SHA256:
        errors.append(f"bundled prompt SHA256 mismatch: expected {PROMPT_SHA256}, got {digest}")
    if rows != PROMPT_ROWS:
        errors.append(f"bundled prompt row count mismatch: expected {PROMPT_ROWS}, got {rows}")
    required_files = (
        DATA_PROJECT / "configs/speed_threshold_mapping.pending.json",
        DATA_PROJECT / "configs/seacache_thresholds.wan22_v1.json",
        DATA_PROJECT / "experiments/random_threshold_collection_v1/launch_4gpu.sh",
        DATA_PROJECT / "experiments/seacache_threshold_collection_v1/launch_4gpu.sh",
        OFFICIAL_CODE / "VideoMetrics/evaluate.py",
        OFFICIAL_CODE / "VideoMetrics/video_metrics/evaluator.py",
        OFFICIAL_CODE / "VideoMetrics/video_metrics/core.py",
        OFFICIAL_CODE / "CalflopsEvaluation/calflops_eval/manual_ops.py",
        OURS_PROJECT / "upstream_lock.json",
        OURS_PROJECT / "NOTICE.md",
        OFFICIAL_CODE / "TeaCache4Wan21/LICENSE.upstream.txt",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        errors.append(f"package files missing: {missing}")
    seacache_config_path = DATA_PROJECT / "configs/seacache_thresholds.wan22_v1.json"
    seacache_thresholds = None
    if seacache_config_path.is_file():
        try:
            from .manifest import load_seacache_threshold_config

            seacache_thresholds = load_seacache_threshold_config(seacache_config_path)[
                "thresholds"
            ]
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"fixed SeaCache threshold config is invalid: {exc}")
    versions = {
        name: package_version(name)
        for name in (
            "numpy", "imageio", "imageio-ffmpeg", "opencv-python", "lpips",
            "torch", "torchvision", "tqdm",
        )
    }
    for name, version in versions.items():
        if version is None:
            errors.append(f"required Python distribution is missing: {name}")
    if versions.get("lpips") != "0.1.4":
        errors.append(f"full-reference metrics require lpips==0.1.4; observed {versions.get('lpips')!r}")
    executables = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    for name, path in executables.items():
        if path is None:
            errors.append(f"required executable is missing from PATH: {name}")
    return {
        "data_project": str(DATA_PROJECT),
        "official_code": str(OFFICIAL_CODE),
        "prompt_pool": str(PROMPT_POOL),
        "prompt_sha256": digest,
        "prompt_rows": rows,
        "seacache_threshold_config": str(seacache_config_path),
        "seacache_thresholds": seacache_thresholds,
        "python_distributions": versions,
        "executables": executables,
    }, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("package", "plan", "profile", "baselines", "candidates", "finalize"),
        default="package",
    )
    parser.add_argument("--exp-base", type=Path)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--wan21-root", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--flops-profile", type=Path)
    parser.add_argument("--runnable-manifest", type=Path)
    parser.add_argument("--metrics-model-cache", type=Path)
    args = parser.parse_args()
    if args.exp_base is not None:
        os.environ[RESULT_BASE_ENV] = str(args.exp_base)

    payload, errors = check_package()
    payload.update({
        "schema": "ours4wan21_remote_preflight_v1",
        "phase": args.phase,
        "result_base": os.environ.get(RESULT_BASE_ENV, "/all/yiran07-disk3/huteng_data/exp"),
        "thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS},
    })
    if args.archive_root is not None:
        try:
            payload["archive_root"] = str(require_result_path(args.archive_root))
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

    gpu_phases = {"profile", "baselines", "candidates"}
    if args.phase in gpu_phases:
        invalid_threads = {key: os.environ.get(key) for key in THREAD_KEYS if os.environ.get(key) != "1"}
        if invalid_threads:
            errors.append(f"BLAS thread limits must all equal one: {invalid_threads}")
        if args.wan21_root is None:
            errors.append("--wan21-root is required for this phase")
        else:
            try:
                payload["wan21_source_sha256"] = validate_wan21_source(args.wan21_root)
            except (OSError, ValueError) as exc:
                errors.append(f"locked Wan2.1 source check failed: {exc}")
        if args.checkpoint_dir is None:
            errors.append("--checkpoint-dir is required for this phase")
        elif not args.checkpoint_dir.is_dir() or "1.3B" not in args.checkpoint_dir.name:
            errors.append(f"Wan2.1-T2V-1.3B checkpoint directory is invalid: {args.checkpoint_dir}")
        try:
            import torch

            payload["cuda_available"] = bool(torch.cuda.is_available())
            payload["cuda_device_count"] = int(torch.cuda.device_count())
            if not torch.cuda.is_available():
                errors.append("CUDA is unavailable in the project wan2.2 Python environment")
        except ImportError:
            errors.append("torch cannot be imported")

    if args.phase == "profile":
        version = package_version("calflops")
        payload["calflops_version"] = version
        if version != "0.3.2":
            errors.append(f"profile requires calflops==0.3.2; observed {version!r}")
    if args.phase in {"baselines", "candidates", "finalize"}:
        if args.flops_profile is None or not args.flops_profile.is_file():
            errors.append(f"Calflops profile is missing: {args.flops_profile}")
    if args.phase in {"candidates", "finalize"}:
        if args.runnable_manifest is None or not args.runnable_manifest.is_file():
            errors.append(f"calibrated runnable manifest is missing: {args.runnable_manifest}")
    if args.phase == "candidates":
        try:
            from .metrics import resolve_metrics_model_cache

            cache_root, weight, weight_sha = resolve_metrics_model_cache(args.metrics_model_cache)
            payload["full_reference_metrics"] = {
                "selected_metrics": ["psnr", "ssim", "lpips"],
                "protocol_id": "rgb_full_reference_v1",
                "torch_home": str(cache_root),
                "alexnet_weight": str(weight),
                "alexnet_weight_sha256": weight_sha,
            }
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"LPIPS model-cache check failed: {exc}")

    payload["status"] = "ok" if not errors else "failed"
    payload["errors"] = errors
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
