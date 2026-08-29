#!/usr/bin/env python3
"""Evaluate a prompt-named Vbench200 staging folder with official VBench."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--full-info",
        type=Path,
        default=script_dir.parent / "Vbench200" / "VBench200_full_info.json",
    )
    parser.add_argument(
        "--dimension-config",
        type=Path,
        default=script_dir / "dimensions.json",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        help="Dimension subset to run (default: all 16 dimensions).",
    )
    parser.add_argument("--name-prefix", default="vbench200")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--load-ckpt-from-local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep all metric assets under VBENCH_CACHE_DIR (default: enabled).",
    )
    parser.add_argument("--read-frame", action="store_true")
    parser.add_argument(
        "--imaging-quality-preprocessing-mode",
        choices=("longer", "shorter", "shorter_centercrop", "None"),
        default="longer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos_dir = args.videos_dir.resolve()
    output_dir = args.output_dir.resolve()
    full_info = args.full_info.resolve()
    dimension_config_path = args.dimension_config.resolve()
    cache_dir = os.environ.get("VBENCH_CACHE_DIR")
    if not cache_dir:
        raise EnvironmentError(
            "VBENCH_CACHE_DIR must be set explicitly; keep evaluation weights under "
            "the repository's models/VBench directory."
        )
    cache_path = Path(cache_dir).resolve()
    os.environ.setdefault("TORCH_HOME", str(cache_path / "torch"))
    os.environ.setdefault("HF_HOME", str(cache_path / "huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_path / "xdg"))
    if not videos_dir.is_dir() or not any(videos_dir.iterdir()):
        raise NotADirectoryError(f"Missing or empty videos directory: {videos_dir}")
    if not full_info.is_file():
        raise FileNotFoundError(full_info)

    config = json.loads(dimension_config_path.read_text(encoding="utf-8"))
    all_dimensions = config["dimensions"]
    dimensions = args.dimensions or all_dimensions
    unknown = sorted(set(dimensions) - set(all_dimensions))
    if unknown:
        raise ValueError(f"Unknown VBench dimensions: {unknown}")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("Duplicate dimensions were requested")

    try:
        import torch
        from vbench import VBench
    except ImportError as error:
        raise RuntimeError(
            "VBench dependencies are not installed; follow VbenchEvaluation/README.md"
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema_version": 1,
        "dataset": "Vbench200",
        "protocol": "VBench standard mode on a fixed 200-prompt subset",
        "official_full_vbench_score": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "videos_dir": str(videos_dir),
        "output_dir": str(output_dir),
        "full_info": str(full_info),
        "full_info_sha256": sha256(full_info),
        "dimension_config": str(dimension_config_path),
        "dimension_config_sha256": sha256(dimension_config_path),
        "dimensions": dimensions,
        "device": args.device,
        "vbench_cache_dir": str(cache_path),
        "load_ckpt_from_local": args.load_ckpt_from_local,
        "read_frame": args.read_frame,
        "imaging_quality_preprocessing_mode": (
            args.imaging_quality_preprocessing_mode
        ),
    }
    (output_dir / "evaluation_run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    evaluator = VBench(torch.device(args.device), str(full_info), str(output_dir))
    for dimension in dimensions:
        print(f"Evaluating {dimension}", flush=True)
        evaluator.evaluate(
            videos_path=str(videos_dir),
            name=f"{args.name_prefix}_{dimension}",
            dimension_list=[dimension],
            local=args.load_ckpt_from_local,
            read_frame=args.read_frame,
            mode="vbench_standard",
            imaging_quality_preprocessing_mode=(
                args.imaging_quality_preprocessing_mode
            ),
        )


if __name__ == "__main__":
    main()
