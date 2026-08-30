#!/usr/bin/env python3
"""Evaluate arbitrary prompt/video pairs with VBench custom-input dimensions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CUSTOM_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
    "human_action",
    "temporal_style",
    "overall_consistency",
)

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


def load_prompt_map(path: Path | None, videos_dir: Path) -> dict[str, str]:
    if path is None:
        return {}
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("prompt map must be a non-empty filename-to-prompt object")
    video_names = {
        item.name for item in videos_dir.iterdir() if item.suffix.lower() in {".mp4", ".gif"}
    }
    if set(payload) != video_names:
        raise ValueError("prompt map keys must exactly match staged video filenames")
    if any(not isinstance(value, str) or not value.strip() for value in payload.values()):
        raise ValueError("prompt map values must be non-empty strings")
    return {str(key): str(value) for key, value in payload.items()}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-map", type=Path)
    parser.add_argument(
        "--full-info",
        type=Path,
        default=script_dir.parent / "Vbench200" / "VBench200_full_info.json",
        help="Constructor provenance only; custom_input builds metadata from inputs.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--name-prefix", default="vbench_custom")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos_dir = args.videos_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    full_info = args.full_info.expanduser().resolve(strict=True)
    videos = [
        item for item in videos_dir.iterdir() if item.suffix.lower() in {".mp4", ".gif"}
    ]
    if not videos:
        raise ValueError(f"no supported videos in {videos_dir}")
    prompt_map = load_prompt_map(
        args.prompt_map.expanduser().resolve(strict=True) if args.prompt_map else None,
        videos_dir,
    )
    cache = os.environ.get("VBENCH_CACHE_DIR")
    if not cache:
        raise EnvironmentError("VBENCH_CACHE_DIR must be set explicitly")
    cache_path = Path(cache).expanduser().resolve()
    os.environ.setdefault("TORCH_HOME", str(cache_path / "torch"))
    os.environ.setdefault("HF_HOME", str(cache_path / "huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_path / "xdg"))

    try:
        import torch
        from vbench import VBench
    except ImportError as error:
        raise RuntimeError("VBench dependencies are not installed") from error

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "mode": "custom_input",
        "score_scope": "official VBench custom-input dimension implementations",
        "official_full_vbench_score": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "videos_dir": str(videos_dir),
        "video_count": len(videos),
        "prompt_map": str(args.prompt_map.resolve()) if args.prompt_map else None,
        "prompt_map_sha256": sha256(args.prompt_map.resolve()) if args.prompt_map else None,
        "dimensions": list(CUSTOM_DIMENSIONS),
        "device": args.device,
        "vbench_cache_dir": str(cache_path),
    }
    (output_dir / "evaluation_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    evaluator = VBench(torch.device(args.device), str(full_info), str(output_dir))
    for dimension in CUSTOM_DIMENSIONS:
        evaluator.evaluate(
            videos_path=str(videos_dir),
            name=f"{args.name_prefix}_{dimension}",
            prompt_list=prompt_map,
            dimension_list=[dimension],
            local=True,
            mode="custom_input",
            imaging_quality_preprocessing_mode="longer",
        )


if __name__ == "__main__":
    main()
