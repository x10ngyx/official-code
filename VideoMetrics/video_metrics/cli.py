"""Command-line interface for full-reference video metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .evaluator import (
    evaluate_pairs,
    resolve_directory_pairs,
    resolve_single_pair,
    write_evaluation,
)


def _infer_workspace_torch_home() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        models_root = parent / "models"
        if models_root.is_dir():
            return models_root / "torch-cache"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate aligned videos with frozen RGB PSNR, SSIM, and LPIPS definitions."
    )
    parser.add_argument("--reference", type=Path, help="single reference video")
    parser.add_argument("--candidate", type=Path, help="single candidate video")
    parser.add_argument("--video-id", help="ID for single-pair output; defaults to candidate stem")
    parser.add_argument("--reference-dir", type=Path, help="directory of reference videos")
    parser.add_argument("--candidate-dir", type=Path, help="directory of candidate videos")
    parser.add_argument("--extension", default=".mp4", help="directory-mode extension (default: .mp4)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=("psnr", "ssim", "lpips"),
        default=("psnr", "ssim", "lpips"),
    )
    parser.add_argument("--device", default="auto", help="LPIPS device: auto, cpu, cuda, cuda:N")
    parser.add_argument("--lpips-batch-size", type=int, default=8)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--model-cache", type=Path, help="TORCH_HOME for AlexNet weights")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_mode(args: argparse.Namespace):
    single_values = (args.reference, args.candidate)
    directory_values = (args.reference_dir, args.candidate_dir)
    if all(single_values) and not any(directory_values):
        return resolve_single_pair(args.reference, args.candidate, args.video_id)
    if all(directory_values) and not any(single_values) and args.video_id is None:
        return resolve_directory_pairs(args.reference_dir, args.candidate_dir, args.extension)
    raise SystemExit(
        "select exactly one complete mode: --reference/--candidate or "
        "--reference-dir/--candidate-dir"
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_cache = args.model_cache or _infer_workspace_torch_home()
    if model_cache is not None:
        model_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(model_cache.resolve()))

    pairs = _resolve_mode(args)
    frame_rows, video_rows, summary = evaluate_pairs(
        pairs,
        args.metrics,
        device=args.device,
        lpips_batch_size=args.lpips_batch_size,
        expected_frames=args.expected_frames,
    )
    paths = write_evaluation(
        args.output_dir,
        frame_rows,
        video_rows,
        summary,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "protocol_id": summary["protocol_id"],
                "video_count": summary["video_count"],
                "frame_count_total": summary["frame_count_total"],
                "metrics": summary["metrics"],
                "outputs": {key: str(path.resolve()) for key, path in paths.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
    )
