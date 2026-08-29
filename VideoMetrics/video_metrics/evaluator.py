"""Dataset orchestration for frozen full-reference video metrics."""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .core import (
    LPIPS_METRIC_NAME,
    PROTOCOL_ID,
    PSNR_CAP_DB,
    PSNR_METRIC_NAME,
    SSIM_METRIC_NAME,
    LPIPSComputer,
    psnr_per_frame,
    ssim_per_frame,
    validate_pair,
)
from .video import decode_video_rgb, sha256_file


METRIC_NAMES = {
    "psnr": PSNR_METRIC_NAME,
    "ssim": SSIM_METRIC_NAME,
    "lpips": LPIPS_METRIC_NAME,
}


@dataclass(frozen=True)
class VideoPair:
    video_id: str
    reference: Path
    candidate: Path


def resolve_single_pair(
    reference: str | Path,
    candidate: str | Path,
    video_id: str | None = None,
) -> list[VideoPair]:
    reference_path = Path(reference).resolve(strict=True)
    candidate_path = Path(candidate).resolve(strict=True)
    resolved_id = video_id or candidate_path.stem
    return [VideoPair(resolved_id, reference_path, candidate_path)]


def resolve_directory_pairs(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    extension: str = ".mp4",
) -> list[VideoPair]:
    reference_root = Path(reference_dir).resolve(strict=True)
    candidate_root = Path(candidate_dir).resolve(strict=True)
    if not reference_root.is_dir() or not candidate_root.is_dir():
        raise ValueError("reference_dir and candidate_dir must both be directories")
    if not extension.startswith("."):
        extension = f".{extension}"

    reference_files = {path.name: path for path in reference_root.glob(f"*{extension}")}
    candidate_files = {path.name: path for path in candidate_root.glob(f"*{extension}")}
    if not candidate_files:
        raise ValueError(f"no {extension} candidate videos found in {candidate_root}")
    if set(reference_files) != set(candidate_files):
        missing_references = sorted(set(candidate_files) - set(reference_files))
        missing_candidates = sorted(set(reference_files) - set(candidate_files))
        raise ValueError(
            "reference/candidate filename sets differ; "
            f"missing references={missing_references[:10]}, "
            f"missing candidates={missing_candidates[:10]}"
        )

    return [
        VideoPair(Path(name).stem, reference_files[name], candidate_files[name])
        for name in sorted(candidate_files)
    ]


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _software_manifest() -> dict[str, object]:
    manifest: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            package: _package_version(package)
            for package in (
                "numpy",
                "opencv-python",
                "imageio",
                "imageio-ffmpeg",
                "torch",
                "torchvision",
                "lpips",
            )
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "torch_home": os.environ.get("TORCH_HOME"),
    }
    try:
        import imageio_ffmpeg

        manifest["ffmpeg"] = {
            "executable": imageio_ffmpeg.get_ffmpeg_exe(),
            "version": imageio_ffmpeg.get_ffmpeg_version(),
        }
    except Exception as error:  # pragma: no cover - diagnostic only
        manifest["ffmpeg"] = {"error": repr(error)}
    return manifest


def _upstream_manifest() -> dict[str, object]:
    lock_path = Path(__file__).resolve().parents[1] / "upstream_lock.json"
    return json.loads(lock_path.read_text(encoding="utf-8"))


def _describe(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot describe an empty metric sequence")
    return {
        "mean": float(statistics.fmean(values)),
        "std_population": float(np.asarray(values, dtype=np.float64).std()),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def evaluate_pairs(
    pairs: Sequence[VideoPair],
    metrics: Sequence[str] = ("psnr", "ssim", "lpips"),
    *,
    device: str = "auto",
    lpips_batch_size: int = 8,
    expected_frames: int | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if not pairs:
        raise ValueError("at least one video pair is required")
    selected_metrics = tuple(dict.fromkeys(metrics))
    unknown = sorted(set(selected_metrics) - set(METRIC_NAMES))
    if unknown:
        raise ValueError(f"unknown metrics: {unknown}")
    if not selected_metrics:
        raise ValueError("at least one metric must be selected")
    if expected_frames is not None and expected_frames < 1:
        raise ValueError("expected_frames must be at least 1")

    lpips_computer = (
        LPIPSComputer(device=device, batch_size=lpips_batch_size)
        if "lpips" in selected_metrics
        else None
    )
    frame_rows: list[dict[str, object]] = []
    video_rows: list[dict[str, object]] = []
    began = time.monotonic()

    seen_ids: set[str] = set()
    for pair in pairs:
        if pair.video_id in seen_ids:
            raise ValueError(f"duplicate video_id: {pair.video_id}")
        seen_ids.add(pair.video_id)

        decode_began = time.monotonic()
        reference = decode_video_rgb(pair.reference)
        candidate = decode_video_rgb(pair.candidate)
        reference, candidate = validate_pair(reference, candidate)
        decode_seconds = time.monotonic() - decode_began
        if expected_frames is not None and reference.shape[0] != expected_frames:
            raise ValueError(
                f"{pair.video_id} decoded {reference.shape[0]} frames; expected {expected_frames}"
            )

        metric_began = time.monotonic()
        per_metric: dict[str, np.ndarray] = {}
        if "psnr" in selected_metrics:
            per_metric[PSNR_METRIC_NAME] = psnr_per_frame(reference, candidate)
        if "ssim" in selected_metrics:
            per_metric[SSIM_METRIC_NAME] = ssim_per_frame(reference, candidate)
        if "lpips" in selected_metrics:
            assert lpips_computer is not None
            per_metric[LPIPS_METRIC_NAME] = lpips_computer.per_frame(reference, candidate)
        metric_seconds = time.monotonic() - metric_began

        frame_count = int(reference.shape[0])
        for frame_index in range(frame_count):
            row: dict[str, object] = {
                "video_id": pair.video_id,
                "frame_index": frame_index,
            }
            for metric_name, values in per_metric.items():
                row[metric_name] = float(values[frame_index])
            frame_rows.append(row)

        video_row: dict[str, object] = {
            "video_id": pair.video_id,
            "reference": str(pair.reference),
            "candidate": str(pair.candidate),
            "reference_sha256": sha256_file(pair.reference),
            "candidate_sha256": sha256_file(pair.candidate),
            "frames": frame_count,
            "height": int(reference.shape[2]),
            "width": int(reference.shape[3]),
            "decode_seconds": decode_seconds,
            "metric_seconds": metric_seconds,
            "exact_matching_frames": int(
                np.count_nonzero(np.all(reference == candidate, axis=(1, 2, 3)))
            ),
        }
        for metric_name, values in per_metric.items():
            description = _describe([float(value) for value in values])
            for statistic, value in description.items():
                video_row[f"{metric_name}_{statistic}"] = value
        if PSNR_METRIC_NAME in per_metric:
            video_row["psnr_capped_frames"] = int(
                np.count_nonzero(per_metric[PSNR_METRIC_NAME] >= PSNR_CAP_DB)
            )
        video_rows.append(video_row)

    aggregate_metrics: dict[str, dict[str, float]] = {}
    for metric in selected_metrics:
        metric_name = METRIC_NAMES[metric]
        aggregate_metrics[metric_name] = _describe(
            [float(row[f"{metric_name}_mean"]) for row in video_rows]
        )

    summary: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "upstream_lock": _upstream_manifest(),
        "selected_metrics": list(selected_metrics),
        "aggregation": "per-frame mean within each video, then equal-weight mean across videos",
        "video_count": len(video_rows),
        "frame_count_total": sum(int(row["frames"]) for row in video_rows),
        "metrics": aggregate_metrics,
        "software": _software_manifest(),
        "evaluation_elapsed_seconds": time.monotonic() - began,
    }
    if lpips_computer is not None:
        summary["lpips_device"] = str(lpips_computer.device)
        summary["lpips_batch_size"] = lpips_computer.batch_size
    return frame_rows, video_rows, summary


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def write_evaluation(
    output_dir: str | Path,
    frame_rows: Sequence[dict[str, object]],
    video_rows: Sequence[dict[str, object]],
    summary: dict[str, object],
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_frame": output_root / "per_frame.csv",
        "per_video": output_root / "per_video.csv",
        "summary": output_root / "summary.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing evaluation files: {existing}")
    _write_csv(paths["per_frame"], frame_rows)
    _write_csv(paths["per_video"], video_rows)
    paths["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths
