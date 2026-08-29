"""Compatibility output for the repository-wide work/compute_psnr.py contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import PROTOCOL_ID, PSNR_CAP_DB, PSNR_MSE_CAP_THRESHOLD, psnr_per_frame, validate_pair
from .video import decode_video_rgb, sha256_file


def compute_psnr_contract(reference: str | Path, candidate: str | Path) -> dict[str, object]:
    reference_path = Path(reference).resolve(strict=True)
    candidate_path = Path(candidate).resolve(strict=True)
    reference_video = decode_video_rgb(reference_path)
    candidate_video = decode_video_rgb(candidate_path)
    reference_video, candidate_video = validate_pair(reference_video, candidate_video)
    values = psnr_per_frame(reference_video, candidate_video)
    exact_matching_frames = int(
        np.count_nonzero(np.all(reference_video == candidate_video, axis=(1, 2, 3)))
    )
    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "method": "rgb_framewise_psnr_v1",
        "protocol_id": PROTOCOL_ID,
        "color_space": "RGB",
        "data_range": 1.0,
        "psnr_mse_cap_threshold": PSNR_MSE_CAP_THRESHOLD,
        "perfect_psnr_threshold": PSNR_CAP_DB,
        "frames": int(values.size),
        "decoded_frames_total": int(values.size),
        "excluded_perfect_frames": 0,
        "psnr_capped_frames": int(np.count_nonzero(values >= PSNR_CAP_DB)),
        "exact_matching_frames": exact_matching_frames,
        "mean_psnr": float(values.mean()),
        "min_psnr": float(values.min()),
        "max_psnr": float(values.max()),
        "per_frame_psnr": [float(value) for value in values],
        "reference_sha256": sha256_file(reference_path),
        "candidate_sha256": sha256_file(candidate_path),
    }


def write_psnr_contract(
    reference: str | Path,
    candidate: str | Path,
    output: str | Path,
) -> dict[str, object]:
    result = compute_psnr_contract(reference, candidate)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result
