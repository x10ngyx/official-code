#!/usr/bin/env python3
"""Strict byte and decoded-pixel comparison for two generated videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


VIDEO_METRICS_ROOT = Path(__file__).resolve().parents[3] / "VideoMetrics"
sys.path.insert(0, str(VIDEO_METRICS_ROOT))

from video_metrics.video import decode_video_rgb, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = decode_video_rgb(args.baseline)
    candidate = decode_video_rgb(args.candidate)
    same_shape = baseline.shape == candidate.shape
    pixel_identical = bool(same_shape and np.array_equal(baseline, candidate))
    if same_shape:
        difference = np.abs(baseline - candidate)
        max_abs_difference = float(difference.max(initial=0.0))
        differing_values = int(np.count_nonzero(difference))
        exact_matching_frames = int(
            np.count_nonzero(np.all(baseline == candidate, axis=(1, 2, 3)))
        )
    else:
        max_abs_difference = None
        differing_values = None
        exact_matching_frames = None

    baseline_hash = sha256_file(args.baseline)
    candidate_hash = sha256_file(args.candidate)
    byte_identical = baseline_hash == candidate_hash
    result = {
        "schema_version": 1,
        "status": "pass" if byte_identical and pixel_identical else "fail",
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "baseline_sha256": baseline_hash,
        "candidate_sha256": candidate_hash,
        "byte_identical": byte_identical,
        "baseline_shape_tchw": list(baseline.shape),
        "candidate_shape_tchw": list(candidate.shape),
        "pixel_identical": pixel_identical,
        "exact_matching_frames": exact_matching_frames,
        "max_abs_rgb_difference_0_1": max_abs_difference,
        "differing_rgb_values": differing_values,
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "pass":
        raise SystemExit("baseline and threshold=0 candidate are not strictly identical")


if __name__ == "__main__":
    main()
