#!/usr/bin/env python3
"""Independently validate the two single-sample VBench smoke outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np


LOCKED_VBENCH_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
CLIP_VIT_B_32_SHA256 = (
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def stream_temporal_score(path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(path))
    ok, previous = capture.read()
    if not ok:
        raise RuntimeError(f"could not decode {path}")
    total_mae = 0.0
    transitions = 0
    while True:
        ok, current = capture.read()
        if not ok:
            break
        total_mae += float(
            np.mean(
                cv2.absdiff(
                    np.asarray(previous, dtype=np.float32),
                    np.asarray(current, dtype=np.float32),
                )
            )
        )
        transitions += 1
        previous = current
    capture.release()
    if transitions == 0:
        raise RuntimeError(f"video has fewer than two decodable frames: {path}")
    return (255.0 - total_mae / transitions) / 255.0, transitions + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.result_root.resolve()
    output = (args.output or root / "VALIDATION.json").resolve()

    metadata = json.loads((root / "sample_metadata.json").read_text(encoding="utf-8"))
    temporal_json = root / "output/single_sample_smoke_temporal_flickering_eval_results.json"
    weighted_json = root / "weighted_clip/output/single_sample_weighted_smoke_background_consistency_eval_results.json"
    temporal = json.loads(temporal_json.read_text(encoding="utf-8"))["temporal_flickering"]
    weighted = json.loads(weighted_json.read_text(encoding="utf-8"))["background_consistency"]

    temporal_score = float(temporal[0])
    temporal_video = Path(temporal[1][0]["video_path"])
    temporal_recomputed, temporal_frames = stream_temporal_score(temporal_video)

    weighted_score = float(weighted[0])
    weighted_detail = weighted[1][0]
    weighted_recomputed = float(weighted_detail["video_sim"]) / int(
        weighted_detail["cnt_per_video"]
    )
    derived_video = root / "weighted_clip/source_clip_smoke_16f.mp4"
    weight = Path(
        json.loads(
            (root / "weighted_clip/output/evaluation_run_manifest.json").read_text(
                encoding="utf-8"
            )
        )["vbench_cache_dir"]
    ) / "clip_model/ViT-B-32.pt"
    source = Path(metadata["sample_source"])
    commit = subprocess.run(
        ["git", "-C", metadata["vbench_source"], "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    checks = {
        "vbench_commit_locked": commit == LOCKED_VBENCH_COMMIT,
        "sample_sha256_matches_metadata": digest(source) == metadata["sample_sha256"],
        "temporal_score_finite_unit_interval": math.isfinite(temporal_score)
        and 0.0 <= temporal_score <= 1.0,
        "temporal_score_independently_recomputed": math.isclose(
            temporal_score, temporal_recomputed, rel_tol=0.0, abs_tol=1e-9
        ),
        "temporal_video_has_multiple_frames": temporal_frames > 1,
        "clip_weight_sha256_matches_official": digest(weight) == CLIP_VIT_B_32_SHA256,
        "derived_clip_has_16_frames": stream_temporal_score(derived_video)[1] == 16,
        "weighted_score_finite_unit_interval": math.isfinite(weighted_score)
        and 0.0 <= weighted_score <= 1.0,
        "weighted_score_matches_detail_aggregation": math.isclose(
            weighted_score, weighted_recomputed, rel_tol=0.0, abs_tol=1e-12
        ),
        "weighted_transition_count_is_15": int(weighted_detail["cnt_per_video"]) == 15,
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "temporal_flickering": {
            "official_score": temporal_score,
            "independently_recomputed_score": temporal_recomputed,
            "absolute_error": abs(temporal_score - temporal_recomputed),
            "decoded_frames": temporal_frames,
        },
        "background_consistency": {
            "official_score": weighted_score,
            "detail_aggregation_score": weighted_recomputed,
            "transitions": int(weighted_detail["cnt_per_video"]),
        },
        "sample_sha256": digest(source),
        "derived_sample_sha256": digest(derived_video),
        "clip_weight_sha256": digest(weight),
        "vbench_commit": commit,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if report["status"] != "pass":
        raise SystemExit(1)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
