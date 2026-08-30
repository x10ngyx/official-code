"""Full-reference PSNR/SSIM/LPIPS orchestration using shared VideoMetrics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


OFFICIAL_CODE = Path(__file__).resolve().parents[4]
VIDEO_METRICS_ROOT = OFFICIAL_CODE / "VideoMetrics"
sys.path.insert(0, str(VIDEO_METRICS_ROOT))

from video_metrics.core import (  # noqa: E402
    LPIPSComputer,
    LPIPS_METRIC_NAME,
    PROTOCOL_ID,
    PSNR_METRIC_NAME,
    SSIM_METRIC_NAME,
)
from video_metrics.evaluator import (  # noqa: E402
    evaluate_pairs,
    resolve_single_pair,
    write_evaluation,
)


SELECTED_METRICS = ("psnr", "ssim", "lpips")
METRIC_NAMES = (PSNR_METRIC_NAME, SSIM_METRIC_NAME, LPIPS_METRIC_NAME)
COMPACT_SCHEMA = "ours4wan21_full_reference_metrics_v1"
VIDEO_METRICS_LOCK = VIDEO_METRICS_ROOT / "upstream_lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_metrics_model_cache(configured: Path | None = None) -> tuple[Path, Path, str]:
    candidates: list[Path] = []
    if configured is not None:
        candidates.append(configured)
    if os.environ.get("TORCH_HOME"):
        candidates.append(Path(os.environ["TORCH_HOME"]))
    candidates.extend(parent / "models" / "torch-cache" for parent in OFFICIAL_CODE.parents)
    lock = json.loads(VIDEO_METRICS_LOCK.read_text(encoding="utf-8"))
    alexnet = lock["model_weights"]["alexnet"]
    relative = Path(alexnet["relative_to_torch_home"])
    expected_sha = str(alexnet["sha256"])
    checked: list[str] = []
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        weight = root / relative
        checked.append(str(weight))
        if weight.is_file() and weight.stat().st_size > 0:
            observed_sha = _sha256(weight)
            if observed_sha != expected_sha:
                raise ValueError(
                    f"AlexNet weight checksum mismatch: expected {expected_sha}, got {observed_sha}: {weight}"
                )
            return root, weight, observed_sha
    raise FileNotFoundError(
        "LPIPS AlexNet weight is missing; set METRICS_MODEL_CACHE/TORCH_HOME to a "
        f"torch cache containing {relative}. Checked: {checked}"
    )


def metric_paths(candidate_root: Path) -> dict[str, Path]:
    root = candidate_root / "video_metrics"
    return {
        "metrics_root": root,
        "metrics_per_frame": root / "per_frame.csv",
        "metrics_per_video": root / "per_video.csv",
        "metrics_summary": root / "summary.json",
        "metrics": root / "metrics.json",
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing/non-numeric metric field {key}: {row}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric field {key}: {value}")
    return value


def normalize_metric_artifacts(candidate_root: Path, *, expected_frames: int = 81) -> dict[str, Any]:
    paths = metric_paths(candidate_root)
    for key in ("metrics_per_frame", "metrics_per_video", "metrics_summary"):
        path = paths[key]
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    summary = json.loads(paths["metrics_summary"].read_text(encoding="utf-8"))
    if summary.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("VideoMetrics protocol mismatch")
    if summary.get("selected_metrics") != list(SELECTED_METRICS):
        raise ValueError("VideoMetrics must contain PSNR, SSIM, and LPIPS in frozen order")
    if int(summary.get("video_count", 0)) != 1 or int(summary.get("frame_count_total", 0)) != expected_frames:
        raise ValueError("VideoMetrics summary video/frame count mismatch")

    video_rows = _read_csv(paths["metrics_per_video"])
    frame_rows = _read_csv(paths["metrics_per_frame"])
    if len(video_rows) != 1 or len(frame_rows) != expected_frames:
        raise ValueError("VideoMetrics per-video/per-frame row count mismatch")
    video = video_rows[0]
    if int(video.get("frames", 0)) != expected_frames:
        raise ValueError("VideoMetrics per-video frame count mismatch")
    if [int(row.get("frame_index", -1)) for row in frame_rows] != list(range(expected_frames)):
        raise ValueError("VideoMetrics frame indices must be contiguous")
    for frame in frame_rows:
        for metric_name in METRIC_NAMES:
            _finite(frame, metric_name)

    metrics: dict[str, dict[str, float]] = {}
    for metric_name in METRIC_NAMES:
        metrics[metric_name] = {
            statistic: _finite(video, f"{metric_name}_{statistic}")
            for statistic in ("mean", "std_population", "min", "max")
        }
        statistics = metrics[metric_name]
        if statistics["std_population"] < 0.0 or not (
            statistics["min"] <= statistics["mean"] <= statistics["max"]
        ):
            raise ValueError(f"invalid metric statistics for {metric_name}: {statistics}")
    evaluation_elapsed = float(summary["evaluation_elapsed_seconds"])
    if not math.isfinite(evaluation_elapsed) or evaluation_elapsed < 0.0:
        raise ValueError("VideoMetrics evaluation elapsed time is invalid")
    payload = {
        "schema": COMPACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "selected_metrics": list(SELECTED_METRICS),
        "metric_names": {
            "psnr": PSNR_METRIC_NAME,
            "ssim": SSIM_METRIC_NAME,
            "lpips": LPIPS_METRIC_NAME,
        },
        "video_id": video.get("video_id"),
        "reference": video.get("reference"),
        "candidate": video.get("candidate"),
        "reference_sha256": video.get("reference_sha256"),
        "candidate_sha256": video.get("candidate_sha256"),
        "frames": expected_frames,
        "height": int(video.get("height", 0)),
        "width": int(video.get("width", 0)),
        "exact_matching_frames": int(video.get("exact_matching_frames", 0)),
        "psnr_capped_frames": int(video.get("psnr_capped_frames", 0)),
        "decode_seconds": _finite(video, "decode_seconds"),
        "metric_seconds": _finite(video, "metric_seconds"),
        "evaluation_elapsed_seconds": evaluation_elapsed,
        "lpips_device": summary.get("lpips_device"),
        "lpips_batch_size": int(summary.get("lpips_batch_size", 0)),
        "model_weights": summary.get("upstream_lock", {}).get("model_weights"),
        "torch_home": summary.get("software", {}).get("torch_home"),
        "metrics": metrics,
        "artifacts": {
            key: str(paths[key].resolve())
            for key in ("metrics_per_frame", "metrics_per_video", "metrics_summary")
        },
        "timing_scope": "evaluation_only_excluded_from_inference_speedup",
    }
    if payload["height"] <= 0 or payload["width"] <= 0 or payload["lpips_batch_size"] <= 0:
        raise ValueError("VideoMetrics geometry or LPIPS batch size is invalid")
    if payload["decode_seconds"] < 0.0 or payload["metric_seconds"] < 0.0:
        raise ValueError("VideoMetrics decode/metric time is invalid")
    if not isinstance(payload["model_weights"], dict) or not payload["torch_home"]:
        raise ValueError("VideoMetrics LPIPS model provenance is missing")
    return payload


def load_metric_artifacts(candidate_root: Path, *, expected_frames: int = 81) -> dict[str, Any]:
    paths = metric_paths(candidate_root)
    if not paths["metrics"].is_file() or paths["metrics"].stat().st_size == 0:
        raise FileNotFoundError(paths["metrics"])
    observed = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    expected = normalize_metric_artifacts(candidate_root, expected_frames=expected_frames)
    if observed != expected:
        raise ValueError(f"compact metric artifact disagrees with shared VideoMetrics outputs: {paths['metrics']}")
    return observed


class FullReferenceMetricEvaluator:
    """Reuse one LPIPS model for every candidate handled by a worker."""

    def __init__(self, *, device: str, lpips_batch_size: int, model_cache: Path | None) -> None:
        resolved, weight, weight_sha = resolve_metrics_model_cache(model_cache)
        os.environ["TORCH_HOME"] = str(resolved)
        self.model_cache = resolved
        self.alexnet_weight = weight
        self.alexnet_weight_sha256 = weight_sha
        self.lpips = LPIPSComputer(device=device, batch_size=lpips_batch_size)

    def evaluate(
        self,
        *,
        reference: Path,
        candidate: Path,
        video_id: str,
        candidate_root: Path,
    ) -> dict[str, Any]:
        paths = metric_paths(candidate_root)
        if paths["metrics_root"].exists():
            raise FileExistsError(f"refusing to overwrite VideoMetrics artifacts: {paths['metrics_root']}")
        pairs = resolve_single_pair(reference, candidate, video_id)
        frame_rows, video_rows, summary = evaluate_pairs(
            pairs,
            SELECTED_METRICS,
            expected_frames=81,
            lpips_computer=self.lpips,
        )
        write_evaluation(paths["metrics_root"], frame_rows, video_rows, summary)
        payload = normalize_metric_artifacts(candidate_root)
        temporary = paths["metrics"].with_name(paths["metrics"].name + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, paths["metrics"])
        return payload
