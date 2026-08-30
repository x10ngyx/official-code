from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


DATA_PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_PROJECT / "src"))

from ours4wan21_data.publisher import load_completion, numeric_summary, write_snapshot  # noqa: E402
from ours4wan21_data.manifest import SCHEMA_CANDIDATE_COMPLETE  # noqa: E402
from ours4wan21_data.metrics import metric_paths, normalize_metric_artifacts  # noqa: E402


def write_synthetic_metrics(candidate_root: Path) -> None:
    paths = metric_paths(candidate_root)
    paths["metrics_root"].mkdir()
    metric_names = ("psnr_rgb_db", "ssim_rgb", "lpips_alex_v0_1_spatial")
    with paths["metrics_per_frame"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("video_id", "frame_index", *metric_names))
        writer.writeheader()
        for index in range(81):
            writer.writerow({
                "video_id": "sample__random_00",
                "frame_index": index,
                "psnr_rgb_db": 30.0,
                "ssim_rgb": 0.95,
                "lpips_alex_v0_1_spatial": 0.05,
            })
    video = {
        "video_id": "sample__random_00",
        "reference": "/tmp/reference.mp4",
        "candidate": "/tmp/candidate.mp4",
        "reference_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
        "frames": 81,
        "height": 480,
        "width": 832,
        "decode_seconds": 1.0,
        "metric_seconds": 2.0,
        "exact_matching_frames": 0,
        "psnr_capped_frames": 0,
    }
    for name, value in zip(metric_names, (30.0, 0.95, 0.05), strict=True):
        for statistic in ("mean", "min", "max"):
            video[f"{name}_{statistic}"] = value
        video[f"{name}_std_population"] = 0.0
    with paths["metrics_per_video"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(video))
        writer.writeheader()
        writer.writerow(video)
    paths["metrics_summary"].write_text(json.dumps({
        "protocol_id": "rgb_full_reference_v1",
        "selected_metrics": ["psnr", "ssim", "lpips"],
        "video_count": 1,
        "frame_count_total": 81,
        "evaluation_elapsed_seconds": 3.0,
        "lpips_device": "cpu",
        "lpips_batch_size": 8,
        "upstream_lock": {"model_weights": {"alexnet": {"sha256": "c" * 64}}},
        "software": {"torch_home": "/tmp/torch-cache"},
    }), encoding="utf-8")
    paths["metrics"].write_text(
        json.dumps(normalize_metric_artifacts(candidate_root)), encoding="utf-8"
    )


class PublicationTests(unittest.TestCase):
    def test_completion_identity_and_step_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            row = {"release_index": 0, "trajectory_id": "sample__random_00", "shard_index": 0}
            bundle = parent / "shards/shard_00/candidates/sample__random_00"
            (bundle / "latents").mkdir(parents=True)
            for name in ("candidate.mp4", "timing.json", "performance.json", "trace.json", "ffprobe.json"):
                (bundle / name).write_text("x", encoding="utf-8")
            write_synthetic_metrics(bundle)
            for index in range(50):
                (bundle / "latents" / f"step_{index:03d}_input.pt").write_text("x", encoding="utf-8")
            marker = parent / "completed/sample__random_00.json"
            marker.parent.mkdir()
            marker.write_text(json.dumps({
                "schema": SCHEMA_CANDIDATE_COMPLETE,
                "release_index": 0,
                "trajectory_id": row["trajectory_id"],
                "trajectory_row": {"trajectory_id": row["trajectory_id"]},
                "step_rows": [{"step_index": index} for index in range(50)],
                "branch_rows": [
                    {
                        "call_index": call_index,
                        "step_index": call_index // 2,
                        "branch": ("cond", "uncond")[call_index % 2],
                        "filtered_relative_l1": None if call_index < 2 else 0.01,
                        "accumulated_distance_with_current": None if call_index < 2 else 0.01,
                    }
                    for call_index in range(100)
                ],
            }), encoding="utf-8")
            self.assertIsNotNone(load_completion(parent, row))
            manifest = parent / "manifest.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            snapshot = parent / "snapshot"
            summary = write_snapshot([row], parent, snapshot, manifest)
            self.assertEqual(summary["published_candidate_count"], 1)
            self.assertEqual(summary["published_step_count"], 50)
            self.assertEqual(summary["published_branch_transition_count"], 100)
            self.assertEqual(len(summary["tables"]), 6)
            self.assertTrue((snapshot / "tables/branch_transitions.jsonl").is_file())
            self.assertTrue((snapshot / "tables/branch_transitions.csv").is_file())
            for table in summary["tables"].values():
                self.assertFalse(Path(table["path"]).is_absolute())
                self.assertTrue((snapshot / table["path"]).is_file())
            payload = json.loads(marker.read_text(encoding="utf-8"))
            payload["step_rows"][-1]["step_index"] = 48
            marker.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "step order"):
                load_completion(parent, row)

    def test_numeric_summary_ignores_nonfinite(self) -> None:
        self.assertEqual(
            numeric_summary([1.0, 2.0, float("nan"), None]),
            {"count": 2, "min": 1.0, "mean": 1.5, "max": 2.0},
        )


if __name__ == "__main__":
    unittest.main()
