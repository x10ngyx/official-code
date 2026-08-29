from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import imageio.v2 as imageio
import numpy as np

from video_metrics.evaluator import evaluate_pairs, resolve_single_pair

warnings.filterwarnings("ignore", category=ResourceWarning)


class VideoIntegrationTest(unittest.TestCase):
    def _make_identical_pair(self, root: Path) -> tuple[Path, Path]:
        reference = root / "reference.mp4"
        candidate = root / "candidate.mp4"
        rng = np.random.default_rng(11)
        frames = rng.integers(0, 256, size=(3, 64, 64, 3), dtype=np.uint8)
        writer = imageio.get_writer(
            reference,
            format="ffmpeg",
            mode="I",
            fps=8,
            codec="libx264",
            macro_block_size=None,
            ffmpeg_log_level="error",
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        shutil.copyfile(reference, candidate)
        return reference, candidate

    def test_pair_evaluation_and_repository_psnr_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, candidate = self._make_identical_pair(root)
            pairs = resolve_single_pair(reference, candidate, "synthetic")
            frame_rows, video_rows, summary = evaluate_pairs(
                pairs,
                metrics=("psnr", "ssim"),
                expected_frames=3,
            )
            self.assertEqual(len(frame_rows), 3)
            self.assertEqual(video_rows[0]["psnr_rgb_db_mean"], 100.0)
            self.assertAlmostEqual(float(video_rows[0]["ssim_rgb_mean"]), 1.0, places=12)
            self.assertEqual(summary["video_count"], 1)

            output = root / "psnr.json"
            repository_entrypoint = Path(__file__).resolve().parents[3] / "compute_psnr.py"
            subprocess.run(
                [
                    sys.executable,
                    str(repository_entrypoint),
                    "--reference",
                    str(reference),
                    "--candidate",
                    str(candidate),
                    "--output",
                    str(output),
                    "--protocol",
                    "rgb_full_reference_v1",
                ],
                check=True,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["protocol_id"], "rgb_full_reference_v1")
            self.assertEqual(payload["method"], "rgb_framewise_psnr_v1")
            self.assertEqual(payload["mean_psnr"], 100.0)
            self.assertEqual(payload["excluded_perfect_frames"], 0)

            legacy_output = root / "legacy_psnr.json"
            subprocess.run(
                [
                    sys.executable,
                    str(repository_entrypoint),
                    "--reference",
                    str(reference),
                    "--candidate",
                    str(candidate),
                    "--output",
                    str(legacy_output),
                ],
                check=True,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            legacy_payload = json.loads(legacy_output.read_text(encoding="utf-8"))
            self.assertEqual(
                legacy_payload["method"],
                "ffmpeg_psnr_filter_psnr_avg_yuv_weighted",
            )

            evaluation_dir = root / "evaluation"
            source_entrypoint = Path(__file__).resolve().parents[1] / "evaluate.py"
            subprocess.run(
                [
                    sys.executable,
                    str(source_entrypoint),
                    "--reference",
                    str(reference),
                    "--candidate",
                    str(candidate),
                    "--output-dir",
                    str(evaluation_dir),
                    "--device",
                    "cpu",
                    "--expected-frames",
                    "3",
                ],
                check=True,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            summary = json.loads((evaluation_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["protocol_id"], "rgb_full_reference_v1")
            self.assertEqual(summary["selected_metrics"], ["psnr", "ssim", "lpips"])
            self.assertEqual(summary["metrics"]["psnr_rgb_db"]["mean"], 100.0)
            self.assertAlmostEqual(summary["metrics"]["ssim_rgb"]["mean"], 1.0, places=12)
            self.assertAlmostEqual(
                summary["metrics"]["lpips_alex_v0_1_spatial"]["mean"],
                0.0,
                places=12,
            )
            self.assertTrue((evaluation_dir / "per_frame.csv").is_file())
            self.assertTrue((evaluation_dir / "per_video.csv").is_file())


if __name__ == "__main__":
    unittest.main()
