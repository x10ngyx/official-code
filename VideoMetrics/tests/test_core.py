from __future__ import annotations

import math
import os
import unittest
from pathlib import Path
from unittest.mock import patch

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import cv2
import numpy as np

from video_metrics.core import psnr_per_frame, ssim_per_frame, validate_pair
from video_metrics.evaluator import VideoPair, evaluate_pairs


def upstream_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = np.mean((reference / 1.0 - candidate / 1.0) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(float(mse)))


def upstream_ssim_plane(reference: np.ndarray, candidate: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu_reference = cv2.filter2D(reference, -1, window)[5:-5, 5:-5]
    mu_candidate = cv2.filter2D(candidate, -1, window)[5:-5, 5:-5]
    mu_reference_sq = mu_reference**2
    mu_candidate_sq = mu_candidate**2
    mu_cross = mu_reference * mu_candidate
    sigma_reference_sq = cv2.filter2D(reference**2, -1, window)[5:-5, 5:-5] - mu_reference_sq
    sigma_candidate_sq = cv2.filter2D(candidate**2, -1, window)[5:-5, 5:-5] - mu_candidate_sq
    sigma_cross = cv2.filter2D(reference * candidate, -1, window)[5:-5, 5:-5] - mu_cross
    ssim_map = ((2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)) / (
        (mu_reference_sq + mu_candidate_sq + c1)
        * (sigma_reference_sq + sigma_candidate_sq + c2)
    )
    return float(ssim_map.mean())


class CoreMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.reference = (
            rng.integers(0, 256, size=(3, 3, 32, 40), dtype=np.uint8).astype(np.float32)
            / np.float32(255.0)
        )
        self.candidate = (
            rng.integers(0, 256, size=(3, 3, 32, 40), dtype=np.uint8).astype(np.float32)
            / np.float32(255.0)
        )

    def test_psnr_matches_locked_upstream_definition(self) -> None:
        actual = psnr_per_frame(self.reference, self.candidate)
        expected = np.asarray(
            [
                upstream_psnr(ref_frame, candidate_frame)
                for ref_frame, candidate_frame in zip(
                    self.reference, self.candidate, strict=True
                )
            ]
        )
        np.testing.assert_array_equal(actual, expected)

    def test_psnr_exact_frames_are_capped_at_100(self) -> None:
        actual = psnr_per_frame(self.reference, self.reference.copy())
        np.testing.assert_array_equal(actual, np.full(3, 100.0))

    def test_known_psnr_offset(self) -> None:
        reference = np.zeros((1, 3, 16, 16), dtype=np.float32)
        candidate = np.full_like(reference, 0.1)
        self.assertAlmostEqual(float(psnr_per_frame(reference, candidate)[0]), 20.0, places=5)

    def test_ssim_matches_locked_upstream_definition(self) -> None:
        actual = ssim_per_frame(self.reference, self.candidate)
        expected = []
        for ref_frame, candidate_frame in zip(self.reference, self.candidate, strict=True):
            expected.append(
                float(
                    np.asarray(
                        [
                            upstream_ssim_plane(ref_frame[channel], candidate_frame[channel])
                            for channel in range(3)
                        ]
                    ).mean()
                )
            )
        np.testing.assert_array_equal(actual, np.asarray(expected))

    def test_ssim_exact_frames_equal_one(self) -> None:
        actual = ssim_per_frame(self.reference, self.reference.copy())
        np.testing.assert_allclose(actual, np.ones(3), rtol=0.0, atol=1e-12)

    def test_shape_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical"):
            validate_pair(self.reference, self.candidate[:, :, :, :-1])

    def test_evaluator_reuses_supplied_lpips_computer(self) -> None:
        class DummyLPIPSComputer:
            device = "cpu"
            batch_size = 4

            def __init__(self) -> None:
                self.calls = 0

            def per_frame(
                self, reference: np.ndarray, candidate: np.ndarray
            ) -> np.ndarray:
                self.calls += 1
                return np.zeros(reference.shape[0], dtype=np.float64)

        video = np.zeros((2, 3, 16, 16), dtype=np.float32)
        lpips_computer = DummyLPIPSComputer()
        pair = VideoPair("sample", Path("reference.mp4"), Path("candidate.mp4"))
        with (
            patch(
                "video_metrics.evaluator.decode_video_rgb",
                side_effect=(video, video.copy()),
            ),
            patch("video_metrics.evaluator.sha256_file", return_value="0" * 64),
        ):
            frame_rows, video_rows, summary = evaluate_pairs(
                [pair],
                expected_frames=2,
                lpips_computer=lpips_computer,  # type: ignore[arg-type]
            )

        self.assertEqual(lpips_computer.calls, 1)
        self.assertEqual(len(frame_rows), 2)
        self.assertEqual(len(video_rows), 1)
        self.assertEqual(summary["lpips_device"], "cpu")
        self.assertEqual(summary["lpips_batch_size"], 4)


if __name__ == "__main__":
    unittest.main()
