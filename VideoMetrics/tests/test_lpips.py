from __future__ import annotations

import os
from pathlib import Path
import unittest

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("TORCH_HOME", str(WORKSPACE_ROOT / "models" / "torch-cache"))

import numpy as np
import torch

from video_metrics.core import LPIPSComputer


class LPIPSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.computer = LPIPSComputer(device="cpu", batch_size=2)

    def test_batched_values_match_locked_per_frame_calls(self) -> None:
        rng = np.random.default_rng(7)
        reference = rng.random((3, 3, 64, 64), dtype=np.float32)
        candidate = rng.random((3, 3, 64, 64), dtype=np.float32)
        actual = self.computer.per_frame(reference, candidate)

        expected = []
        reference_tensor = torch.from_numpy(reference).mul(2.0).sub(1.0)
        candidate_tensor = torch.from_numpy(candidate).mul(2.0).sub(1.0)
        with torch.inference_mode():
            for frame_index in range(reference.shape[0]):
                spatial_map = self.computer.model.forward(
                    reference_tensor[frame_index].unsqueeze(0),
                    candidate_tensor[frame_index].unsqueeze(0),
                )
                expected.append(float(spatial_map.mean()))
        np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-6, atol=1e-7)

    def test_identical_frames_are_zero_with_numerical_tolerance(self) -> None:
        frames = np.zeros((2, 3, 64, 64), dtype=np.float32)
        actual = self.computer.per_frame(frames, frames.copy())
        np.testing.assert_allclose(actual, np.zeros(2), rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
