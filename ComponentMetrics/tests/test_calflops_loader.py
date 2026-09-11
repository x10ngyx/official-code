from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


COMPONENT_METRICS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT_METRICS))

from calflops_loader import (  # noqa: E402
    _upsample_flops_compute_compat,
    load_calflops,
)


class CalflopsUpsampleCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.input = torch.zeros((1, 4, 2, 3, 5))

    def test_tuple_scale_factor_counts_output_elements(self) -> None:
        self.assertEqual(
            _upsample_flops_compute_compat(
                self.input, scale_factor=(1, 2, 2)
            ),
            (1 * 4 * 2 * 6 * 10, 0),
        )

    def test_scalar_scale_factor_counts_all_spatial_dimensions(self) -> None:
        self.assertEqual(
            _upsample_flops_compute_compat(self.input, scale_factor=2),
            (1 * 4 * 4 * 6 * 10, 0),
        )

    def test_explicit_size_counts_output_elements(self) -> None:
        self.assertEqual(
            _upsample_flops_compute_compat(self.input, size=(2, 6, 10)),
            (1 * 4 * 2 * 6 * 10, 0),
        )

    def test_locked_loader_installs_and_records_patch(self) -> None:
        _calculate_flops, metadata = load_calflops()
        from calflops import pytorch_ops

        self.assertIs(
            pytorch_ops._upsample_flops_compute,
            _upsample_flops_compute_compat,
        )
        self.assertEqual(metadata["version"], "0.3.2")
        self.assertIn(
            "calflops_0.3.2_tuple_scale_factor_upsample_output_elements",
            metadata["compatibility_patches"],
        )


if __name__ == "__main__":
    unittest.main()
