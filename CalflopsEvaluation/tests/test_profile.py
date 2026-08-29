from __future__ import annotations

import os
import unittest

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import torch
from torch import nn

from calflops_eval.models import ManualComponent, ProfileCase
from calflops_eval.profiling import profile_items


class TinyLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ProfileTest(unittest.TestCase):
    def test_calflops_and_manual_components(self) -> None:
        actual = profile_items(
            [
                ProfileCase(
                    name="linear",
                    model=TinyLinear(),
                    args=(torch.zeros(2, 4),),
                ),
                ManualComponent(
                    name="custom",
                    flops=123,
                    formula="known test count",
                ),
            ]
        )
        self.assertGreater(actual["components"]["linear"]["flops"], 0)
        self.assertEqual(actual["components"]["linear"]["source"], "calflops")
        self.assertEqual(actual["components"]["custom"]["flops"], 123)
        self.assertEqual(actual["components"]["custom"]["source"], "manual_formula")


if __name__ == "__main__":
    unittest.main()
