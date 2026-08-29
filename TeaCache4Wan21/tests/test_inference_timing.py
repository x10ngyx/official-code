from __future__ import annotations

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AddBlock(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([AddBlock(), AddBlock()])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return value


class FakePipeline:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = FakeModel()

    def generate(self) -> torch.Tensor:
        first = self.model(torch.zeros(1))
        return self.model(first)


class InferenceTimingTests(unittest.TestCase):
    def test_pipeline_and_block_trace_are_written(self) -> None:
        timing = load_module(
            "inference_timing_test", PROJECT_DIR / "inference_timing.py"
        )
        wan_module = types.SimpleNamespace(WanT2V=FakePipeline)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "timing.json"
            with timing.patch_pipeline_timing(
                wan_module,
                task="t2v-1.3B",
                output_path=output,
                implementation="wan21",
            ):
                pipeline = FakePipeline()
                actual = pipeline.generate()
            self.assertTrue(torch.equal(actual, torch.tensor([4.0])))
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["model_forward_call_count"], 2)
        self.assertEqual(payload["transformer_block_count"], 2)
        self.assertEqual(payload["full_compute_forward_calls"], 2)
        self.assertEqual(payload["reuse_forward_calls"], 0)
        self.assertEqual([row["blocks_executed"] for row in payload["calls"]], [2, 2])
        self.assertIsNone(payload["model_forward_cuda_seconds"])

    def test_flops_summary_uses_actual_block_trace(self) -> None:
        profile = load_module(
            "wan21_flops_profile_test",
            PROJECT_DIR
            / "experiments"
            / "threshold_zero_smoke"
            / "profile_wan21_dit.py",
        )
        timing = {
            "pipeline_init_wall_seconds": 2.0,
            "pipeline_generate_wall_seconds": 3.0,
            "model_forward_cuda_seconds": 0.5,
            "model_forward_call_count": 2,
            "full_compute_forward_calls": 1,
            "reuse_forward_calls": 1,
            "calls": [{"blocks_executed": 2}, {"blocks_executed": 0}],
        }
        actual = profile.summarize_run(
            timing,
            {"process_wall_seconds": 6.0},
            always_on_flops=10.0,
            full_forward_flops=100.0,
            block_count=2,
        )
        self.assertEqual(actual["estimated_dit_flops"], 110.0)
        self.assertEqual(actual["estimated_dit_tflops_per_video"], 110.0 / 1e12)
        self.assertEqual(
            actual["estimated_achieved_tflops_per_second"], 110.0 / 0.5 / 1e12
        )


if __name__ == "__main__":
    unittest.main()
