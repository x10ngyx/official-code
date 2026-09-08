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
        self.blocks = nn.ModuleList([AddBlock(), AddBlock(), AddBlock()])
        self.reuse_next = False

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.blocks[0](value)
        if not self.reuse_next:
            for block in self.blocks[1:]:
                value = block(value)
        self.reuse_next = not self.reuse_next
        return value


class FakeTextEncoder:
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value


class FakeVAE:
    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return value


class FakePipeline:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = FakeModel()
        self.text_encoder = FakeTextEncoder()
        self.vae = FakeVAE()

    def generate(self) -> torch.Tensor:
        self.text_encoder(torch.zeros(1))
        self.text_encoder(torch.zeros(1))
        first = self.model(torch.zeros(1))
        return self.vae.decode(self.model(first))


class InferenceTimingTests(unittest.TestCase):
    def test_dicache_counts_probe_cost_on_reuse(self) -> None:
        timing = load_module(
            "dicache_inference_timing_test", PROJECT_DIR / "inference_timing.py"
        )
        wan_module = types.SimpleNamespace(WanT2V=FakePipeline)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "timing.json"
            with timing.patch_pipeline_timing(
                wan_module,
                task="t2v-1.3B",
                output_path=output,
                implementation="dicache",
            ):
                pipeline = FakePipeline()
                actual = pipeline.generate()
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(torch.equal(actual, torch.tensor([4.0])))
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["model_forward_call_count"], 2)
        self.assertEqual([row["blocks_executed"] for row in payload["calls"]], [3, 1])
        self.assertEqual(payload["full_compute_forward_calls"], 1)
        self.assertEqual(payload["probe_reuse_forward_calls"], 1)
        self.assertEqual(payload["total_probe_blocks_executed"], 2)
        self.assertEqual(payload["total_deep_blocks_executed"], 2)

    def test_invalid_partial_deep_execution_fails_closed(self) -> None:
        class PartialModel(FakeModel):
            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.blocks[1](self.blocks[0](value))

        class PartialPipeline(FakePipeline):
            def __init__(self) -> None:
                super().__init__()
                self.model = PartialModel()

            def generate(self) -> torch.Tensor:
                return self.model(torch.zeros(1))

        timing = load_module(
            "dicache_inference_timing_partial_test",
            PROJECT_DIR / "inference_timing.py",
        )
        wan_module = types.SimpleNamespace(WanT2V=PartialPipeline)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "timing.json"
            with timing.patch_pipeline_timing(
                wan_module,
                task="t2v-1.3B",
                output_path=output,
                implementation="dicache",
            ):
                pipeline = PartialPipeline()
                with self.assertRaisesRegex(ValueError, "invalid block count"):
                    pipeline.generate()


if __name__ == "__main__":
    unittest.main()
