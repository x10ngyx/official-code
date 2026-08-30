from __future__ import annotations

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

import torch
from torch import nn


MODULE_PATH = Path(__file__).resolve().parents[1] / "runtime" / "inference_timing.py"


def load_timing_module():
    spec = importlib.util.spec_from_file_location("wan22_inference_timing", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(MODULE_PATH)
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

    def forward(self, value: torch.Tensor, *, reuse: bool = False) -> torch.Tensor:
        if reuse:
            return value
        for block in self.blocks:
            value = block(value)
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
        self.high_noise_model = FakeModel()
        self.low_noise_model = FakeModel()
        self.text_encoder = FakeTextEncoder()
        self.vae = FakeVAE()

    def generate(self) -> torch.Tensor:
        self.text_encoder(torch.zeros(1))
        self.text_encoder(torch.zeros(1))
        value = torch.zeros(1)
        value = self.high_noise_model(value)
        value = self.high_noise_model(value, reuse=True)
        value = self.low_noise_model(value)
        return self.vae.decode(self.low_noise_model(value))


class InferenceTimingTests(unittest.TestCase):
    def test_wan21_style_pipeline_and_block_trace_are_written(self) -> None:
        timing = load_timing_module()
        wan_module = types.SimpleNamespace(WanT2V=FakePipeline)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "timing.json"
            with timing.patch_pipeline_timing(
                wan_module,
                task="t2v-A14B",
                output_path=output,
                implementation="wan22",
            ):
                pipeline = FakePipeline()
            actual = pipeline.generate()
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(torch.equal(actual, torch.tensor([6.0])))
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["component_latency"]["t5"]["call_count"], 2)
        self.assertEqual(payload["component_latency"]["vae_decode"]["call_count"], 1)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["implementation"], "wan22")
        self.assertEqual(payload["model_forward_call_count"], 4)
        self.assertEqual(
            payload["transformer_block_count_by_stage"], {"high": 2, "low": 2}
        )
        self.assertEqual(payload["full_compute_forward_calls"], 3)
        self.assertEqual(payload["reuse_forward_calls"], 1)
        self.assertEqual(
            [row["blocks_executed"] for row in payload["calls"]], [2, 0, 2, 2]
        )
        self.assertEqual(
            [row["model_stage"] for row in payload["calls"]],
            ["high", "high", "low", "low"],
        )
        self.assertEqual(
            [row["cfg_branch"] for row in payload["calls"]],
            ["cond", "uncond", "cond", "uncond"],
        )
        self.assertIsNone(payload["model_forward_cuda_seconds"])

    def test_refuses_to_overwrite_timing_output(self) -> None:
        timing = load_timing_module()
        wan_module = types.SimpleNamespace(WanT2V=FakePipeline)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "timing.json"
            output.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                with timing.patch_pipeline_timing(
                    wan_module,
                    task="t2v-A14B",
                    output_path=output,
                    implementation="wan22",
                ):
                    pass


if __name__ == "__main__":
    unittest.main()
