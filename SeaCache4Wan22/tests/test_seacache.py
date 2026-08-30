from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "runtime"))

from seacache import SeaCacheConfig, SeaCacheController


class SeaCacheControllerTests(unittest.TestCase):
    def test_forced_boundary_stores_filtered_feature(self) -> None:
        controller = SeaCacheController(SeaCacheConfig(threshold=0.2))
        controller.set_scheduler_sigmas(torch.tensor([0.95, 0.5, 0.05]))
        grid = torch.tensor([2, 2, 2])
        raw = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8, 1)
        filtered = controller._filter_feature(raw, grid, 0, 3)
        self.assertFalse(torch.equal(raw, filtered))

        self.assertFalse(
            controller.plan_step(
                stage="high",
                step_index=0,
                num_steps=3,
                feature=raw,
                grid_size=grid,
            )
        )
        self.assertTrue(
            torch.equal(controller.states["high"].previous_feature, filtered)
        )
        controller.record_recompute("high", "cond", 0, raw)
        controller.record_recompute("high", "uncond", 0, raw)

    def test_stage_gate_and_branch_residuals(self) -> None:
        controller = SeaCacheController(SeaCacheConfig(threshold=1e9))
        grid = torch.tensor([2, 2, 2])
        feature = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1) + 1

        self.assertFalse(
            controller.plan_step(
                stage="high",
                step_index=0,
                num_steps=4,
                feature=feature,
                grid_size=grid,
            )
        )
        cond = torch.full((1, 8, 1), 2.0)
        uncond = torch.full((1, 8, 1), 7.0)
        controller.record_recompute("high", "cond", 0, cond)
        controller.record_recompute("high", "uncond", 0, uncond)

        self.assertTrue(
            controller.plan_step(
                stage="high",
                step_index=1,
                num_steps=4,
                feature=feature + 0.01,
                grid_size=grid,
            )
        )
        self.assertTrue(
            controller.plan_step(
                stage="high",
                step_index=1,
                num_steps=4,
                feature=feature * 100,
                grid_size=grid,
            )
        )
        self.assertTrue(torch.equal(controller.reuse_residual("high", "cond", 1), cond))
        self.assertTrue(
            torch.equal(controller.reuse_residual("high", "uncond", 1), uncond)
        )

        controller.clear_stage("high")
        self.assertFalse(
            controller.plan_step(
                stage="low",
                step_index=2,
                num_steps=4,
                feature=feature + 0.02,
                grid_size=grid,
            )
        )
        controller.record_recompute("low", "cond", 2, cond)
        controller.record_recompute("low", "uncond", 2, uncond)
        self.assertFalse(
            controller.plan_step(
                stage="low",
                step_index=3,
                num_steps=4,
                feature=feature + 0.03,
                grid_size=grid,
            )
        )

    def test_fixed_protocol_validation(self) -> None:
        controller = SeaCacheController(SeaCacheConfig(threshold=0.2))
        protocol = {
            "task": "t2v-A14B",
            "size_wh": [832, 480],
            "frame_num": 45,
            "sampling_steps": 50,
            "sample_solver": "dpm++",
            "shift": 12.0,
            "guide_scale_low_high": [3.0, 4.0],
            "boundary": 0.875,
            "param_dtype": "torch.bfloat16",
        }
        controller.validate_runtime_protocol(protocol)
        protocol["sampling_steps"] = 40
        with self.assertRaises(ValueError):
            controller.validate_runtime_protocol(protocol)

    def test_trace_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = Path(temp_dir) / "trace.json"
            controller = SeaCacheController(
                SeaCacheConfig(threshold=0.2, trace_path=str(trace))
            )
            feature = torch.ones((1, 1, 1))
            grid = torch.tensor([1, 1, 1])
            controller.plan_step(
                stage="high",
                step_index=0,
                num_steps=1,
                feature=feature,
                grid_size=grid,
            )
            controller.record_recompute("high", "cond", 0, feature)
            controller.record_recompute("high", "uncond", 0, feature * 2)
            controller.write_trace(extra={"runtime_protocol": {"task": "t2v-A14B"}})
            payload = json.loads(trace.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "seacache4wan22_trace_v1")
            self.assertEqual(payload["total_steps"], 1)
            self.assertEqual(
                payload["decisions"][0]["branches"],
                {"cond": "recompute", "uncond": "recompute"},
            )


if __name__ == "__main__":
    unittest.main()
