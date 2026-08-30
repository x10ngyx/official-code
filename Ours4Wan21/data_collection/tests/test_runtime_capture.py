from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


DATA_PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_PROJECT / "src"))

from ours4wan21_data.controller import (  # noqa: E402
    BRANCHES,
    RandomThresholdConfig,
    RandomThresholdSeaCacheController,
)
from ours4wan21_data.runtime import RuntimeCapture  # noqa: E402


class RuntimeCaptureTests(unittest.TestCase):
    def test_candidate_trace_preserves_branch_and_step_distance_views(self) -> None:
        controller = RandomThresholdSeaCacheController(
            RandomThresholdConfig(tuple([0.5] * 50))
        )
        feature = torch.ones((1, 1, 1))
        grid = torch.tensor([1, 1, 1])
        residual = torch.ones((1, 1, 1))
        for step in range(50):
            for branch_index, branch in enumerate(BRANCHES):
                current = feature * (1.0 + 0.01 * step * (branch_index + 1))
                reuse = controller.plan_step(
                    branch=branch,
                    step_index=step,
                    num_steps=50,
                    feature=current,
                    grid_size=grid,
                )
                if reuse:
                    controller.reuse_residual(branch, step)
                else:
                    controller.record_recompute(branch, step, residual)

        capture = RuntimeCapture(mode="candidate")
        capture.reset("trajectory", {"trajectory_id": "trajectory"})
        capture.latents = [torch.full((1, 1, 1, 1), float(step)) for step in range(50)]
        capture.step_metadata = [
            {
                "step_index": step,
                "step_fraction": step / 49.0,
                "timestep": float(50 - step),
                "sigma": 1.0 - step / 50.0,
                "model_stage": "single",
            }
            for step in range(50)
        ]
        capture.trace_payload = controller.summary()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = capture.save_artifacts(root / "trace.json", root / "latents")
            persisted = json.loads((root / "trace.json").read_text(encoding="utf-8"))
            self.assertEqual(payload, persisted)
            self.assertEqual(len(persisted["decisions"]), 100)
            self.assertEqual(len(persisted["step_records"]), 50)
            step = persisted["step_records"][1]
            self.assertEqual(set(step["branch_decisions"]), {"cond", "uncond"})
            self.assertEqual(
                step["cond_filtered_relative_l1"],
                persisted["decisions"][2]["filtered_relative_l1"],
            )
            self.assertEqual(
                step["uncond_accumulated_distance_with_current"],
                persisted["decisions"][3]["accumulated_distance_with_current"],
            )
            self.assertEqual(len(list((root / "latents").iterdir())), 50)

    def test_fixed_seacache_trace_uses_the_same_lossless_branch_capture(self) -> None:
        controller = RandomThresholdSeaCacheController(
            RandomThresholdConfig(tuple([0.3] * 50))
        )
        feature = torch.ones((1, 1, 1))
        grid = torch.tensor([1, 1, 1])
        residual = torch.ones((1, 1, 1))
        for step in range(50):
            for branch_index, branch in enumerate(BRANCHES):
                current = feature * (1.0 + 0.01 * step * (branch_index + 1))
                reuse = controller.plan_step(
                    branch=branch,
                    step_index=step,
                    num_steps=50,
                    feature=current,
                    grid_size=grid,
                )
                if reuse:
                    controller.reuse_residual(branch, step)
                else:
                    controller.record_recompute(branch, step, residual)
        trace = controller.summary()
        trace.update({
            "schema": "ours4wan21_seacache_fixed_threshold_trace_v1",
            "gate_mode": (
                "seacache_aligned_independent_cfg_branches_filtered_boundary_fixed_threshold"
            ),
            "policy_family": "fixed_seacache_threshold",
            "fixed_threshold": 0.3,
        })
        capture = RuntimeCapture(mode="candidate")
        capture.reset(
            "trajectory",
            {"trajectory_id": "trajectory", "policy_family": "fixed_seacache_threshold"},
        )
        capture.latents = [torch.full((1, 1, 1, 1), float(step)) for step in range(50)]
        capture.step_metadata = [
            {
                "step_index": step,
                "step_fraction": step / 49.0,
                "timestep": float(50 - step),
                "sigma": 1.0 - step / 50.0,
                "model_stage": "single",
            }
            for step in range(50)
        ]
        capture.trace_payload = trace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = capture.save_artifacts(root / "trace.json", root / "latents")
            self.assertEqual(payload["schema"], "ours4wan21_seacache_fixed_threshold_trace_v1")
            self.assertEqual(len(payload["decisions"]), 100)
            self.assertEqual(len(payload["step_records"]), 50)


if __name__ == "__main__":
    unittest.main()
