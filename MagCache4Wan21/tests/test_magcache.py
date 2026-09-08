from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from magcache import BRANCHES, MagCacheConfig, MagCacheController, WAN21_T2V13B_MAG_RATIOS


def reference_actions(threshold: float, max_skip_steps: int) -> list[str]:
    residual_ready = {branch: False for branch in BRANCHES}
    accumulated_ratio = {branch: 1.0 for branch in BRANCHES}
    accumulated_error = {branch: 0.0 for branch in BRANCHES}
    accumulated_steps = {branch: 0 for branch in BRANCHES}
    actions = []
    for call_index, ratio in enumerate(WAN21_T2V13B_MAG_RATIOS):
        branch = BRANCHES[call_index % 2]
        reuse = False
        if call_index >= 20:
            accumulated_ratio[branch] *= ratio
            accumulated_steps[branch] += 1
            accumulated_error[branch] += abs(1.0 - accumulated_ratio[branch])
            reuse = (
                residual_ready[branch]
                and accumulated_error[branch] < threshold
                and accumulated_steps[branch] <= max_skip_steps
            )
            if not reuse:
                accumulated_ratio[branch] = 1.0
                accumulated_error[branch] = 0.0
                accumulated_steps[branch] = 0
        if not reuse:
            residual_ready[branch] = True
        actions.append("reuse" if reuse else "recompute")
    return actions


class MagCacheControllerTests(unittest.TestCase):
    def run_controller(self, threshold: float, max_skip_steps: int) -> MagCacheController:
        controller = MagCacheController(
            MagCacheConfig(threshold=threshold, max_skip_steps=max_skip_steps)
        )
        residuals = {
            "cond": torch.full((1, 2, 3), 2.0),
            "uncond": torch.full((1, 2, 3), 5.0),
        }
        for step_index in range(50):
            for branch in BRANCHES:
                if controller.plan_step(branch, step_index, 50):
                    self.assertTrue(torch.equal(controller.reuse_residual(branch, step_index), residuals[branch]))
                else:
                    controller.record_recompute(branch, step_index, residuals[branch])
        return controller

    def test_matches_official_state_machine(self) -> None:
        for threshold in (0.02, 0.06, 0.12, 0.24):
            for max_skip_steps in (1, 2, 4, 6):
                controller = self.run_controller(threshold, max_skip_steps)
                self.assertEqual(
                    [row["action"] for row in controller.decisions],
                    reference_actions(threshold, max_skip_steps),
                )

    def test_cfg_branch_residuals_are_independent(self) -> None:
        controller = self.run_controller(0.12, 4)
        self.assertTrue(torch.equal(controller.residuals["cond"], torch.full((1, 2, 3), 2.0)))
        self.assertTrue(torch.equal(controller.residuals["uncond"], torch.full((1, 2, 3), 5.0)))

    def test_reset_prevents_cross_prompt_state(self) -> None:
        controller = self.run_controller(0.12, 4)
        controller.reset()
        self.assertEqual(controller.decisions, [])
        self.assertEqual(controller.residuals, {})
        self.assertFalse(controller.plan_step("cond", 0, 50))

    def test_trace_is_atomic_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.json"
            controller = self.run_controller(0.12, 4)
            controller.write_trace(str(path), extra={"task": "t2v-1.3B"})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "magcache4wan21_trace_v1")
            self.assertEqual(payload["total_branch_calls"], 100)
            self.assertEqual(payload["reuse"] + payload["recompute"], 100)
            self.assertEqual(payload["task"], "t2v-1.3B")

    def test_invalid_configuration_and_call_order_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            MagCacheConfig(threshold=0, max_skip_steps=4)
        with self.assertRaises(ValueError):
            MagCacheConfig(threshold=0.12, max_skip_steps=0)
        controller = MagCacheController(MagCacheConfig(threshold=0.12, max_skip_steps=4))
        with self.assertRaises(RuntimeError):
            controller.plan_step("uncond", 0, 50)
        with self.assertRaises(ValueError):
            controller.plan_step("cond", 0, 49)


if __name__ == "__main__":
    unittest.main()
