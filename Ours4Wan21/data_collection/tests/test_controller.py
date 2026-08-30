from __future__ import annotations

import importlib.util
import sys
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


def load_locked_seacache_module():
    path = DATA_PROJECT.parents[1] / "SeaCache4Wan21" / "seacache.py"
    spec = importlib.util.spec_from_file_location("locked_seacache4wan21", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ControllerTests(unittest.TestCase):
    def test_filtered_distance_and_all_accumulator_states_are_recorded(self) -> None:
        controller = RandomThresholdSeaCacheController(
            RandomThresholdConfig(tuple([1.5] * 50))
        )
        grid = torch.tensor([1, 1, 1])
        residual = torch.ones((1, 1, 1))
        for step in range(50):
            value = 1.0 if step == 0 else (2.0 if step == 1 else 4.0)
            for branch in BRANCHES:
                reuse = controller.plan_step(
                    branch=branch,
                    step_index=step,
                    num_steps=50,
                    feature=torch.full((1, 1, 1), value),
                    grid_size=grid,
                )
                if reuse:
                    controller.reuse_residual(branch, step)
                else:
                    controller.record_recompute(branch, step, residual)

        step1_cond = controller.decisions[2]
        self.assertAlmostEqual(step1_cond["filtered_relative_l1"], 1.0)
        self.assertAlmostEqual(step1_cond["accumulated_distance_before"], 0.0)
        self.assertAlmostEqual(step1_cond["accumulated_distance_with_current"], 1.0)
        self.assertAlmostEqual(step1_cond["accumulated_distance_after"], 1.0)
        self.assertEqual(step1_cond["action"], "reuse")

        step2_cond = controller.decisions[4]
        self.assertAlmostEqual(step2_cond["filtered_relative_l1"], 1.0)
        self.assertAlmostEqual(step2_cond["accumulated_distance_before"], 1.0)
        self.assertAlmostEqual(step2_cond["accumulated_distance_with_current"], 2.0)
        self.assertAlmostEqual(step2_cond["accumulated_distance_after"], 0.0)
        self.assertEqual(step2_cond["action"], "recompute")
        self.assertEqual(step2_cond["stored_feature"], "sea_filtered")
        self.assertEqual(step2_cond["distance_reference"], "previous_step_same_cfg_branch")

        summary = controller.summary()
        self.assertEqual(summary["schema"], "ours4wan21_random_threshold_trace_v2")
        self.assertEqual(summary["total_steps"], 50)
        self.assertEqual(summary["total_branch_calls"], 100)
        self.assertEqual(summary["count_unit"], "cfg_branch_call")
        self.assertEqual(summary["reuse"] + summary["recompute"], 100)

    def test_independent_cfg_branch_state_and_residuals(self) -> None:
        controller = RandomThresholdSeaCacheController(
            RandomThresholdConfig(tuple([0.2] * 50))
        )
        grid = torch.tensor([1, 1, 1])
        residuals = {
            "cond": torch.full((1, 1, 1), 2.0),
            "uncond": torch.full((1, 1, 1), 5.0),
        }
        for step in range(50):
            features = {
                "cond": torch.full((1, 1, 1), 1.0 + 0.01 * step),
                "uncond": torch.full((1, 1, 1), 10.0 + 10.0 * step),
            }
            for branch in BRANCHES:
                reuse = controller.plan_step(
                    branch=branch,
                    step_index=step,
                    num_steps=50,
                    feature=features[branch],
                    grid_size=grid,
                )
                if reuse:
                    self.assertTrue(
                        torch.equal(controller.reuse_residual(branch, step), residuals[branch])
                    )
                else:
                    controller.record_recompute(branch, step, residuals[branch])
        self.assertEqual(controller.decisions[2]["branch"], "cond")
        self.assertEqual(controller.decisions[3]["branch"], "uncond")
        self.assertNotEqual(
            controller.decisions[2]["filtered_relative_l1"],
            controller.decisions[3]["filtered_relative_l1"],
        )

    def test_matches_locked_seacache_at_constant_threshold(self) -> None:
        seacache = load_locked_seacache_module()
        threshold = 0.2
        observed = RandomThresholdSeaCacheController(
            RandomThresholdConfig(tuple([threshold] * 50))
        )
        reference = seacache.SeaCacheController(seacache.SeaCacheConfig(threshold=threshold))
        sigmas = torch.linspace(0.99, 0.01, 51)
        observed.set_scheduler_sigmas(sigmas)
        reference.set_scheduler_sigmas(sigmas)
        grid = torch.tensor([2, 2, 2])
        base = torch.arange(1, 17, dtype=torch.float32).reshape(1, 8, 2)

        for step in range(50):
            for branch_index, branch in enumerate(BRANCHES):
                phase = 0.017 * (step + 1) * (branch_index + 1)
                feature = torch.sin(base * 0.13 + phase) * (1.0 + phase)
                observed_reuse = observed.plan_step(
                    branch=branch,
                    step_index=step,
                    num_steps=50,
                    feature=feature,
                    grid_size=grid,
                )
                reference_reuse = reference.plan_step(
                    branch=branch,
                    step_index=step,
                    num_steps=50,
                    feature=feature,
                    grid_size=grid,
                )
                self.assertEqual(observed_reuse, reference_reuse)
                ours = observed.decisions[-1]
                sea = reference.decisions[-1]
                for key in ("action", "relative_l1", "accumulator_before", "accumulator_after"):
                    self.assertEqual(ours[key], sea[key])
                residual = torch.full_like(feature, float(step + branch_index + 1))
                if observed_reuse:
                    self.assertTrue(
                        torch.equal(
                            observed.reuse_residual(branch, step),
                            reference.reuse_residual(branch, step),
                        )
                    )
                else:
                    observed.record_recompute(branch, step, residual)
                    reference.record_recompute(branch, step, residual)

    def test_call_order_and_pending_execution_are_enforced(self) -> None:
        controller = RandomThresholdSeaCacheController(
            RandomThresholdConfig(tuple([0.2] * 50))
        )
        feature = torch.ones((1, 1, 1))
        grid = torch.tensor([1, 1, 1])
        with self.assertRaises(RuntimeError):
            controller.plan_step(
                branch="uncond", step_index=0, num_steps=50,
                feature=feature, grid_size=grid,
            )
        controller.plan_step(
            branch="cond", step_index=0, num_steps=50,
            feature=feature, grid_size=grid,
        )
        with self.assertRaises(RuntimeError):
            controller.plan_step(
                branch="uncond", step_index=0, num_steps=50,
                feature=feature, grid_size=grid,
            )


if __name__ == "__main__":
    unittest.main()
