from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from seacache import BRANCHES, SeaCacheConfig, SeaCacheController


def official_filter(
    feature: torch.Tensor,
    grid: torch.Tensor,
    sigma: float,
    config: SeaCacheConfig,
) -> torch.Tensor:
    """Small independent transcription of official util_seacache.py."""

    f, h, w = [int(value) for value in grid.tolist()]
    x = feature.reshape(feature.shape[0], f, h, w, feature.shape[-1])
    original_dtype = x.dtype
    x32 = x.contiguous().to(torch.float32)
    dims = (-2, -3, -4)
    spectrum = torch.fft.fftn(x32, dim=dims)
    sigma = max(1e-6, min(1.0 - 1e-6, sigma))
    a, b = 1.0 - sigma, sigma
    gain = None
    for axis in dims:
        frequency = torch.fft.fftfreq(
            x32.shape[axis], device=x32.device, dtype=torch.float32
        ).abs()
        signal_power = config.power_const / (
            frequency.pow(config.power_exp) + config.eps
        )
        axis_gain = (a * signal_power) / (
            a * a * signal_power + b * b + config.eps
        )
        shape = [1] * x32.ndim
        shape[axis] = axis_gain.shape[0]
        axis_gain = axis_gain.reshape(shape)
        gain = axis_gain if gain is None else gain * axis_gain
    assert gain is not None
    normalizer = torch.mean(gain)
    if torch.isfinite(normalizer) and normalizer > 0:
        gain = gain / normalizer
    filtered = torch.fft.ifftn(spectrum * gain, dim=dims).real.to(original_dtype)
    return filtered.reshape(feature.shape[0], -1, feature.shape[-1])


class SeaCacheControllerTests(unittest.TestCase):
    def test_independent_branch_gates_and_residuals(self) -> None:
        controller = SeaCacheController(SeaCacheConfig(threshold=0.2))
        grid = torch.tensor([1, 1, 1])
        cond_residual = torch.full((1, 1, 1), 2.0)
        uncond_residual = torch.full((1, 1, 1), 5.0)

        self.assertFalse(
            controller.plan_step(
                branch="cond",
                step_index=0,
                num_steps=3,
                feature=torch.ones((1, 1, 1)),
                grid_size=grid,
            )
        )
        controller.record_recompute("cond", 0, cond_residual)
        self.assertFalse(
            controller.plan_step(
                branch="uncond",
                step_index=0,
                num_steps=3,
                feature=torch.full((1, 1, 1), 10.0),
                grid_size=grid,
            )
        )
        controller.record_recompute("uncond", 0, uncond_residual)

        self.assertTrue(
            controller.plan_step(
                branch="cond",
                step_index=1,
                num_steps=3,
                feature=torch.full((1, 1, 1), 1.01),
                grid_size=grid,
            )
        )
        self.assertTrue(torch.equal(controller.reuse_residual("cond", 1), cond_residual))
        self.assertFalse(
            controller.plan_step(
                branch="uncond",
                step_index=1,
                num_steps=3,
                feature=torch.full((1, 1, 1), 20.0),
                grid_size=grid,
            )
        )
        controller.record_recompute("uncond", 1, torch.full((1, 1, 1), 7.0))

        self.assertEqual(controller.decisions[2]["action"], "reuse")
        self.assertEqual(controller.decisions[3]["action"], "recompute")

    def test_forced_boundary_stores_filtered_feature(self) -> None:
        controller = SeaCacheController(SeaCacheConfig(threshold=0.2))
        controller.set_scheduler_sigmas(torch.tensor([0.95, 0.5, 0.05]))
        grid = torch.tensor([2, 2, 2])
        raw = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8, 1)
        filtered = controller._filter_feature(raw, grid, 0, 3)
        self.assertFalse(torch.equal(raw, filtered))

        self.assertFalse(
            controller.plan_step(
                branch="cond",
                step_index=0,
                num_steps=3,
                feature=raw,
                grid_size=grid,
            )
        )
        self.assertTrue(torch.equal(controller.previous_features["cond"], filtered))
        self.assertEqual(controller.decisions[-1]["stored_feature"], "sea_filtered")

    def test_matches_corrected_state_machine_on_synthetic_trajectories(self) -> None:
        num_steps = 50
        grid = torch.tensor([2, 2, 2])
        sigmas = torch.linspace(0.99, 0.01, num_steps + 1)
        base = torch.arange(1, 17, dtype=torch.float32).reshape(1, 8, 2)

        for seed in range(4):
            for threshold in (0.05, 0.2, 0.5, 1.0):
                for use_ret_steps in (False, True):
                    config = SeaCacheConfig(
                        threshold=threshold, use_ret_steps=use_ret_steps
                    )
                    controller = SeaCacheController(config)
                    controller.set_scheduler_sigmas(sigmas)
                    previous: dict[str, torch.Tensor] = {}
                    accumulators = {branch: 0.0 for branch in BRANCHES}

                    for step_index in range(num_steps):
                        for branch_index, branch in enumerate(BRANCHES):
                            phase = 0.013 * (seed + 1) * (step_index + 1)
                            scale = 1.0 + phase * (1.0 + 0.7 * branch_index)
                            feature = torch.sin(base * 0.11 + phase + branch_index) * scale
                            call_index = step_index * 2 + branch_index
                            ret_calls = 10 if use_ret_steps else 2
                            cutoff_calls = num_steps * 2 if use_ret_steps else num_steps * 2 - 2
                            forced = call_index < ret_calls or call_index >= cutoff_calls

                            expected_previous = official_filter(
                                feature,
                                grid,
                                float(sigmas[step_index]),
                                config,
                            )
                            expected_relative_l1 = None
                            if forced:
                                expected_reuse = False
                                accumulators[branch] = 0.0
                            else:
                                numerator = (expected_previous - previous[branch]).abs().mean()
                                denominator = previous[branch].abs().mean() + config.eps
                                expected_relative_l1 = float(
                                    (numerator / denominator).detach().cpu()
                                )
                                accumulators[branch] += expected_relative_l1
                                expected_reuse = accumulators[branch] < threshold
                                if not expected_reuse:
                                    accumulators[branch] = 0.0
                            previous[branch] = expected_previous.clone()

                            observed_reuse = controller.plan_step(
                                branch=branch,
                                step_index=step_index,
                                num_steps=num_steps,
                                feature=feature,
                                grid_size=grid,
                            )
                            self.assertEqual(observed_reuse, expected_reuse)
                            observed = controller.decisions[-1]
                            if expected_relative_l1 is None:
                                self.assertIsNone(observed["relative_l1"])
                            else:
                                self.assertEqual(
                                    observed["relative_l1"], expected_relative_l1
                                )
                            self.assertEqual(
                                controller.accumulators[branch], accumulators[branch]
                            )
                            self.assertTrue(
                                torch.equal(
                                    controller.previous_features[branch],
                                    expected_previous,
                                )
                            )
                            if observed_reuse:
                                controller.reuse_residual(branch, step_index)
                            else:
                                residual = torch.full_like(
                                    feature, float(step_index + branch_index + 1)
                                )
                                controller.record_recompute(
                                    branch, step_index, residual
                                )

                    self.assertEqual(len(controller.decisions), num_steps * 2)

    def test_trace_is_atomic_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.json"
            controller = SeaCacheController(
                SeaCacheConfig(threshold=0.2, trace_path=str(path))
            )
            grid = torch.tensor([1, 1, 1])
            feature = torch.ones((1, 1, 1))
            for branch, value in (("cond", 1.0), ("uncond", 2.0)):
                controller.plan_step(
                    branch=branch,
                    step_index=0,
                    num_steps=1,
                    feature=feature,
                    grid_size=grid,
                )
                controller.record_recompute(
                    branch, 0, torch.full((1, 1, 1), value)
                )
            controller.write_trace(extra={"task": "t2v-14B"})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "seacache4wan21_trace_v3")
            self.assertEqual(
                payload["gate_mode"],
                "corrected_independent_cfg_branches_filtered_boundary",
            )
            self.assertEqual(payload["task"], "t2v-14B")
            self.assertEqual(payload["total_steps"], 1)
            self.assertEqual(payload["total_branch_calls"], 2)
            self.assertEqual(payload["per_branch"]["cond"]["recompute_path"], [0])
            self.assertEqual(payload["per_branch"]["uncond"]["recompute_path"], [0])

    def test_official_call_order_is_enforced(self) -> None:
        controller = SeaCacheController(SeaCacheConfig(threshold=0.2))
        with self.assertRaises(RuntimeError):
            controller.plan_step(
                branch="uncond",
                step_index=0,
                num_steps=1,
                feature=torch.ones((1, 1, 1)),
                grid_size=torch.tensor([1, 1, 1]),
            )

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SeaCacheConfig(threshold=0)
        with self.assertRaises(ValueError):
            SeaCacheConfig(threshold=0.2, norm_mode="other")


if __name__ == "__main__":
    unittest.main()
