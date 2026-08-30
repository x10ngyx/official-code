"""Dynamic-threshold SeaCache controller for locked Wan2.1 T2V collection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .manifest import FORCED_RECOMPUTE_STEPS, NUM_STEPS


BRANCHES = ("cond", "uncond")


@dataclass(frozen=True)
class RandomThresholdConfig:
    threshold_path: tuple[float, ...]
    power_exp: float = 3.0
    power_const: float = 1.0
    eps: float = 1e-16
    norm_mode: str = "mean"

    def __post_init__(self) -> None:
        if len(self.threshold_path) != NUM_STEPS:
            raise ValueError(f"expected {NUM_STEPS} threshold values")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.threshold_path):
            raise ValueError("all threshold values must be finite and positive")
        if not math.isfinite(self.power_exp) or self.power_exp <= 0.0:
            raise ValueError("power_exp must be finite and positive")
        if not math.isfinite(self.power_const) or self.power_const <= 0.0:
            raise ValueError("power_const must be finite and positive")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be finite and positive")
        if self.norm_mode not in {"mean", "peak"}:
            raise ValueError("norm_mode must be mean or peak")


class RandomThresholdSeaCacheController:
    """SeaCache state machine with a dynamic threshold and independent CFG branches.

    The comparison feature is SEA-filtered on every call, including forced
    boundaries. Each branch compares against its own immediately preceding
    filtered feature and owns its accumulator and residual, matching the
    corrected SeaCache4Wan21 controller.
    """

    def __init__(self, config: RandomThresholdConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.previous_features: dict[str, torch.Tensor] = {}
        self.accumulators = {branch: 0.0 for branch in BRANCHES}
        self.scheduler_sigmas: Optional[torch.Tensor] = None
        self.residuals: dict[str, torch.Tensor] = {}
        self.decisions: list[dict[str, Any]] = []
        self._expected_call_index = 0
        self._num_steps: Optional[int] = None
        self._pending_decision: Optional[dict[str, Any]] = None

    def set_scheduler_sigmas(self, sigmas: Optional[torch.Tensor]) -> None:
        self.scheduler_sigmas = sigmas

    def plan_step(
        self,
        *,
        branch: str,
        step_index: int,
        num_steps: int,
        feature: torch.Tensor,
        grid_size: torch.Tensor,
    ) -> bool:
        """Plan one cond/uncond call and record its filtered distance state."""

        self._validate_step(branch, step_index, num_steps, feature, grid_size)
        filtered = self._filter_feature(feature, grid_size, step_index, num_steps)
        requested_threshold = float(self.config.threshold_path[step_index])
        forced = step_index in FORCED_RECOMPUTE_STEPS
        accumulator_before = self.accumulators[branch]
        relative_l1: Optional[float] = None
        accumulator_with_current: Optional[float] = None

        if forced:
            action = "recompute"
            reason = "forced_boundary"
            self.accumulators[branch] = 0.0
        else:
            previous = self.previous_features.get(branch)
            if previous is None:
                raise RuntimeError(
                    f"SeaCache branch {branch!r} has no preceding filtered feature"
                )
            relative_l1 = self._relative_l1(filtered, previous)
            accumulator_with_current = accumulator_before + relative_l1
            self.accumulators[branch] = accumulator_with_current
            if accumulator_with_current < requested_threshold:
                action = "reuse"
                reason = "accumulator_below_requested_threshold"
            else:
                action = "recompute"
                reason = "requested_threshold_reached"
                self.accumulators[branch] = 0.0

        self.previous_features[branch] = filtered.detach().clone()
        decision = {
            "call_index": self._expected_call_index,
            "step_index": step_index,
            "branch": branch,
            "requested_threshold": requested_threshold,
            "action": action,
            "reason": reason,
            # Explicit names are the public data contract. The accumulator_*
            # aliases retain compatibility with earlier internal traces.
            "filtered_relative_l1": relative_l1,
            "accumulated_distance_before": accumulator_before,
            "accumulated_distance_with_current": accumulator_with_current,
            "accumulated_distance_after": self.accumulators[branch],
            "relative_l1": relative_l1,
            "accumulator_before": accumulator_before,
            "accumulator_with_current": accumulator_with_current,
            "accumulator_after": self.accumulators[branch],
            "distance_reference": "previous_step_same_cfg_branch",
            "distance_feature": "sea_filtered_first_block_modulated_input",
            "distance_metric": "relative_l1_mean",
            "stored_feature": "sea_filtered",
            "native_forced_recompute": forced,
            "execution": None,
        }
        self.decisions.append(decision)
        self._pending_decision = decision
        self._expected_call_index += 1
        return action == "reuse"

    def reuse_residual(self, branch: str, step_index: int) -> torch.Tensor:
        self._validate_branch_action(branch, step_index, "reuse")
        residual = self.residuals.get(branch)
        if residual is None:
            raise RuntimeError(f"no cached residual for {branch}")
        assert self._pending_decision is not None
        self._pending_decision["execution"] = "reuse"
        self._pending_decision = None
        return residual.clone()

    def record_recompute(self, branch: str, step_index: int, residual: torch.Tensor) -> None:
        self._validate_branch_action(branch, step_index, "recompute")
        if not torch.is_tensor(residual) or residual.numel() == 0:
            raise ValueError("residual must be a non-empty tensor")
        self.residuals[branch] = residual.detach().clone()
        assert self._pending_decision is not None
        self._pending_decision["execution"] = "recompute"
        self._pending_decision = None

    def summary(self) -> dict[str, Any]:
        if self._pending_decision is not None:
            raise RuntimeError("trace requested before the current branch action finished")
        per_branch = {
            branch: {
                "reuse": sum(
                    row["branch"] == branch and row["action"] == "reuse"
                    for row in self.decisions
                ),
                "recompute": sum(
                    row["branch"] == branch and row["action"] == "recompute"
                    for row in self.decisions
                ),
                "reuse_path": [
                    row["step_index"]
                    for row in self.decisions
                    if row["branch"] == branch and row["action"] == "reuse"
                ],
                "recompute_path": [
                    row["step_index"]
                    for row in self.decisions
                    if row["branch"] == branch and row["action"] == "recompute"
                ],
            }
            for branch in BRANCHES
        }
        return {
            "schema": "ours4wan21_random_threshold_trace_v2",
            "gate_mode": "seacache_aligned_independent_cfg_branches_filtered_boundary_dynamic_threshold",
            "count_unit": "cfg_branch_call",
            "threshold_path": list(self.config.threshold_path),
            "power_exp": self.config.power_exp,
            "power_const": self.config.power_const,
            "eps": self.config.eps,
            "norm_mode": self.config.norm_mode,
            "forced_recompute_steps": sorted(FORCED_RECOMPUTE_STEPS),
            "total_steps": self._num_steps or 0,
            "total_branch_calls": len(self.decisions),
            "reuse": sum(row["action"] == "reuse" for row in self.decisions),
            "recompute": sum(row["action"] == "recompute" for row in self.decisions),
            "per_branch": per_branch,
            "distance_contract": {
                "feature": "sea_filtered_first_block_modulated_input",
                "reference": "previous_step_same_cfg_branch",
                "metric": "relative_l1_mean",
                "accumulation": "sum_since_last_recompute_excluding_forced_boundary",
                "threshold_operand": "accumulated_distance_with_current",
                "reset": "zero_after_recompute",
            },
            "decisions": self.decisions,
        }

    def _validate_step(
        self,
        branch: str,
        step_index: int,
        num_steps: int,
        feature: torch.Tensor,
        grid_size: torch.Tensor,
    ) -> None:
        if branch not in BRANCHES:
            raise ValueError(f"unknown CFG branch {branch}")
        if num_steps != NUM_STEPS or not 0 <= step_index < NUM_STEPS:
            raise ValueError(f"invalid step {step_index}/{num_steps}")
        if self._pending_decision is not None:
            raise RuntimeError("finish the current branch action before planning another")
        if self._num_steps is None:
            self._num_steps = num_steps
        elif self._num_steps != num_steps:
            raise RuntimeError(f"num_steps changed: {self._num_steps}->{num_steps}")
        expected_step = self._expected_call_index // len(BRANCHES)
        expected_branch = BRANCHES[self._expected_call_index % len(BRANCHES)]
        if step_index != expected_step or branch != expected_branch:
            raise RuntimeError(
                "SeaCache calls must follow cond/uncond order; expected "
                f"step {expected_step} {expected_branch}, got step {step_index} {branch}"
            )
        if not torch.is_tensor(feature) or feature.ndim != 3 or feature.numel() == 0:
            raise ValueError("feature must be non-empty [B,L,C]")
        if not bool(torch.isfinite(feature).all()):
            raise RuntimeError("feature contains non-finite values")
        if not torch.is_tensor(grid_size) or grid_size.numel() != 3:
            raise ValueError("grid_size must contain [F,H,W]")

    def _validate_branch_action(self, branch: str, step_index: int, action: str) -> None:
        if branch not in BRANCHES:
            raise ValueError(f"unknown CFG branch {branch}")
        if self._pending_decision is None:
            raise RuntimeError("branch action has no matching step plan")
        decision = self._pending_decision
        if decision["branch"] != branch or decision["step_index"] != step_index:
            raise RuntimeError("branch action does not match its pending plan")
        if decision["action"] != action:
            raise RuntimeError("branch action disagrees with planned gate")

    def _filter_feature(
        self,
        feature: torch.Tensor,
        grid_size: torch.Tensor,
        step_index: int,
        num_steps: int,
    ) -> torch.Tensor:
        f, h, w = [int(value) for value in grid_size.detach().cpu().tolist()]
        if feature.shape[1] != f * h * w:
            raise ValueError("feature token count does not match video grid")
        feature_5d = feature.reshape(feature.shape[0], f, h, w, feature.shape[-1])
        a, b = self._ab_from_flow_scheduler(step_index, num_steps)
        filtered = self._apply_sea_from_ab(feature_5d, a, b, dims=(-2, -3, -4))
        return filtered.reshape(filtered.shape[0], -1, filtered.shape[-1]).detach()

    def _ab_from_flow_scheduler(self, step_index: int, num_steps: int) -> tuple[float, float]:
        if self.scheduler_sigmas is not None:
            sigma = float(self.scheduler_sigmas[step_index].detach().cpu().item())
        else:
            sigma = 1.0 - (step_index + 1) / float(num_steps)
        sigma = max(1e-6, min(1.0 - 1e-6, sigma))
        return 1.0 - sigma, sigma

    def _apply_sea_from_ab(
        self,
        x: torch.Tensor,
        a: float,
        b: float,
        dims: tuple[int, ...],
    ) -> torch.Tensor:
        original_dtype = x.dtype
        x32 = x.contiguous().to(torch.float32)
        spectrum = torch.fft.fftn(x32, dim=dims)
        gain: Optional[torch.Tensor] = None
        for axis in dims:
            frequency = torch.fft.fftfreq(
                x32.shape[axis], device=x32.device, dtype=torch.float32
            ).abs()
            signal_power = self.config.power_const / (
                frequency.pow(self.config.power_exp) + self.config.eps
            )
            axis_gain = (a * signal_power) / (
                a * a * signal_power + b * b + self.config.eps
            )
            shape = [1] * x32.ndim
            shape[axis] = axis_gain.shape[0]
            axis_gain = axis_gain.reshape(shape)
            gain = axis_gain if gain is None else gain * axis_gain
        assert gain is not None
        normalizer = torch.amax(gain) if self.config.norm_mode == "peak" else torch.mean(gain)
        if bool(torch.isfinite(normalizer)) and float(normalizer) > 0.0:
            gain = gain / normalizer
        return torch.fft.ifftn(spectrum * gain, dim=dims).real.to(original_dtype)

    def _relative_l1(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        numerator = (current.float() - previous.float()).abs().mean()
        denominator = previous.float().abs().mean().clamp_min(self.config.eps)
        value = numerator / denominator
        if not bool(torch.isfinite(value)):
            raise RuntimeError("SEA-filtered relative-L1 is non-finite")
        return float(value.detach().item())


__all__ = ["BRANCHES", "RandomThresholdConfig", "RandomThresholdSeaCacheController"]
