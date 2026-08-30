"""Corrected SeaCache controller for locked Wan2.1 T2V inference.

The controller owns only the timestep-level Transformer-block residual cache.
Each CFG branch has the independent feature, accumulator, decision, and
residual state used by the official Wan2.1 SeaCache implementation. Unlike the
official raw-boundary behavior, every stored comparison feature is SEA-filtered
so relative-L1 never mixes raw and filtered representations. It has no
block-cache, CFG-cache, ZEUS, policy, or learned-controller dependency.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch


BRANCHES = ("cond", "uncond")


@dataclass(frozen=True)
class SeaCacheConfig:
    threshold: float
    trace_path: Optional[str] = None
    use_ret_steps: bool = False
    power_exp: float = 3.0
    power_const: float = 1.0
    eps: float = 1e-16
    norm_mode: str = "mean"

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("SeaCache threshold must be finite and positive.")
        if not math.isfinite(self.power_exp) or self.power_exp <= 0:
            raise ValueError("SeaCache power_exp must be finite and positive.")
        if not math.isfinite(self.power_const) or self.power_const <= 0:
            raise ValueError("SeaCache power_const must be finite and positive.")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("SeaCache eps must be finite and positive.")
        if self.norm_mode not in {"mean", "peak"}:
            raise ValueError("SeaCache norm_mode must be 'mean' or 'peak'.")


class SeaCacheController:
    """Corrected independent-branch SeaCache state for one Wan2.1 sample."""

    def __init__(self, config: SeaCacheConfig):
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
        """Return True for reuse using a filtered-boundary branch-local gate."""

        self._validate_step(branch, step_index, num_steps, feature, grid_size)
        filtered = self._filter_feature(feature, grid_size, step_index, num_steps)
        ret_steps, cutoff_steps = self._ret_cutoff_steps(num_steps)
        forced = step_index < ret_steps or step_index >= cutoff_steps
        relative_l1: Optional[float] = None
        accumulator_before = self.accumulators[branch]
        if forced:
            stored_feature = filtered
            action = "recompute"
            reason = "forced_boundary"
            self.accumulators[branch] = 0.0
            feature_state = "sea_filtered"
        else:
            previous = self.previous_features.get(branch)
            if previous is None:
                raise RuntimeError(
                    f"SeaCache branch {branch!r} has no preceding boundary feature."
                )
            relative_l1 = self._relative_l1(filtered, previous)
            self.accumulators[branch] += relative_l1
            stored_feature = filtered
            feature_state = "sea_filtered"
            if self.accumulators[branch] < self.config.threshold:
                action = "reuse"
                reason = "accumulator_below_threshold"
            else:
                action = "recompute"
                reason = "threshold_reached"
                self.accumulators[branch] = 0.0

        self.previous_features[branch] = stored_feature.detach().clone()
        decision = {
            "call_index": self._expected_call_index,
            "step_index": step_index,
            "branch": branch,
            "action": action,
            "reason": reason,
            "relative_l1": relative_l1,
            "accumulator_before": accumulator_before,
            "accumulator_after": self.accumulators[branch],
            "stored_feature": feature_state,
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
            raise RuntimeError(f"SeaCache has no residual for branch {branch!r}.")
        assert self._pending_decision is not None
        self._pending_decision["execution"] = "reuse"
        self._pending_decision = None
        return residual

    def record_recompute(
        self, branch: str, step_index: int, residual: torch.Tensor
    ) -> None:
        self._validate_branch_action(branch, step_index, "recompute")
        if not torch.is_tensor(residual) or residual.numel() == 0:
            raise ValueError("SeaCache residual must be a non-empty tensor.")
        self.residuals[branch] = residual.detach().clone()
        assert self._pending_decision is not None
        self._pending_decision["execution"] = "recompute"
        self._pending_decision = None

    def summary(self) -> dict[str, Any]:
        if self._pending_decision is not None:
            raise RuntimeError("SeaCache trace requested before a branch action finished.")
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
            "schema": "seacache4wan21_trace_v3",
            "gate_mode": "corrected_independent_cfg_branches_filtered_boundary",
            "threshold": self.config.threshold,
            "use_ret_steps": self.config.use_ret_steps,
            "power_exp": self.config.power_exp,
            "power_const": self.config.power_const,
            "eps": self.config.eps,
            "norm_mode": self.config.norm_mode,
            "total_steps": self._num_steps or 0,
            "total_branch_calls": len(self.decisions),
            "reuse": sum(row["action"] == "reuse" for row in self.decisions),
            "recompute": sum(
                row["action"] == "recompute" for row in self.decisions
            ),
            "per_branch": per_branch,
            "decisions": self.decisions,
        }

    def write_trace(
        self,
        path: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Path]:
        target_text = path or self.config.trace_path
        if not target_text:
            return None
        payload = self.summary()
        if extra:
            overlap = sorted(set(payload).intersection(extra))
            if overlap:
                raise ValueError(f"SeaCache trace fields overlap: {overlap}")
            payload.update(dict(extra))
        target = Path(target_text).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return target

    def _validate_step(
        self,
        branch: str,
        step_index: int,
        num_steps: int,
        feature: torch.Tensor,
        grid_size: torch.Tensor,
    ) -> None:
        if branch not in BRANCHES:
            raise ValueError(f"Unknown CFG branch: {branch!r}.")
        if num_steps <= 0 or not 0 <= step_index < num_steps:
            raise ValueError(f"Invalid SeaCache step {step_index}/{num_steps}.")
        if self._pending_decision is not None:
            raise RuntimeError("Finish the current SeaCache branch action before planning another.")
        if self._num_steps is None:
            self._num_steps = num_steps
        elif num_steps != self._num_steps:
            raise RuntimeError(
                f"SeaCache num_steps changed within one sample: {self._num_steps} -> {num_steps}."
            )
        expected_step = self._expected_call_index // len(BRANCHES)
        expected_branch = BRANCHES[self._expected_call_index % len(BRANCHES)]
        if step_index != expected_step or branch != expected_branch:
            raise RuntimeError(
                "SeaCache calls must follow official cond/uncond order; expected "
                f"step {expected_step} {expected_branch}, got step {step_index} {branch}."
            )
        if not torch.is_tensor(feature) or feature.ndim != 3 or feature.numel() == 0:
            raise ValueError("SeaCache feature must be a non-empty [B,L,C] tensor.")
        if not torch.isfinite(feature).all():
            raise RuntimeError("SeaCache feature contains non-finite values.")
        if not torch.is_tensor(grid_size) or grid_size.numel() != 3:
            raise ValueError("SeaCache grid_size must contain [F,H,W].")

    def _validate_branch_action(
        self, branch: str, step_index: int, expected_action: str
    ) -> None:
        if branch not in BRANCHES:
            raise ValueError(f"Unknown CFG branch: {branch!r}.")
        if self._pending_decision is None:
            raise RuntimeError("SeaCache branch action has no matching step plan.")
        decision = self._pending_decision
        if decision["branch"] != branch or decision["step_index"] != step_index:
            raise RuntimeError("SeaCache branch action does not match its pending plan.")
        if decision["action"] != expected_action:
            raise RuntimeError(
                f"Branch action {expected_action!r} disagrees with {decision['action']!r}."
            )

    def _ret_cutoff_steps(self, num_steps: int) -> Tuple[int, int]:
        if self.config.use_ret_steps:
            return min(5, num_steps), num_steps
        return min(1, num_steps), max(0, num_steps - 1)

    def _filter_feature(
        self,
        feature: torch.Tensor,
        grid_size: torch.Tensor,
        step_index: int,
        num_steps: int,
    ) -> torch.Tensor:
        f, h, w = [int(value) for value in grid_size.detach().cpu().tolist()]
        if feature.shape[1] != f * h * w:
            raise ValueError(
                f"SeaCache feature tokens {feature.shape[1]} != grid product {f*h*w}."
            )
        feature_5d = feature.reshape(feature.shape[0], f, h, w, feature.shape[-1])
        a, b = self._ab_from_flow_scheduler(step_index, num_steps)
        filtered = self._apply_sea_from_ab(
            feature_5d, a, b, dims=(-2, -3, -4)
        )
        return filtered.reshape(filtered.shape[0], -1, filtered.shape[-1]).detach()

    def _ab_from_flow_scheduler(
        self, step_index: int, num_steps: int
    ) -> Tuple[float, float]:
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
        dims: Tuple[int, ...],
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
        if torch.isfinite(normalizer) and normalizer > 0:
            gain = gain / normalizer
        return torch.fft.ifftn(spectrum * gain, dim=dims).real.to(original_dtype)

    def _relative_l1(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        numerator = (current - previous).abs().mean()
        denominator = previous.abs().mean() + self.config.eps
        return float((numerator / denominator).detach().cpu())


__all__ = ["SeaCacheConfig", "SeaCacheController"]
