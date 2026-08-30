"""Clean stage-aware SeaCache controller for pinned Wan2.2 T2V-A14B."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch


STAGES = ("high", "low")
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


@dataclass
class _StageState:
    previous_feature: Optional[torch.Tensor] = None
    accumulator: float = 0.0
    residuals: dict[str, torch.Tensor] = field(default_factory=dict)


class SeaCacheController:
    """One shared gate per model stage and one residual per stage/CFG branch."""

    def __init__(self, config: SeaCacheConfig):
        self.config = config
        self.scheduler_sigmas: Optional[torch.Tensor] = None
        self.states: dict[str, _StageState] = {}
        self.decisions: list[dict[str, Any]] = []
        self._current_step: Optional[int] = None
        self._current_stage: Optional[str] = None

    def set_scheduler_sigmas(self, sigmas: Optional[torch.Tensor]) -> None:
        self.scheduler_sigmas = sigmas

    def validate_runtime_protocol(self, protocol: Mapping[str, Any]) -> None:
        expected = {
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
        mismatches = {
            key: {"expected": value, "observed": protocol.get(key)}
            for key, value in expected.items()
            if protocol.get(key) != value
        }
        if mismatches:
            raise ValueError(f"SeaCache4Wan22 protocol mismatch: {mismatches}")

    def plan_step(
        self,
        *,
        stage: str,
        step_index: int,
        num_steps: int,
        feature: torch.Tensor,
        grid_size: torch.Tensor,
    ) -> bool:
        self._validate_step(stage, step_index, num_steps, feature, grid_size)
        if self._current_step == step_index:
            if self._current_stage != stage:
                raise RuntimeError("A denoising step cannot use two Wan2.2 model stages.")
            return self.decisions[-1]["action"] == "reuse"
        if self._current_step is not None and step_index != self._current_step + 1:
            raise RuntimeError(
                f"SeaCache steps must be consecutive: {self._current_step} -> {step_index}."
            )
        if self.decisions and set(self.decisions[-1]["branches"]) != set(BRANCHES):
            raise RuntimeError("Both CFG branches must finish before the next SeaCache step.")

        state = self.states.setdefault(stage, _StageState())
        filtered = self._filter_feature(feature, grid_size, step_index, num_steps)
        ret_steps, cutoff_steps = self._ret_cutoff_steps(num_steps)
        forced = (
            step_index < ret_steps
            or step_index >= cutoff_steps
            or state.previous_feature is None
            or set(state.residuals) != set(BRANCHES)
        )
        relative_l1: Optional[float] = None
        accumulator_before = state.accumulator
        if forced:
            action = "recompute"
            reason = "boundary_stage_start_or_cold_start"
            state.accumulator = 0.0
        else:
            relative_l1 = self._relative_l1(filtered, state.previous_feature)
            state.accumulator += relative_l1
            if state.accumulator < self.config.threshold:
                action = "reuse"
                reason = "accumulator_below_threshold"
            else:
                action = "recompute"
                reason = "threshold_reached"
                state.accumulator = 0.0

        state.previous_feature = filtered.detach().clone()
        self._current_step = step_index
        self._current_stage = stage
        self.decisions.append(
            {
                "step_index": step_index,
                "stage": stage,
                "action": action,
                "reason": reason,
                "relative_l1": relative_l1,
                "accumulator_before": accumulator_before,
                "accumulator_after": state.accumulator,
                "branches": {},
            }
        )
        return action == "reuse"

    def reuse_residual(self, stage: str, branch: str, step_index: int) -> torch.Tensor:
        self._validate_branch_action(stage, branch, step_index, "reuse")
        residual = self.states[stage].residuals.get(branch)
        if residual is None:
            raise RuntimeError(f"SeaCache has no residual for {(stage, branch)}.")
        self.decisions[-1]["branches"][branch] = "reuse"
        return residual.clone()

    def record_recompute(
        self,
        stage: str,
        branch: str,
        step_index: int,
        residual: torch.Tensor,
    ) -> None:
        self._validate_branch_action(stage, branch, step_index, "recompute")
        if not torch.is_tensor(residual) or residual.numel() == 0:
            raise ValueError("SeaCache residual must be a non-empty tensor.")
        self.states[stage].residuals[branch] = residual.detach().clone()
        self.decisions[-1]["branches"][branch] = "recompute"

    def clear_stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown Wan2.2 stage: {stage!r}.")
        self.states.pop(stage, None)

    def summary(self) -> dict[str, Any]:
        per_stage = {}
        for stage in STAGES:
            rows = [row for row in self.decisions if row["stage"] == stage]
            per_stage[stage] = {
                "steps": len(rows),
                "reuse": sum(row["action"] == "reuse" for row in rows),
                "recompute": sum(row["action"] == "recompute" for row in rows),
                "reuse_path": [
                    row["step_index"] for row in rows if row["action"] == "reuse"
                ],
                "recompute_path": [
                    row["step_index"]
                    for row in rows
                    if row["action"] == "recompute"
                ],
            }
        return {
            "schema": "seacache4wan22_trace_v1",
            "threshold": self.config.threshold,
            "use_ret_steps": self.config.use_ret_steps,
            "power_exp": self.config.power_exp,
            "power_const": self.config.power_const,
            "eps": self.config.eps,
            "norm_mode": self.config.norm_mode,
            "total_steps": len(self.decisions),
            "reuse": sum(row["action"] == "reuse" for row in self.decisions),
            "recompute": sum(
                row["action"] == "recompute" for row in self.decisions
            ),
            "per_stage": per_stage,
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
        stage: str,
        step_index: int,
        num_steps: int,
        feature: torch.Tensor,
        grid_size: torch.Tensor,
    ) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown Wan2.2 stage: {stage!r}.")
        if num_steps <= 0 or not 0 <= step_index < num_steps:
            raise ValueError(f"Invalid SeaCache step {step_index}/{num_steps}.")
        if not torch.is_tensor(feature) or feature.ndim != 3 or feature.numel() == 0:
            raise ValueError("SeaCache feature must be a non-empty [B,L,C] tensor.")
        if not torch.isfinite(feature).all():
            raise RuntimeError("SeaCache feature contains non-finite values.")
        if not torch.is_tensor(grid_size) or grid_size.numel() != 3:
            raise ValueError("SeaCache grid_size must contain [F,H,W].")

    def _validate_branch_action(
        self,
        stage: str,
        branch: str,
        step_index: int,
        expected_action: str,
    ) -> None:
        if branch not in BRANCHES:
            raise ValueError(f"Unknown CFG branch: {branch!r}.")
        if (
            not self.decisions
            or self._current_step != step_index
            or self._current_stage != stage
        ):
            raise RuntimeError("SeaCache branch action has no matching step plan.")
        decision = self.decisions[-1]
        if decision["action"] != expected_action:
            raise RuntimeError(
                f"Branch action {expected_action!r} disagrees with {decision['action']!r}."
            )
        if branch in decision["branches"]:
            raise RuntimeError(f"CFG branch {branch!r} executed twice at step {step_index}.")

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
        numerator = (current.float() - previous.float()).abs().mean()
        denominator = previous.float().abs().mean().clamp_min(self.config.eps)
        value = numerator / denominator
        if not torch.isfinite(value):
            return math.inf
        return float(value.detach().item())


__all__ = ["SeaCacheConfig", "SeaCacheController"]
