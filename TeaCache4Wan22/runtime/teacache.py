"""TeaCache controller for the pinned Wan2.2 T2V-A14B reproduction.

This module implements the non-retention TeaCache gate.  The gate state is
shared by the conditional and unconditional CFG branches, while the cached
Transformer-block residual is kept separately for each branch.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch


STAGES = ("high", "low")
BRANCHES = ("cond", "uncond")
PROTOCOL_KEYS = (
    "task",
    "size_wh",
    "frame_num",
    "sampling_steps",
    "sample_solver",
    "shift",
    "guide_scale_low_high",
    "boundary",
    "param_dtype",
    "use_ret_steps",
)


@dataclass(frozen=True)
class TeaCacheConfig:
    """Runtime configuration for the fixed-protocol TeaCache reproduction."""

    threshold: float
    coefficients_path: str
    trace_path: Optional[str] = None
    eps: float = 1e-12
    use_ret_steps: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("TeaCache threshold must be a finite positive value.")
        if not self.coefficients_path:
            raise ValueError("TeaCache coefficients_path is required.")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("TeaCache eps must be a finite positive value.")
        if self.use_ret_steps:
            raise ValueError(
                "TeaCache4Wan22 v0.1 reproduces only use_ret_steps=False."
            )


@dataclass
class _GateState:
    previous_feature: Optional[torch.Tensor] = None
    accumulator: float = 0.0


class TeaCacheController:
    """Shared-gate, branch-separated block-residual TeaCache controller."""

    def __init__(self, config: TeaCacheConfig):
        self.config = config
        coefficient_path = Path(config.coefficients_path).expanduser().resolve()
        raw = coefficient_path.read_bytes()
        self.coefficient_path = coefficient_path
        self.coefficient_sha256 = hashlib.sha256(raw).hexdigest()
        self.coefficient_payload = json.loads(raw.decode("utf-8"))
        self.coefficients, self.protocol = self._parse_coefficients(
            self.coefficient_payload
        )

        self._gate_states: Dict[str, _GateState] = {}
        self._residuals: Dict[Tuple[str, str], torch.Tensor] = {}
        self._decisions: list[dict[str, Any]] = []
        self._current_plan_step: Optional[int] = None
        self._current_plan_stage: Optional[str] = None
        self._current_plan_feature: Optional[torch.Tensor] = None

    @staticmethod
    def _parse_coefficients(
        payload: Mapping[str, Any]
    ) -> tuple[dict[str, tuple[float, ...]], dict[str, Any]]:
        if payload.get("schema") != "teacache4wan22_coefficients_v1":
            raise ValueError(
                "Coefficient JSON must use the validated teacache4wan22_coefficients_v1 schema."
            )
        stage_payload = payload.get("stages", {})
        protocol = dict(payload.get("protocol", {}))

        coefficients: dict[str, tuple[float, ...]] = {}
        for stage in STAGES:
            if stage not in stage_payload:
                raise ValueError(f"Coefficient JSON is missing stage {stage!r}.")
            values = tuple(
                float(value)
                for value in stage_payload[stage]["coefficients_descending"]
            )
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"Stage {stage!r} must provide five finite descending quartic coefficients."
                )
            coefficients[stage] = values

        missing_protocol = [key for key in PROTOCOL_KEYS if key not in protocol]
        if missing_protocol:
            raise ValueError(
                f"Coefficient JSON is missing protocol fields: {missing_protocol}"
            )
        if protocol["use_ret_steps"] is not False:
            raise ValueError("Coefficient JSON is not a non-retention TeaCache fit.")
        return coefficients, protocol

    def validate_runtime_protocol(self, runtime: Mapping[str, Any]) -> None:
        """Fail closed when coefficients and the active inference protocol differ."""

        missing = [key for key in PROTOCOL_KEYS if key not in runtime]
        if missing:
            raise ValueError(f"Runtime protocol is missing fields: {missing}")
        mismatches = []
        for key in PROTOCOL_KEYS:
            expected = self.protocol[key]
            observed = runtime[key]
            if not self._protocol_values_equal(expected, observed):
                mismatches.append(
                    {"field": key, "coefficient": expected, "runtime": observed}
                )
        if mismatches:
            raise ValueError(
                "TeaCache coefficient/runtime protocol mismatch: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )

    @staticmethod
    def _protocol_values_equal(expected: Any, observed: Any) -> bool:
        if isinstance(expected, (list, tuple)) or isinstance(observed, (list, tuple)):
            try:
                expected_values = list(expected)
                observed_values = list(observed)
            except TypeError:
                return False
            return len(expected_values) == len(observed_values) and all(
                TeaCacheController._protocol_values_equal(left, right)
                for left, right in zip(expected_values, observed_values)
            )
        if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
            return math.isclose(float(expected), float(observed), rel_tol=0.0, abs_tol=1e-12)
        return expected == observed

    def plan_step(
        self,
        stage: str,
        step_index: int,
        num_steps: int,
        compact_time_embedding: torch.Tensor,
    ) -> bool:
        """Return True when both CFG branches must reuse their cached residual."""

        self._validate_call(stage, step_index, num_steps, compact_time_embedding)
        feature = compact_time_embedding.detach().float().reshape(1, -1).clone()

        if self._current_plan_step == step_index:
            if self._current_plan_stage != stage:
                raise RuntimeError("The two CFG calls for one step used different model stages.")
            if self._current_plan_feature is None or not torch.equal(
                feature, self._current_plan_feature
            ):
                raise RuntimeError(
                    "Conditional and unconditional CFG calls produced different TeaCache gate features."
                )
            return self._decisions[-1]["action"] == "reuse"

        expected_step = len(self._decisions)
        if step_index != expected_step:
            raise RuntimeError(
                f"TeaCache steps must be monotonic and contiguous: expected {expected_step}, got {step_index}."
            )

        state = self._gate_states.setdefault(stage, _GateState())
        accumulator_before = state.accumulator
        relative_l1: Optional[float] = None
        rescaled_distance: Optional[float] = None
        accumulator_candidate: Optional[float] = None
        forced_reason: Optional[str] = None

        if step_index == 0:
            forced_reason = "global_first"
        elif state.previous_feature is None:
            forced_reason = "stage_first"
        elif step_index == num_steps - 1:
            forced_reason = "global_final"
        elif any((stage, branch) not in self._residuals for branch in BRANCHES):
            forced_reason = "missing_branch_residual"

        if forced_reason is not None:
            action = "recompute"
            state.accumulator = 0.0
        else:
            relative_l1 = self._relative_l1(feature, state.previous_feature)
            rescaled_distance = self._polyval(self.coefficients[stage], relative_l1)
            if not math.isfinite(rescaled_distance):
                raise RuntimeError("TeaCache polynomial produced a non-finite value.")
            accumulator_candidate = accumulator_before + rescaled_distance
            if accumulator_candidate < self.config.threshold:
                action = "reuse"
                state.accumulator = accumulator_candidate
            else:
                action = "recompute"
                state.accumulator = 0.0

        state.previous_feature = feature
        decision = {
            "step_index": step_index,
            "stage": stage,
            "action": action,
            "forced_reason": forced_reason,
            "relative_l1": relative_l1,
            "rescaled_distance": rescaled_distance,
            "accumulator_before": accumulator_before,
            "accumulator_candidate": accumulator_candidate,
            "accumulator_after": state.accumulator,
            "threshold": self.config.threshold,
            "branches": {},
        }
        self._decisions.append(decision)
        self._current_plan_step = step_index
        self._current_plan_stage = stage
        self._current_plan_feature = feature
        return action == "reuse"

    def reuse_residual(
        self, stage: str, branch: str, step_index: int
    ) -> torch.Tensor:
        self._validate_branch_action(stage, branch, step_index, "reuse")
        key = (stage, branch)
        residual = self._residuals.get(key)
        if residual is None:
            raise RuntimeError(
                f"TeaCache reuse requested without a residual for stage={stage}, branch={branch}."
            )
        self._decisions[-1]["branches"][branch] = "reuse"
        return residual

    def record_recompute(
        self,
        stage: str,
        branch: str,
        step_index: int,
        residual: torch.Tensor,
    ) -> None:
        self._validate_branch_action(stage, branch, step_index, "recompute")
        if not torch.is_tensor(residual) or residual.numel() == 0:
            raise RuntimeError("TeaCache received an invalid block residual.")
        # `x - block_input` already owns fresh storage.  Avoid a second
        # full-latent clone here because it adds hundreds of MiB of transient
        # memory and is not part of the reference TeaCache algorithm.
        self._residuals[(stage, branch)] = residual.detach()
        self._decisions[-1]["branches"][branch] = "recompute"

    def clear_stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown TeaCache stage: {stage!r}")
        self._gate_states.pop(stage, None)
        for branch in BRANCHES:
            self._residuals.pop((stage, branch), None)
        if self._current_plan_stage == stage:
            self._current_plan_step = None
            self._current_plan_stage = None
            self._current_plan_feature = None

    def validate_complete(self, num_steps: int) -> None:
        observed_steps = [row["step_index"] for row in self._decisions]
        if observed_steps != list(range(num_steps)):
            raise RuntimeError(
                f"TeaCache trace is incomplete: observed={observed_steps}, expected 0..{num_steps - 1}."
            )
        seen_stages = set()
        for row in self._decisions:
            if row["stage"] not in seen_stages:
                if row["action"] != "recompute":
                    raise RuntimeError("Each model stage must begin with a recompute.")
                seen_stages.add(row["stage"])
            if set(row["branches"]) != set(BRANCHES):
                raise RuntimeError(
                    f"TeaCache step {row['step_index']} does not contain both CFG branches."
                )
            if any(action != row["action"] for action in row["branches"].values()):
                raise RuntimeError(
                    f"TeaCache CFG branches diverged at step {row['step_index']}."
                )
        if self._decisions[0]["action"] != "recompute":
            raise RuntimeError("TeaCache global first step was not recomputed.")
        if self._decisions[-1]["action"] != "recompute":
            raise RuntimeError("TeaCache global final step was not recomputed.")

    def summary(self) -> dict[str, Any]:
        per_stage = {}
        for stage in STAGES:
            rows = [row for row in self._decisions if row["stage"] == stage]
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
            "schema": "teacache4wan22_trace_v1",
            "threshold": self.config.threshold,
            "use_ret_steps": False,
            "coefficients_path": str(self.coefficient_path),
            "coefficients_sha256": self.coefficient_sha256,
            "coefficient_protocol": self.protocol,
            "total_steps": len(self._decisions),
            "reuse": sum(row["action"] == "reuse" for row in self._decisions),
            "recompute": sum(
                row["action"] == "recompute" for row in self._decisions
            ),
            "per_stage": per_stage,
            "decisions": self._decisions,
        }

    def write_trace(
        self,
        path: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Path]:
        target_text = path or self.config.trace_path
        if not target_text:
            return None
        target = Path(target_text).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
        payload = self.summary()
        if extra:
            overlap = sorted(set(payload).intersection(extra))
            if overlap:
                raise ValueError(f"TeaCache trace extra fields overlap built-ins: {overlap}")
            payload.update(dict(extra))
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return target

    def release(self) -> None:
        self._gate_states.clear()
        self._residuals.clear()
        self._current_plan_step = None
        self._current_plan_stage = None
        self._current_plan_feature = None

    def _validate_call(
        self,
        stage: str,
        step_index: int,
        num_steps: int,
        feature: torch.Tensor,
    ) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown TeaCache stage: {stage!r}")
        if num_steps <= 0 or not 0 <= step_index < num_steps:
            raise ValueError(
                f"Invalid TeaCache step {step_index} for num_steps={num_steps}."
            )
        if not torch.is_tensor(feature) or feature.numel() == 0:
            raise ValueError("TeaCache compact time embedding must be a non-empty tensor.")
        if not torch.isfinite(feature).all():
            raise RuntimeError("TeaCache compact time embedding contains non-finite values.")

    def _validate_branch_action(
        self, stage: str, branch: str, step_index: int, expected_action: str
    ) -> None:
        if branch not in BRANCHES:
            raise ValueError(f"Unknown TeaCache CFG branch: {branch!r}")
        if not self._decisions or self._decisions[-1]["step_index"] != step_index:
            raise RuntimeError("TeaCache branch action has no matching step decision.")
        decision = self._decisions[-1]
        if decision["stage"] != stage or decision["action"] != expected_action:
            raise RuntimeError(
                f"TeaCache branch action disagrees with the shared gate: {decision}."
            )
        if branch in decision["branches"]:
            raise RuntimeError(
                f"TeaCache branch {branch!r} was executed twice at step {step_index}."
            )

    def _relative_l1(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        numerator = (current - previous).abs().mean()
        denominator = previous.abs().mean().clamp_min(self.config.eps)
        value = numerator / denominator
        if not torch.isfinite(value):
            raise RuntimeError("TeaCache relative-L1 is non-finite.")
        return float(value.detach().item())

    @staticmethod
    def _polyval(coefficients: Sequence[float], value: float) -> float:
        result = 0.0
        for coefficient in coefficients:
            result = result * value + coefficient
        return result


__all__ = ["TeaCacheConfig", "TeaCacheController"]
