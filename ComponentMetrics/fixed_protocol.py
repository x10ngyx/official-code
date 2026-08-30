"""Fail-closed validation for workspace-fixed Wan inference protocols."""

from __future__ import annotations

from typing import Any


def validate_wan21_t2v_1_3b_args(args: Any) -> None:
    expected = {
        "task": "t2v-1.3B",
        "size": "832*480",
        "frame_num": 81,
        "sample_steps": 50,
        "sample_solver": "unipc",
        "sample_shift": 5.0,
        "sample_guide_scale": 5.0,
        "base_seed": 42,
        "offload_model": False,
        "t5_cpu": False,
        "t5_fsdp": False,
        "dit_fsdp": False,
        "ulysses_size": 1,
        "ring_size": 1,
        "use_prompt_extend": False,
    }
    mismatches = {
        name: {"expected": value, "observed": getattr(args, name, None)}
        for name, value in expected.items()
        if getattr(args, name, None) != value
    }
    if mismatches:
        raise ValueError(f"Wan2.1-1.3B fixed protocol mismatch: {mismatches}")


__all__ = ["validate_wan21_t2v_1_3b_args"]
