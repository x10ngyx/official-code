"""Calflops adapters for fixed-protocol T5 forward and VAE decode."""

from __future__ import annotations

import time
from typing import Any, Sequence

import torch
from torch import nn


TFLOP_DIVISOR = 1_000_000_000_000


class T5ForwardProfile(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        with torch.no_grad():
            return self.model(token_ids, attention_mask)


class VAEDecodeProfile(nn.Module):
    def __init__(self, model: nn.Module, scale: Sequence[Any]) -> None:
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, latent: torch.Tensor) -> Any:
        with torch.no_grad():
            return self.model.decode(latent, self.scale)


def _calculate(
    wrapper: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    calculate_flops_fn: Any,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    flops, macs, params = calculate_flops_fn(
        model=wrapper,
        args=list(inputs),
        kwargs={},
        include_backPropagation=False,
        print_results=False,
        print_detailed=False,
        output_as_string=False,
    )
    torch.cuda.synchronize()
    return {
        "flops": float(flops),
        "tflops": float(flops) / TFLOP_DIVISOR,
        "macs": float(macs),
        "tmacs": float(macs) / TFLOP_DIVISOR,
        "params": float(params),
        "profile_wall_seconds": time.perf_counter() - started,
    }


def profile_t5(
    *,
    model: nn.Module,
    text_tokens: int,
    device: torch.device,
    calculate_flops_fn: Any,
    calls_per_video: int = 2,
) -> dict[str, Any]:
    if text_tokens < 1 or calls_per_video < 1:
        raise ValueError("text_tokens and calls_per_video must be positive")
    wrapper = T5ForwardProfile(model).to(device)
    token_ids = torch.zeros((1, text_tokens), dtype=torch.long, device=device)
    attention_mask = torch.ones((1, text_tokens), dtype=torch.long, device=device)
    per_call = _calculate(wrapper, (token_ids, attention_mask), calculate_flops_fn)
    result = {
        "scope": "UMT5 encoder model forward; tokenizer CPU work excluded",
        "input_token_shape": [1, text_tokens],
        "calls_per_video": calls_per_video,
        "per_call": per_call,
        "estimated_flops_per_video": per_call["flops"] * calls_per_video,
        "estimated_tflops_per_video": per_call["tflops"] * calls_per_video,
    }
    del wrapper, token_ids, attention_mask
    return result


def profile_vae_decode(
    *,
    model: nn.Module,
    scale: Sequence[Any],
    latent_shape: tuple[int, int, int, int, int],
    device: torch.device,
    calculate_flops_fn: Any,
) -> dict[str, Any]:
    if len(latent_shape) != 5 or any(int(value) < 1 for value in latent_shape):
        raise ValueError("latent_shape must contain five positive dimensions")
    wrapper = VAEDecodeProfile(model, scale).to(device)
    latent = torch.zeros(latent_shape, dtype=torch.float32, device=device)
    per_call = _calculate(wrapper, (latent,), calculate_flops_fn)
    result = {
        "scope": "Wan VAE model.decode for one output video",
        "input_latent_shape_bcfhw": list(latent_shape),
        "calls_per_video": 1,
        "per_call": per_call,
        "estimated_flops_per_video": per_call["flops"],
        "estimated_tflops_per_video": per_call["tflops"],
    }
    del wrapper, latent
    return result


def validate_component_profiles(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"t5", "vae_decode"}:
        raise ValueError("component_profiles must contain exactly t5 and vae_decode")
    for component in ("t5", "vae_decode"):
        profile = payload[component]
        calls = profile.get("calls_per_video")
        flops = profile.get("estimated_flops_per_video")
        tflops = profile.get("estimated_tflops_per_video")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
            raise ValueError(f"invalid {component} calls_per_video")
        if not isinstance(flops, (int, float)) or float(flops) <= 0:
            raise ValueError(f"invalid {component} estimated_flops_per_video")
        if not isinstance(tflops, (int, float)) or float(tflops) <= 0:
            raise ValueError(f"invalid {component} estimated_tflops_per_video")
    return payload


__all__ = [
    "TFLOP_DIVISOR",
    "profile_t5",
    "profile_vae_decode",
    "validate_component_profiles",
]
