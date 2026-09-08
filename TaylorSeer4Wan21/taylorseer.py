#!/usr/bin/env python3
"""Independent TaylorSeer integration for the locked original Wan2.1 pipeline.

The implementation follows the algorithm and public parameterization described
by the TaylorSeer paper and its Wan2.1 experiment, while deliberately avoiding
source-code reuse from the GPL-3.0 reference repository.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.cuda.amp as amp


MODULE_NAMES = ("self_attention", "cross_attention", "ffn")


@dataclass
class TaylorSeerState:
    fresh_threshold: int
    sample_steps: int
    max_order: int = 1
    first_enhance: int = 1
    call_index: int = 0
    cache_counter: int = 0
    current_step: int = -1
    current_stream: str = ""
    current_type: str = ""
    last_full_step: int | None = None
    previous_full_step: int | None = None
    cache: dict[str, dict[int, dict[str, dict[int, torch.Tensor]]]] = field(
        default_factory=dict
    )

    def reset(self) -> None:
        self.call_index = 0
        self.cache_counter = 0
        self.current_step = -1
        self.current_stream = ""
        self.current_type = ""
        self.last_full_step = None
        self.previous_full_step = None
        self.cache = {"cond_stream": {}, "uncond_stream": {}}

    def begin_call(self) -> tuple[int, str, str]:
        if self.call_index >= self.sample_steps * 2:
            raise RuntimeError(
                "TaylorSeer state was not reset before a new sample: "
                f"call_index={self.call_index}"
            )
        step = self.call_index // 2
        stream = "cond_stream" if self.call_index % 2 == 0 else "uncond_stream"
        if stream == "cond_stream":
            full = step < self.first_enhance or self.cache_counter == self.fresh_threshold - 1
            if full:
                self.previous_full_step = self.last_full_step
                self.last_full_step = step
                self.cache_counter = 0
                self.current_type = "full"
            else:
                self.cache_counter += 1
                self.current_type = "taylor"
        elif step != self.current_step or self.current_type not in {"full", "taylor"}:
            raise RuntimeError("unconditional CFG call did not follow its conditional call")
        self.current_step = step
        self.current_stream = stream
        self.call_index += 1
        return step, stream, self.current_type

    def update_feature(self, layer: int, module: str, feature: torch.Tensor) -> None:
        layer_cache = self.cache[self.current_stream].setdefault(layer, {})
        previous = layer_cache.get(module)
        updated = {0: feature}
        if self.max_order == 1 and previous is not None and 0 in previous:
            if self.previous_full_step is None or self.last_full_step is None:
                raise RuntimeError("missing full-step history for first-order Taylor update")
            interval = self.last_full_step - self.previous_full_step
            if interval <= 0:
                raise RuntimeError(f"invalid Taylor difference interval: {interval}")
            updated[1] = (feature - previous[0]) / interval
        layer_cache[module] = updated

    def predict_feature(self, layer: int, module: str) -> torch.Tensor:
        if self.last_full_step is None:
            raise RuntimeError("Taylor prediction requested before a full step")
        factors = self.cache[self.current_stream][layer][module]
        distance = self.current_step - self.last_full_step
        prediction = factors[0]
        if self.max_order == 1 and 1 in factors:
            prediction = prediction + factors[1] * distance
        return prediction


def _full_block_forward(
    block: Any,
    x: torch.Tensor,
    e: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    context: torch.Tensor,
    context_lens: torch.Tensor | None,
) -> torch.Tensor:
    state: TaylorSeerState = block._taylorseer_state
    layer: int = block._taylorseer_layer
    if state.current_type != "full":
        raise RuntimeError("full block forward called during a Taylor cache step")
    if e.dtype != torch.float32:
        raise TypeError(f"expected float32 modulation embedding, got {e.dtype}")
    with amp.autocast(dtype=torch.float32):
        modulation = (block.modulation + e).chunk(6, dim=1)

    residual = block.self_attn(
        block.norm1(x).float() * (1 + modulation[1]) + modulation[0],
        seq_lens,
        grid_sizes,
        freqs,
    )
    state.update_feature(layer, "self_attention", residual)
    with amp.autocast(dtype=torch.float32):
        x = x + residual * modulation[2]

    residual = block.cross_attn(block.norm3(x), context, context_lens)
    state.update_feature(layer, "cross_attention", residual)
    x = x + residual

    residual = block.ffn(
        block.norm2(x).float() * (1 + modulation[4]) + modulation[3]
    )
    state.update_feature(layer, "ffn", residual)
    with amp.autocast(dtype=torch.float32):
        x = x + residual * modulation[5]
    return x


def _cached_block_step(
    block: Any,
    x: torch.Tensor,
    e: torch.Tensor,
    state: TaylorSeerState,
    layer: int,
) -> torch.Tensor:
    if e.dtype != torch.float32:
        raise TypeError(f"expected float32 modulation embedding, got {e.dtype}")
    with amp.autocast(dtype=torch.float32):
        modulation = (block.modulation + e).chunk(6, dim=1)
        x = x + state.predict_feature(layer, "self_attention") * modulation[2]
    x = x + state.predict_feature(layer, "cross_attention")
    with amp.autocast(dtype=torch.float32):
        x = x + state.predict_feature(layer, "ffn") * modulation[5]
    return x


def _taylorseer_model_forward(
    model: Any,
    x: list[torch.Tensor],
    t: torch.Tensor,
    context: list[torch.Tensor],
    seq_len: int,
    clip_fea: torch.Tensor | None = None,
    y: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    state: TaylorSeerState = model._taylorseer_state
    _, _, calculation_type = state.begin_call()
    if model.model_type == "i2v":
        if clip_fea is None or y is None:
            raise ValueError("Wan I2V requires clip_fea and y")
    device = model.patch_embedding.weight.device
    if model.freqs.device != device:
        model.freqs = model.freqs.to(device)
    if y is not None:
        x = [torch.cat([left, right], dim=0) for left, right in zip(x, y)]

    x = [model.patch_embedding(item.unsqueeze(0)) for item in x]
    grid_sizes = torch.stack(
        [torch.tensor(item.shape[2:], dtype=torch.long) for item in x]
    )
    x = [item.flatten(2).transpose(1, 2) for item in x]
    seq_lens = torch.tensor([item.size(1) for item in x], dtype=torch.long)
    if seq_lens.max() > seq_len:
        raise ValueError("patch sequence exceeds configured sequence length")
    x = torch.cat(
        [
            torch.cat(
                [item, item.new_zeros(1, seq_len - item.size(1), item.size(2))],
                dim=1,
            )
            for item in x
        ]
    )

    from wan.modules.model import sinusoidal_embedding_1d

    with amp.autocast(dtype=torch.float32):
        e = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, t).float())
        e0 = model.time_projection(e).unflatten(1, (6, model.dim))
    context_lens = None
    context = model.text_embedding(
        torch.stack(
            [
                torch.cat(
                    [item, item.new_zeros(model.text_len - item.size(0), item.size(1))]
                )
                for item in context
            ]
        )
    )
    if clip_fea is not None:
        context = torch.cat([model.img_emb(clip_fea), context], dim=1)

    kwargs = {
        "e": e0,
        "seq_lens": seq_lens,
        "grid_sizes": grid_sizes,
        "freqs": model.freqs,
        "context": context,
        "context_lens": context_lens,
    }
    if calculation_type == "full":
        for block in model.blocks:
            x = block(x, **kwargs)
    else:
        for layer, block in enumerate(model.blocks):
            x = _cached_block_step(block, x, e0, state, layer)

    x = model.head(x, e)
    x = model.unpatchify(x, grid_sizes)
    return [item.float() for item in x]


def apply_taylorseer(
    pipeline: Any,
    *,
    fresh_threshold: int,
    sample_steps: int,
    max_order: int = 1,
    first_enhance: int = 1,
) -> None:
    """Enable or reconfigure TaylorSeer on one persistent Wan pipeline."""
    if isinstance(fresh_threshold, bool) or not isinstance(fresh_threshold, int):
        raise TypeError("fresh_threshold must be an integer")
    if fresh_threshold < 1:
        raise ValueError("fresh_threshold must be at least 1")
    if sample_steps < 1:
        raise ValueError("sample_steps must be positive")
    if max_order != 1 or first_enhance != 1:
        raise ValueError("the locked TaylorSeer protocol requires max_order=1, first_enhance=1")

    model = pipeline.model
    state = getattr(model, "_taylorseer_state", None)
    if state is None:
        state = TaylorSeerState(
            fresh_threshold=fresh_threshold,
            sample_steps=sample_steps,
            max_order=max_order,
            first_enhance=first_enhance,
        )
        model._taylorseer_state = state
        model._taylorseer_original_forward = model.forward
        for layer, block in enumerate(model.blocks):
            block._taylorseer_state = state
            block._taylorseer_layer = layer
            block._taylorseer_original_forward = block.forward
            block.forward = types.MethodType(_full_block_forward, block)
        model.forward = types.MethodType(_taylorseer_model_forward, model)
    else:
        state.fresh_threshold = fresh_threshold
        state.sample_steps = sample_steps
        state.max_order = max_order
        state.first_enhance = first_enhance
    state.reset()


def reset_taylorseer(pipeline: Any, *, fresh_threshold: int | None = None) -> None:
    """Reset all per-sample caches and optionally select a new integer interval."""
    state: TaylorSeerState | None = getattr(pipeline.model, "_taylorseer_state", None)
    if state is None:
        raise RuntimeError("TaylorSeer has not been applied to this pipeline")
    if fresh_threshold is not None:
        if isinstance(fresh_threshold, bool) or not isinstance(fresh_threshold, int):
            raise TypeError("fresh_threshold must be an integer")
        if fresh_threshold < 1:
            raise ValueError("fresh_threshold must be at least 1")
        state.fresh_threshold = fresh_threshold
    state.reset()


def expected_full_steps(sample_steps: int, fresh_threshold: int) -> list[int]:
    """Return the full-compute diffusion steps for the locked scheduler."""
    if sample_steps < 1 or fresh_threshold < 1:
        raise ValueError("sample_steps and fresh_threshold must be positive")
    return list(range(0, sample_steps, fresh_threshold))
