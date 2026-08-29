from __future__ import annotations

import torch
from torch import nn

from calflops_eval import ManualComponent, ProfileCase, dense_attention_counts


class TinyDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(8, 16)
        self.output = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(torch.nn.functional.gelu(self.input(x)))


class TinyController(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.projection(state)


def build_profile_items():
    attention = dense_attention_counts(
        batch_size=1,
        query_tokens=4,
        key_value_tokens=4,
        num_heads=2,
        head_dim=4,
    )
    return [
        ProfileCase(
            name="high_full",
            model=TinyDenoiser(),
            args=(torch.zeros(1, 4, 8),),
            metadata={"stage": "high", "branch_cost": "one CFG branch"},
        ),
        ProfileCase(
            name="low_full",
            model=TinyDenoiser(),
            args=(torch.zeros(1, 4, 8),),
            metadata={"stage": "low", "branch_cost": "one CFG branch"},
        ),
        ProfileCase(
            name="controller",
            model=TinyController(),
            args=(torch.zeros(1, 4),),
        ),
        ManualComponent(
            name="attention_core",
            flops=attention["flops"],
            macs=attention["macs"],
            formula=attention["formula"],
            metadata={"inputs": attention["inputs"]},
        ),
        ManualComponent(
            name="reuse_path",
            flops=64,
            formula="toy example: 64 elementwise operations",
        ),
    ]
