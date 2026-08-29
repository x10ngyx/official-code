from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ProfileCase:
    """One concrete torch.nn.Module forward path to profile with Calflops."""

    name: str
    model: Any
    args: Sequence[Any] = field(default_factory=tuple)
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManualComponent:
    """Analytic correction for an operator path that Calflops does not count."""

    name: str
    flops: int | float
    formula: str
    macs: int | float | None = None
    params: int | float = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
