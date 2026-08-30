"""Reusable low-overhead T5 and VAE-decode timing instrumentation."""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable

import torch


class _TimedCallableProxy:
    def __init__(self, target: Any, timer: "ComponentTimer", component: str) -> None:
        self._target = target
        self._timer = timer
        self._component = component

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._timer.measure(
            self._component, self._target, *args, **kwargs
        )


class ComponentTimer:
    """Record component spans and restore all monkey patches after one sample."""

    COMPONENTS = ("t5", "vae_decode")

    def __init__(self, pipeline: Any, cuda_device: torch.device | None) -> None:
        self.pipeline = pipeline
        self.cuda_device = cuda_device
        self.records: dict[str, list[dict[str, Any]]] = {
            component: [] for component in self.COMPONENTS
        }
        self._original_text_encoder: Any | None = None
        self._original_vae_decode: Callable[..., Any] | None = None

    def _events(self) -> tuple[Any | None, Any | None]:
        if self.cuda_device is None:
            return None, None
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def measure(
        self,
        component: str,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if component not in self.records:
            raise ValueError(f"unsupported timed component: {component}")
        start_event, end_event = self._events()
        record: dict[str, Any] = {
            "call_index": len(self.records[component]),
            "_start_event": start_event,
            "_end_event": end_event,
        }
        started = time.perf_counter()
        if start_event is not None:
            start_event.record()
        try:
            return function(*args, **kwargs)
        finally:
            if end_event is not None:
                end_event.record()
            record["host_span_seconds"] = time.perf_counter() - started
            self.records[component].append(record)

    def install(self) -> None:
        text_encoder = getattr(self.pipeline, "text_encoder", None)
        if text_encoder is not None:
            self._original_text_encoder = text_encoder
            self.pipeline.text_encoder = _TimedCallableProxy(
                text_encoder, self, "t5"
            )

        vae = getattr(self.pipeline, "vae", None)
        decode = getattr(vae, "decode", None) if vae is not None else None
        if callable(decode):
            self._original_vae_decode = decode

            @wraps(decode)
            def wrapped_decode(*args: Any, **kwargs: Any) -> Any:
                return self.measure("vae_decode", decode, *args, **kwargs)

            vae.decode = wrapped_decode

    def finalize(self) -> None:
        for component in self.COMPONENTS:
            for record in self.records[component]:
                start_event = record.pop("_start_event")
                end_event = record.pop("_end_event")
                record["cuda_seconds"] = (
                    float(start_event.elapsed_time(end_event)) / 1000.0
                    if start_event is not None
                    else None
                )

    def restore(self) -> None:
        if self._original_text_encoder is not None:
            self.pipeline.text_encoder = self._original_text_encoder
        vae = getattr(self.pipeline, "vae", None)
        if vae is not None and self._original_vae_decode is not None:
            vae.decode = self._original_vae_decode

    def summary(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for component in self.COMPONENTS:
            rows = self.records[component]
            cuda_values = [
                float(row["cuda_seconds"])
                for row in rows
                if row["cuda_seconds"] is not None
            ]
            result[component] = {
                "call_count": len(rows),
                "cuda_seconds": sum(cuda_values) if cuda_values else None,
                "host_span_seconds": sum(
                    float(row["host_span_seconds"]) for row in rows
                ),
                "calls": rows,
            }
        return result


__all__ = ["ComponentTimer"]
