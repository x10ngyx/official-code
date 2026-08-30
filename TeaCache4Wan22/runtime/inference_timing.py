#!/usr/bin/env python3
"""Low-overhead pipeline and DiT latency instrumentation for Wan2.2."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Iterator

import torch

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_DIR / "ComponentMetrics"))
from component_timing import ComponentTimer  # noqa: E402


STAGES = ("high", "low")


class _PipelineProfiler:
    def __init__(
        self,
        pipeline: Any,
        *,
        init_wall_seconds: float,
        output_path: Path,
        implementation: str,
    ) -> None:
        self.pipeline = pipeline
        self.models = {
            "high": pipeline.high_noise_model,
            "low": pipeline.low_noise_model,
        }
        self.init_wall_seconds = init_wall_seconds
        self.output_path = output_path
        self.implementation = implementation
        self.blocks = {
            stage: list(getattr(model, "blocks", ()))
            for stage, model in self.models.items()
        }
        self.block_counts = {
            stage: len(blocks) for stage, blocks in self.blocks.items()
        }
        self.calls: list[dict[str, Any]] = []
        self._active_call: dict[str, Any] | None = None
        self._original_block_forwards: dict[str, list[Any]] = {
            stage: [] for stage in STAGES
        }
        self._original_model_forwards = {
            stage: model.forward for stage, model in self.models.items()
        }
        self._original_generate = pipeline.generate
        device = getattr(pipeline, "device", None)
        self.cuda_device = (
            torch.device(device)
            if device is not None
            and torch.cuda.is_available()
            and torch.device(device).type == "cuda"
            else None
        )
        self.component_timer = ComponentTimer(pipeline, self.cuda_device)

    def _synchronize(self) -> None:
        if self.cuda_device is not None:
            torch.cuda.synchronize(self.cuda_device)

    def _new_events(self) -> tuple[Any | None, Any | None]:
        if self.cuda_device is None:
            return None, None
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def install(self) -> None:
        self.component_timer.install()
        for stage in STAGES:
            for block in self.blocks[stage]:
                original = block.forward
                self._original_block_forwards[stage].append(original)

                @wraps(original)
                def wrapped_block(*args: Any, _original: Any = original, **kwargs: Any):
                    if self._active_call is not None:
                        self._active_call["blocks_executed"] += 1
                    return _original(*args, **kwargs)

                block.forward = wrapped_block

            original_model_forward = self._original_model_forwards[stage]

            @wraps(original_model_forward)
            def wrapped_model_forward(
                *args: Any,
                _stage: str = stage,
                _original: Any = original_model_forward,
                **kwargs: Any,
            ):
                call_index = len(self.calls)
                start_event, end_event = self._new_events()
                record: dict[str, Any] = {
                    "call_index": call_index,
                    "step_index": call_index // 2,
                    "model_stage": _stage,
                    "cfg_branch": "cond" if call_index % 2 == 0 else "uncond",
                    "blocks_executed": 0,
                    "_start_event": start_event,
                    "_end_event": end_event,
                }
                started = time.perf_counter()
                if start_event is not None:
                    start_event.record()
                self._active_call = record
                try:
                    return _original(*args, **kwargs)
                finally:
                    if end_event is not None:
                        end_event.record()
                    record["host_span_seconds"] = time.perf_counter() - started
                    self._active_call = None
                    self.calls.append(record)

            self.models[stage].forward = wrapped_model_forward

        @wraps(self._original_generate)
        def wrapped_generate(*args: Any, **kwargs: Any):
            self._synchronize()
            started = time.perf_counter()
            status = "success"
            error: str | None = None
            try:
                return self._original_generate(*args, **kwargs)
            except BaseException as exc:
                status = "error"
                error = repr(exc)
                raise
            finally:
                self._synchronize()
                generate_wall_seconds = time.perf_counter() - started
                self._finalize_calls()
                self.component_timer.finalize()
                self._restore()
                self._write(
                    status=status,
                    error=error,
                    generate_wall_seconds=generate_wall_seconds,
                )

        self.pipeline.generate = wrapped_generate

    def _finalize_calls(self) -> None:
        for record in self.calls:
            start_event = record.pop("_start_event")
            end_event = record.pop("_end_event")
            record["cuda_seconds"] = (
                float(start_event.elapsed_time(end_event)) / 1000.0
                if start_event is not None
                else None
            )
            block_count = self.block_counts[record["model_stage"]]
            record["full_compute"] = record["blocks_executed"] == block_count
            record["reuse"] = record["blocks_executed"] == 0

    def _restore(self) -> None:
        self.component_timer.restore()
        self.pipeline.generate = self._original_generate
        for stage in STAGES:
            self.models[stage].forward = self._original_model_forwards[stage]
            for block, original in zip(
                self.blocks[stage], self._original_block_forwards[stage]
            ):
                block.forward = original

    def _write(
        self,
        *,
        status: str,
        error: str | None,
        generate_wall_seconds: float,
    ) -> None:
        cuda_values = [
            float(record["cuda_seconds"])
            for record in self.calls
            if record["cuda_seconds"] is not None
        ]
        component_latency = self.component_timer.summary()
        payload = {
            "schema_version": 2,
            "status": status,
            "implementation": self.implementation,
            "latency_scope": {
                "pipeline_init_wall_seconds": (
                    "Wan pipeline construction including checkpoint loading"
                ),
                "pipeline_generate_wall_seconds": (
                    "text encoding, denoising loop, and VAE decode; excludes MP4 export"
                ),
                "model_forward_cuda_seconds": (
                    "sum of CUDA-event spans for all Wan DiT forward calls"
                ),
                "t5_cuda_seconds": "sum of CUDA-event spans for T5 encoder calls",
                "vae_decode_cuda_seconds": (
                    "sum of CUDA-event spans for VAE decode calls"
                ),
            },
            "cuda_device": (
                str(self.cuda_device) if self.cuda_device is not None else None
            ),
            "cuda_device_name": (
                torch.cuda.get_device_name(self.cuda_device)
                if self.cuda_device is not None
                else None
            ),
            "pipeline_init_wall_seconds": self.init_wall_seconds,
            "pipeline_generate_wall_seconds": generate_wall_seconds,
            "model_forward_call_count": len(self.calls),
            "model_forward_cuda_seconds": sum(cuda_values) if cuda_values else None,
            "dit_cuda_seconds": sum(cuda_values) if cuda_values else None,
            "t5_cuda_seconds": component_latency["t5"]["cuda_seconds"],
            "vae_decode_cuda_seconds": component_latency["vae_decode"]["cuda_seconds"],
            "component_latency": {
                "t5": component_latency["t5"],
                "dit": {
                    "call_count": len(self.calls),
                    "cuda_seconds": sum(cuda_values) if cuda_values else None,
                    "host_span_seconds": sum(
                        float(record["host_span_seconds"]) for record in self.calls
                    ),
                },
                "vae_decode": component_latency["vae_decode"],
            },
            "model_forward_host_span_seconds": sum(
                float(record["host_span_seconds"]) for record in self.calls
            ),
            "transformer_block_count_by_stage": dict(self.block_counts),
            "full_compute_forward_calls": sum(
                int(bool(record["full_compute"])) for record in self.calls
            ),
            "reuse_forward_calls": sum(
                int(bool(record["reuse"])) for record in self.calls
            ),
            "calls": self.calls,
            "error": error,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


@contextmanager
def patch_pipeline_timing(
    wan_module: Any,
    *,
    task: str,
    output_path: str | Path,
    implementation: str,
) -> Iterator[None]:
    """Instrument the constructed pipeline without changing numerical outputs."""

    if task != "t2v-A14B":
        raise ValueError("TeaCache4Wan22 timing supports only t2v-A14B")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite timing output: {destination}")
    pipeline_class = wan_module.WanT2V
    original_init = pipeline_class.__init__

    @wraps(original_init)
    def patched_init(instance: Any, *args: Any, **kwargs: Any) -> None:
        started = time.perf_counter()
        original_init(instance, *args, **kwargs)
        device = getattr(instance, "device", None)
        if (
            device is not None
            and torch.cuda.is_available()
            and torch.device(device).type == "cuda"
        ):
            torch.cuda.synchronize(torch.device(device))
        profiler = _PipelineProfiler(
            instance,
            init_wall_seconds=time.perf_counter() - started,
            output_path=destination,
            implementation=implementation,
        )
        profiler.install()

    pipeline_class.__init__ = patched_init
    try:
        yield
    finally:
        pipeline_class.__init__ = original_init


__all__ = ["patch_pipeline_timing"]
