#!/usr/bin/env python3
"""Low-overhead pipeline and DiT latency instrumentation for Wan2.1."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Iterator

import torch

REPOSITORY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_DIR / "ComponentMetrics"))
from component_timing import ComponentTimer  # noqa: E402


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
        self.model = pipeline.model
        self.init_wall_seconds = init_wall_seconds
        self.output_path = output_path
        self.implementation = implementation
        self.blocks = list(getattr(self.model, "blocks", ()))
        self.block_count = len(self.blocks)
        self.calls: list[dict[str, Any]] = []
        self.text_encoder_calls: list[dict[str, Any]] = []
        self.vae_decode_calls: list[dict[str, Any]] = []
        self._active_call: dict[str, Any] | None = None
        self._original_block_forwards: list[Any] = []
        self._original_model_forward = self.model.forward
        self._original_generate = pipeline.generate
        self._text_encoder = getattr(pipeline, "text_encoder", None)
        self._text_encoder_class: type[Any] | None = None
        self._original_text_encoder_call: Any | None = None
        self._vae = getattr(pipeline, "vae", None)
        self._original_vae_decode: Any | None = None
        self._denoising_started_at: float | None = None
        self._denoising_wall_seconds: float | None = None
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
        self._install_stage_wrappers()
        for block in self.blocks:
            original = block.forward
            self._original_block_forwards.append(original)

            @wraps(original)
            def wrapped_block(*args: Any, _original: Any = original, **kwargs: Any):
                if self._active_call is not None:
                    self._active_call["blocks_executed"] += 1
                return _original(*args, **kwargs)

            block.forward = wrapped_block

        @wraps(self._original_model_forward)
        def wrapped_model_forward(*args: Any, **kwargs: Any):
            if self._denoising_started_at is None:
                self._synchronize()
                self._denoising_started_at = time.perf_counter()
            call_index = len(self.calls)
            start_event, end_event = self._new_events()
            record: dict[str, Any] = {
                "call_index": call_index,
                "cfg_branch": "condition" if call_index % 2 == 0 else "uncondition",
                "blocks_executed": 0,
                "_start_event": start_event,
                "_end_event": end_event,
            }
            started = time.perf_counter()
            if start_event is not None:
                start_event.record()
            self._active_call = record
            try:
                return self._original_model_forward(*args, **kwargs)
            finally:
                if end_event is not None:
                    end_event.record()
                record["host_span_seconds"] = time.perf_counter() - started
                self._active_call = None
                self.calls.append(record)

        self.model.forward = wrapped_model_forward

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
                self._finish_denoising()
                self._finalize_calls()
                self.component_timer.finalize()
                self._restore()
                self._write(
                    status=status,
                    error=error,
                    generate_wall_seconds=generate_wall_seconds,
                )

        self.pipeline.generate = wrapped_generate

    def _install_stage_wrappers(self) -> None:
        if self._text_encoder is not None:
            encoder_class = type(self._text_encoder)
            original_call = encoder_class.__call__
            self._text_encoder_class = encoder_class
            self._original_text_encoder_call = original_call

            @wraps(original_call)
            def wrapped_text_encoder_call(
                instance: Any, *args: Any, **kwargs: Any
            ) -> Any:
                if instance is not self._text_encoder:
                    return original_call(instance, *args, **kwargs)
                self._synchronize()
                started = time.perf_counter()
                status = "success"
                try:
                    return original_call(instance, *args, **kwargs)
                except BaseException:
                    status = "error"
                    raise
                finally:
                    self._synchronize()
                    self.text_encoder_calls.append(
                        {
                            "call_index": len(self.text_encoder_calls),
                            "wall_seconds": time.perf_counter() - started,
                            "status": status,
                        }
                    )

            encoder_class.__call__ = wrapped_text_encoder_call

        if self._vae is not None and callable(getattr(self._vae, "decode", None)):
            original_decode = self._vae.decode
            self._original_vae_decode = original_decode

            @wraps(original_decode)
            def wrapped_vae_decode(*args: Any, **kwargs: Any) -> Any:
                self._synchronize()
                self._finish_denoising(synchronized=True)
                started = time.perf_counter()
                status = "success"
                try:
                    return original_decode(*args, **kwargs)
                except BaseException:
                    status = "error"
                    raise
                finally:
                    self._synchronize()
                    self.vae_decode_calls.append(
                        {
                            "call_index": len(self.vae_decode_calls),
                            "wall_seconds": time.perf_counter() - started,
                            "status": status,
                        }
                    )

            self._vae.decode = wrapped_vae_decode

    def _finish_denoising(self, *, synchronized: bool = False) -> None:
        if self._denoising_started_at is None or self._denoising_wall_seconds is not None:
            return
        if not synchronized:
            self._synchronize()
        self._denoising_wall_seconds = time.perf_counter() - self._denoising_started_at

    def _finalize_calls(self) -> None:
        for record in self.calls:
            start_event = record.pop("_start_event")
            end_event = record.pop("_end_event")
            record["cuda_seconds"] = (
                float(start_event.elapsed_time(end_event)) / 1000.0
                if start_event is not None
                else None
            )
            record["full_compute"] = record["blocks_executed"] == self.block_count
            record["reuse"] = record["blocks_executed"] == 0

    def _restore(self) -> None:
        self.model.forward = self._original_model_forward
        self.pipeline.generate = self._original_generate
        for block, original in zip(self.blocks, self._original_block_forwards):
            block.forward = original
        if (
            self._text_encoder_class is not None
            and self._original_text_encoder_call is not None
        ):
            self._text_encoder_class.__call__ = self._original_text_encoder_call
        if self._vae is not None and self._original_vae_decode is not None:
            self._vae.decode = self._original_vae_decode
        self.component_timer.restore()

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
        text_encoding_seconds = sum(
            float(record["wall_seconds"]) for record in self.text_encoder_calls
        )
        vae_decode_seconds = sum(
            float(record["wall_seconds"]) for record in self.vae_decode_calls
        )
        denoising_seconds = self._denoising_wall_seconds or 0.0
        stage_accounted_seconds = (
            text_encoding_seconds + denoising_seconds + vae_decode_seconds
        )
        model_forward_cuda_seconds = sum(cuda_values) if cuda_values else None
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
                "stage_wall_seconds": (
                    "synchronized, non-overlapping text encoding, denoising-core, "
                    "VAE decode, and residual pipeline wall-time decomposition"
                ),
            },
            "cuda_device": str(self.cuda_device) if self.cuda_device is not None else None,
            "cuda_device_name": (
                torch.cuda.get_device_name(self.cuda_device)
                if self.cuda_device is not None
                else None
            ),
            "pipeline_init_wall_seconds": self.init_wall_seconds,
            "pipeline_generate_wall_seconds": generate_wall_seconds,
            "t5_cuda_seconds": component_latency["t5"]["cuda_seconds"],
            "dit_cuda_seconds": model_forward_cuda_seconds,
            "vae_decode_cuda_seconds": component_latency["vae_decode"][
                "cuda_seconds"
            ],
            "component_latency": {
                "t5": component_latency["t5"],
                "dit": {
                    "call_count": len(self.calls),
                    "cuda_seconds": model_forward_cuda_seconds,
                    "host_span_seconds": sum(
                        float(record["host_span_seconds"])
                        for record in self.calls
                    ),
                },
                "vae_decode": component_latency["vae_decode"],
            },
            "stage_wall_seconds": {
                "text_encoding": text_encoding_seconds,
                "denoising_core": denoising_seconds,
                "vae_decode": vae_decode_seconds,
                "pipeline_other": generate_wall_seconds - stage_accounted_seconds,
                "accounted_total": stage_accounted_seconds,
                "denoising_non_dit_cuda_estimate": (
                    denoising_seconds - model_forward_cuda_seconds
                    if model_forward_cuda_seconds is not None
                    else None
                ),
            },
            "stage_calls": {
                "text_encoder": self.text_encoder_calls,
                "vae_decode": self.vae_decode_calls,
            },
            "model_forward_call_count": len(self.calls),
            "model_forward_cuda_seconds": model_forward_cuda_seconds,
            "model_forward_host_span_seconds": sum(
                float(record["host_span_seconds"]) for record in self.calls
            ),
            "transformer_block_count": self.block_count,
            "full_compute_forward_calls": sum(
                int(bool(record["full_compute"])) for record in self.calls
            ),
            "reuse_forward_calls": sum(int(bool(record["reuse"])) for record in self.calls),
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

    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite timing output: {destination}")
    pipeline_class = wan_module.WanI2V if task == "i2v-14B" else wan_module.WanT2V
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
