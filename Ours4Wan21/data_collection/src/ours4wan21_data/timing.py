"""TeaCache4Wan21-compatible one-sample pipeline and DiT instrumentation."""

from __future__ import annotations

import json
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any

import torch

REPOSITORY_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_DIR / "ComponentMetrics"))
from component_timing import ComponentTimer  # noqa: E402


class PipelineProfiler:
    """Instrument one `WanT2V.generate()` call and then restore the pipeline."""

    def __init__(
        self,
        pipeline: Any,
        *,
        pipeline_init_wall_seconds: float,
        output_path: Path,
        implementation: str,
    ) -> None:
        self.pipeline = pipeline
        self.model = pipeline.model
        self.pipeline_init_wall_seconds = pipeline_init_wall_seconds
        self.output_path = output_path
        self.implementation = implementation
        self.blocks = list(getattr(self.model, "blocks", ()))
        self.block_count = len(self.blocks)
        self.calls: list[dict[str, Any]] = []
        self._active_call: dict[str, Any] | None = None
        self._original_blocks: list[Any] = []
        self._original_model_forward = self.model.forward
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

    def _events(self) -> tuple[Any | None, Any | None]:
        if self.cuda_device is None:
            return None, None
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def install(self) -> None:
        if self.output_path.exists():
            raise FileExistsError(f"refusing to overwrite timing trace {self.output_path}")
        if not self.blocks:
            raise ValueError("Wan model exposes no transformer blocks")
        self.component_timer.install()
        for block in self.blocks:
            original = block.forward
            self._original_blocks.append(original)

            @wraps(original)
            def wrapped_block(*args: Any, _original: Any = original, **kwargs: Any):
                if self._active_call is not None:
                    self._active_call["blocks_executed"] += 1
                return _original(*args, **kwargs)

            block.forward = wrapped_block

        @wraps(self._original_model_forward)
        def wrapped_model(*args: Any, **kwargs: Any):
            index = len(self.calls)
            start_event, end_event = self._events()
            record: dict[str, Any] = {
                "call_index": index,
                "step_index": index // 2,
                "cfg_branch": "condition" if index % 2 == 0 else "uncondition",
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

        self.model.forward = wrapped_model

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
                elapsed = time.perf_counter() - started
                self._finalize_calls()
                self.component_timer.finalize()
                self._restore()
                self._write(status=status, error=error, generate_wall_seconds=elapsed)

        self.pipeline.generate = wrapped_generate

    def _finalize_calls(self) -> None:
        for record in self.calls:
            start_event = record.pop("_start_event")
            end_event = record.pop("_end_event")
            record["cuda_seconds"] = (
                float(start_event.elapsed_time(end_event)) / 1000.0
                if start_event is not None else None
            )
            record["full_compute"] = record["blocks_executed"] == self.block_count
            record["reuse"] = record["blocks_executed"] == 0

    def _restore(self) -> None:
        self.component_timer.restore()
        self.model.forward = self._original_model_forward
        self.pipeline.generate = self._original_generate
        for block, original in zip(self.blocks, self._original_blocks):
            block.forward = original

    def _write(self, *, status: str, error: str | None, generate_wall_seconds: float) -> None:
        cuda_values = [
            float(row["cuda_seconds"])
            for row in self.calls if row["cuda_seconds"] is not None
        ]
        component_latency = self.component_timer.summary()
        payload = {
            "schema_version": 2,
            "status": status,
            "implementation": self.implementation,
            "latency_scope": {
                "pipeline_init_wall_seconds": "Wan pipeline construction including checkpoint loading",
                "pipeline_generate_wall_seconds": (
                    "text encoding, denoising/cache/CFG/scheduler, in-generate transfer/offload, "
                    "and VAE decode; excludes MP4 export, latent/file I/O, FFprobe, "
                    "PSNR/SSIM/LPIPS, and aggregation"
                ),
                "model_forward_cuda_seconds": "sum of CUDA-event spans for all Wan DiT forward calls",
                "t5_cuda_seconds": "sum of CUDA-event spans for T5 encoder calls",
                "vae_decode_cuda_seconds": "sum of CUDA-event spans for VAE decode calls",
            },
            "cuda_device": str(self.cuda_device) if self.cuda_device is not None else None,
            "cuda_device_name": (
                torch.cuda.get_device_name(self.cuda_device)
                if self.cuda_device is not None else None
            ),
            "pipeline_init_wall_seconds": self.pipeline_init_wall_seconds,
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
                        float(row["host_span_seconds"]) for row in self.calls
                    ),
                },
                "vae_decode": component_latency["vae_decode"],
            },
            "model_forward_host_span_seconds": sum(float(row["host_span_seconds"]) for row in self.calls),
            "transformer_block_count": self.block_count,
            "full_compute_forward_calls": sum(int(row["full_compute"]) for row in self.calls),
            "reuse_forward_calls": sum(int(row["reuse"]) for row in self.calls),
            "calls": self.calls,
            "error": error,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = ["PipelineProfiler"]
