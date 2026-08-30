"""Locked Wan2.1 runtime with full-compute or random-threshold capture."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .controller import RandomThresholdConfig, RandomThresholdSeaCacheController
from .manifest import NUM_STEPS, PROTOCOL
from .source_lock import WAN21_COMMIT, file_sha256, validate_wan21_source


@dataclass
class RuntimeCapture:
    mode: str
    trajectory_id: str = ""
    manifest_record: dict[str, Any] = field(default_factory=dict)
    latents: list[torch.Tensor] = field(default_factory=list)
    step_metadata: list[dict[str, Any]] = field(default_factory=list)
    trace_payload: Optional[dict[str, Any]] = None

    def reset(self, trajectory_id: str, manifest_record: dict[str, Any]) -> None:
        self.trajectory_id = trajectory_id
        self.manifest_record = dict(manifest_record)
        self.latents = []
        self.step_metadata = []
        self.trace_payload = None

    def save_artifacts(
        self,
        trace_path: Path,
        latent_dir: Path,
        *,
        latent_dtype: torch.dtype = torch.float16,
    ) -> dict[str, Any]:
        if self.trace_payload is None or len(self.latents) != NUM_STEPS:
            raise RuntimeError("runtime capture is incomplete")
        if len(self.step_metadata) != NUM_STEPS:
            raise RuntimeError("runtime step metadata is incomplete")
        if trace_path.exists() or latent_dir.exists():
            raise FileExistsError("refusing to overwrite trace/latent artifacts")
        latent_dir.mkdir(parents=True)
        latent_rows: list[dict[str, Any]] = []
        for step, latent in enumerate(self.latents):
            path = latent_dir / f"step_{step:03d}_input.pt"
            cpu = latent.detach().to(device="cpu", dtype=latent_dtype).contiguous()
            torch.save(cpu, path)
            latent_rows.append({
                "step_index": step,
                "latent_path": str(path.resolve()),
                "latent_shape": list(cpu.shape),
                "latent_dtype": str(cpu.dtype),
                "latent_mean": float(cpu.float().mean().item()),
                "latent_std": float(cpu.float().std(unbiased=False).item()),
                "latent_min": float(cpu.float().min().item()),
                "latent_max": float(cpu.float().max().item()),
            })
        decisions = list(self.trace_payload["decisions"])
        steps = []
        if self.trace_payload.get("schema") == "ours4wan21_random_threshold_trace_v2":
            if len(decisions) != 2 * NUM_STEPS:
                raise RuntimeError("candidate trace must contain 100 ordered branch decisions")
            for index in range(NUM_STEPS):
                branch_rows = decisions[2 * index:2 * index + 2]
                if [row.get("branch") for row in branch_rows] != ["cond", "uncond"]:
                    raise RuntimeError("candidate trace CFG branch order is malformed")
                if any(int(row.get("step_index", -1)) != index for row in branch_rows):
                    raise RuntimeError("candidate trace steps are not ordered 0..49")
                by_branch = {str(row["branch"]): row for row in branch_rows}
                actions = {branch: by_branch[branch]["action"] for branch in ("cond", "uncond")}
                aggregate_action = (
                    actions["cond"] if actions["cond"] == actions["uncond"] else "mixed"
                )
                threshold = float(by_branch["cond"]["requested_threshold"])
                if not math.isclose(
                    threshold,
                    float(by_branch["uncond"]["requested_threshold"]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    raise RuntimeError("CFG branches used different requested thresholds")
                step_decision = {
                    "requested_threshold": threshold,
                    "action": aggregate_action,
                    "reason": {
                        branch: by_branch[branch]["reason"] for branch in ("cond", "uncond")
                    },
                    "branches": actions,
                    "branch_decisions": by_branch,
                    "filtered_relative_l1": {
                        branch: by_branch[branch]["filtered_relative_l1"]
                        for branch in ("cond", "uncond")
                    },
                    "accumulated_distance_before": {
                        branch: by_branch[branch]["accumulated_distance_before"]
                        for branch in ("cond", "uncond")
                    },
                    "accumulated_distance_with_current": {
                        branch: by_branch[branch]["accumulated_distance_with_current"]
                        for branch in ("cond", "uncond")
                    },
                    "accumulated_distance_after": {
                        branch: by_branch[branch]["accumulated_distance_after"]
                        for branch in ("cond", "uncond")
                    },
                    "distance_reference": "previous_step_same_cfg_branch",
                    "distance_feature": "sea_filtered_first_block_modulated_input",
                    "distance_metric": "relative_l1_mean",
                    "native_forced_recompute": bool(
                        by_branch["cond"]["native_forced_recompute"]
                    ),
                }
                for branch in ("cond", "uncond"):
                    step_decision[f"{branch}_action"] = by_branch[branch]["action"]
                    step_decision[f"{branch}_filtered_relative_l1"] = by_branch[branch][
                        "filtered_relative_l1"
                    ]
                    for suffix in ("before", "with_current", "after"):
                        step_decision[f"{branch}_accumulated_distance_{suffix}"] = by_branch[
                            branch
                        ][f"accumulated_distance_{suffix}"]
                steps.append({
                    **self.step_metadata[index],
                    **step_decision,
                    **latent_rows[index],
                })
        else:
            if len(decisions) != NUM_STEPS:
                raise RuntimeError("baseline trace must contain 50 ordered step decisions")
            for index in range(NUM_STEPS):
                decision = decisions[index]
                if int(decision["step_index"]) != index:
                    raise RuntimeError("baseline trace decisions are not ordered 0..49")
                steps.append({**self.step_metadata[index], **decision, **latent_rows[index]})
        payload = {
            **self.trace_payload,
            "trajectory_id": self.trajectory_id,
            "manifest_record": self.manifest_record,
            "step_records": steps,
            "latent_save_dtype": str(latent_dtype),
            "artifact_io_outside_pipeline_generate_timing": True,
        }
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = trace_path.with_name(trace_path.name + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, trace_path)
        self.latents.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return payload


class Wan21DataRuntime:
    """Install one stable runtime mode on a reusable WanT2V pipeline."""

    def __init__(self, pipeline: Any, mode: str):
        if mode not in {"baseline", "candidate"}:
            raise ValueError("mode must be baseline or candidate")
        self.pipeline = pipeline
        self.mode = mode
        self.capture = RuntimeCapture(mode=mode)
        self.controller: Optional[RandomThresholdSeaCacheController] = None
        self.original_model_forward = pipeline.model.forward
        self.original_generate = pipeline.generate
        pipeline._ours_data_runtime = self
        pipeline.generate = types.MethodType(_capturing_t2v_generate, pipeline)
        if mode == "candidate":
            pipeline.model.forward = types.MethodType(_random_threshold_forward, pipeline.model)

    def configure_sample(self, record: dict[str, Any]) -> None:
        trajectory_id = str(record.get("trajectory_id") or record.get("sample_id") or "")
        if not trajectory_id:
            raise ValueError("sample record has no identity")
        self.capture.reset(trajectory_id, record)
        if self.mode == "candidate":
            threshold_path = record.get("threshold_path")
            if not isinstance(threshold_path, list) or len(threshold_path) != NUM_STEPS:
                raise ValueError("candidate requires a calibrated 50-step threshold path")
            self.controller = RandomThresholdSeaCacheController(
                RandomThresholdConfig(tuple(float(value) for value in threshold_path))
            )
            self.pipeline.model.ours_seacache_controller = self.controller
        else:
            self.controller = None

    def finish_trace(self, *, sample_solver: str, sampling_steps: int, shift: float, guide_scale: float, frame_num: int, size: tuple[int, int]) -> None:
        if self.mode == "candidate":
            if self.controller is None:
                raise RuntimeError("candidate controller was not configured")
            payload = self.controller.summary()
        else:
            decisions = [
                {
                    "step_index": step,
                    "requested_threshold": None,
                    "action": "recompute",
                    "reason": "full_compute_reference",
                    "relative_l1": None,
                    "accumulator_before": None,
                    "accumulator_after": None,
                    "native_forced_recompute": step in {0, NUM_STEPS - 1},
                    "branches": {"cond": "recompute", "uncond": "recompute"},
                }
                for step in range(NUM_STEPS)
            ]
            payload = {
                "schema": "ours4wan21_full_compute_trace_v1",
                "threshold_path": None,
                "forced_recompute_steps": [0, NUM_STEPS - 1],
                "total_steps": NUM_STEPS,
                "reuse": 0,
                "recompute": NUM_STEPS,
                "reuse_path": [],
                "recompute_path": list(range(NUM_STEPS)),
                "decisions": decisions,
            }
        payload.update({
            "implementation": self.mode,
            "task": "t2v-1.3B",
            "sampling_steps": int(sampling_steps),
            "sample_solver": sample_solver,
            "shift": float(shift),
            "guide_scale": float(guide_scale),
            "frame_num": int(frame_num),
            "size_wh": [int(size[0]), int(size[1])],
        })
        self.capture.trace_payload = payload


def _random_threshold_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
    ours_branch=None,
    ours_step_index=None,
    ours_num_steps=None,
):
    from wan.modules.model import sinusoidal_embedding_1d

    if self.model_type == "i2v" or clip_fea is not None or y is not None:
        raise ValueError("Ours4Wan21 data runtime supports T2V only")
    if ours_branch not in {"cond", "uncond"}:
        raise ValueError("candidate forward requires cond/uncond branch")
    if ours_step_index is None or ours_num_steps is None:
        raise ValueError("candidate forward requires step metadata")
    controller: RandomThresholdSeaCacheController = self.ours_seacache_controller
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)
    x = [self.patch_embedding(item.unsqueeze(0)) for item in x]
    grid_sizes = torch.stack([torch.tensor(item.shape[2:], dtype=torch.long) for item in x])
    x = [item.flatten(2).transpose(1, 2) for item in x]
    seq_lens = torch.tensor([item.size(1) for item in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([item, item.new_zeros(1, seq_len - item.size(1), item.size(2))], dim=1)
        for item in x
    ])
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
    context = self.text_embedding(torch.stack([
        torch.cat([item, item.new_zeros(self.text_len - item.size(0), item.size(1))])
        for item in context
    ]))
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=None,
    )
    e_first = (self.blocks[0].modulation + e0).chunk(6, dim=1)
    feature = self.blocks[0].norm1(x).float() * (1 + e_first[1]) + e_first[0]
    reuse = controller.plan_step(
        branch=ours_branch,
        step_index=ours_step_index,
        num_steps=ours_num_steps,
        feature=feature,
        grid_size=grid_sizes[0],
    )
    if reuse:
        x = x + controller.reuse_residual(ours_branch, ours_step_index)
    else:
        block_input = x.detach().clone()
        for block in self.blocks:
            x = block(x, **kwargs)
        controller.record_recompute(ours_branch, ours_step_index, x - block_input)
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    return [item.float() for item in x]


def _capturing_t2v_generate(
    self,
    input_prompt,
    size=(832, 480),
    frame_num=81,
    shift=5.0,
    sample_solver="unipc",
    sampling_steps=50,
    guide_scale=5.0,
    n_prompt="",
    seed=-1,
    offload_model=False,
):
    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    runtime: Wan21DataRuntime = self._ours_data_runtime
    if runtime.capture.trajectory_id == "":
        raise RuntimeError("runtime.configure_sample must run before generate")
    if tuple(size) != (832, 480) or frame_num != 81 or sampling_steps != 50:
        raise ValueError("runtime is frozen to 832x480/81f/50 steps")
    if sample_solver != "unipc" or shift != 5.0 or guide_scale != 5.0 or seed != 42:
        raise ValueError("runtime is frozen to UniPC/shift5/CFG5/seed42")
    if offload_model:
        raise ValueError("frozen Wan2.1 data protocol forbids model offload")

    target_shape = (
        self.vae.model.z_dim,
        (frame_num - 1) // self.vae_stride[0] + 1,
        size[1] // self.vae_stride[1],
        size[0] // self.vae_stride[2],
    )
    seq_len = math.ceil(
        target_shape[2] * target_shape[3]
        / (self.patch_size[1] * self.patch_size[2])
        * target_shape[1] / self.sp_size
    ) * self.sp_size
    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    generator = torch.Generator(device=self.device)
    generator.manual_seed(seed)
    if self.t5_cpu:
        raise ValueError("frozen Wan2.1 data protocol requires T5 on GPU")
    context = self.text_encoder([input_prompt], self.device)
    context_null = self.text_encoder([n_prompt], self.device)
    noise = [torch.randn(*target_shape, dtype=torch.float32, device=self.device, generator=generator)]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, "no_sync", noop_no_sync)
    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
        timesteps = sample_scheduler.timesteps
        if runtime.controller is not None:
            runtime.controller.reset()
            runtime.controller.set_scheduler_sigmas(getattr(sample_scheduler, "sigmas", None))
        latents = noise
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}
        sigmas = getattr(sample_scheduler, "sigmas", None)
        for step_index, timestep_value in enumerate(tqdm(timesteps)):
            runtime.capture.latents.append(latents[0])
            runtime.capture.step_metadata.append({
                "step_index": step_index,
                "step_fraction": step_index / max(sampling_steps - 1, 1),
                "timestep": float(timestep_value.detach().cpu().item()),
                "sigma": (
                    float(sigmas[step_index].detach().cpu().item())
                    if sigmas is not None and step_index < len(sigmas) else None
                ),
                "model_stage": "single",
            })
            timestep = torch.stack([timestep_value])
            self.model.to(self.device)
            if runtime.mode == "candidate":
                cond = self.model(
                    latents, t=timestep, ours_branch="cond",
                    ours_step_index=step_index, ours_num_steps=len(timesteps), **arg_c
                )[0]
                uncond = self.model(
                    latents, t=timestep, ours_branch="uncond",
                    ours_step_index=step_index, ours_num_steps=len(timesteps), **arg_null
                )[0]
            else:
                cond = self.model(latents, t=timestep, **arg_c)[0]
                uncond = self.model(latents, t=timestep, **arg_null)[0]
            noise_pred = uncond + guide_scale * (cond - uncond)
            updated = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                timestep_value,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=generator,
            )[0]
            latents = [updated.squeeze(0)]
        x0 = latents
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        videos = self.vae.decode(x0) if self.rank == 0 else None
    runtime.finish_trace(
        sample_solver=sample_solver,
        sampling_steps=sampling_steps,
        shift=shift,
        guide_scale=guide_scale,
        frame_num=frame_num,
        size=tuple(size),
    )
    del noise, latents, sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    return videos[0] if self.rank == 0 else None


def create_pipeline(wan21_root: Path, checkpoint_dir: Path) -> tuple[Any, float, Any]:
    validate_wan21_source(wan21_root)
    checkpoint_dir = checkpoint_dir.expanduser().resolve(strict=True)
    if "1.3B" not in checkpoint_dir.name:
        raise ValueError("checkpoint directory name must retain the 1.3B marker")
    sys.path.insert(0, str(wan21_root.resolve()))
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import cache_video

    config = WAN_CONFIGS["t2v-1.3B"]
    started = time.perf_counter()
    pipeline = wan.WanT2V(
        config=config,
        checkpoint_dir=str(checkpoint_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    if pipeline.param_dtype != torch.bfloat16 or pipeline.t5_cpu:
        raise RuntimeError("constructed pipeline violates BF16/T5-GPU protocol")
    if torch.cuda.is_available():
        torch.cuda.synchronize(pipeline.device)
    return pipeline, time.perf_counter() - started, cache_video


__all__ = [
    "PROTOCOL",
    "RuntimeCapture",
    "WAN21_COMMIT",
    "Wan21DataRuntime",
    "create_pipeline",
    "file_sha256",
    "validate_wan21_source",
]
