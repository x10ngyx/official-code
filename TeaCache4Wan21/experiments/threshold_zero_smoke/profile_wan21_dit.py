#!/usr/bin/env python3
"""Profile Wan2.1 DiT FLOPs with Calflops and actual inference call traces."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPOSITORY_DIR / "CalflopsEvaluation"))
sys.path.insert(0, str(REPOSITORY_DIR / "ComponentMetrics"))
from calflops_eval import dense_attention_counts  # noqa: E402
from calflops_loader import load_calflops  # noqa: E402
from component_flops import profile_t5, profile_vae_decode  # noqa: E402
from reporting import extract_component_latency, extract_component_tflops  # noqa: E402


TFLOP_DIVISOR = 1_000_000_000_000


class WanForwardProfile(nn.Module):
    """Tensor-only wrapper around the list-based WanModel forward signature."""

    def __init__(self, model: nn.Module, seq_len: int) -> None:
        super().__init__()
        self.wan_model = model
        self.seq_len = seq_len

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> list[torch.Tensor]:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            return self.wan_model(
                [latent],
                t=timestep,
                context=[context],
                seq_len=self.seq_len,
            )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def profile_case(
    wrapper: nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
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


def summarize_run(
    timing: dict[str, Any],
    process_timing: dict[str, Any],
    *,
    always_on_flops: float,
    full_forward_flops: float,
    block_count: int,
    component_tflops: dict[str, float],
) -> dict[str, Any]:
    component_latency = extract_component_latency(timing)
    calls = timing.get("calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("timing JSON must contain non-empty calls")
    total_flops = 0.0
    for call in calls:
        executed = int(call["blocks_executed"])
        if executed < 0 or executed > block_count:
            raise ValueError(f"invalid blocks_executed={executed}")
        total_flops += always_on_flops + (full_forward_flops - always_on_flops) * (
            executed / block_count
        )
    cuda_seconds = timing.get("model_forward_cuda_seconds")
    throughput = (
        total_flops / float(cuda_seconds) / TFLOP_DIVISOR
        if isinstance(cuda_seconds, (int, float)) and cuda_seconds > 0
        else None
    )
    return {
        "process_wall_seconds": process_timing["process_wall_seconds"],
        "pipeline_init_wall_seconds": timing["pipeline_init_wall_seconds"],
        "pipeline_generate_wall_seconds": timing["pipeline_generate_wall_seconds"],
        "model_forward_cuda_seconds": cuda_seconds,
        "model_forward_call_count": timing["model_forward_call_count"],
        "full_compute_forward_calls": timing["full_compute_forward_calls"],
        "reuse_forward_calls": timing["reuse_forward_calls"],
        "estimated_dit_flops": total_flops,
        "estimated_dit_tflops_per_video": total_flops / TFLOP_DIVISOR,
        **component_latency,
        **component_tflops,
        "estimated_achieved_tflops_per_second": throughput,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan21-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--baseline-timing", type=Path, required=True)
    parser.add_argument("--candidate-timing", type=Path, required=True)
    parser.add_argument("--baseline-process-timing", type=Path, required=True)
    parser.add_argument("--candidate-process-timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calflops-source", type=Path)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frame-num", type=int, default=81)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    if (args.width, args.height, args.frame_num) != (832, 480, 81):
        raise ValueError("the fixed Wan2.1 protocol requires 832x480 and 81 frames")

    args.wan21_root = args.wan21_root.resolve(strict=True)
    args.checkpoint_dir = args.checkpoint_dir.resolve(strict=True)
    sys.path.insert(0, str(args.wan21_root))
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wan.modules.t5 import umt5_xxl
    from wan.modules.vae import WanVAE

    if not torch.cuda.is_available():
        raise RuntimeError("Calflops Wan profiling requires CUDA")
    calculate_flops_fn, calflops_metadata = load_calflops(args.calflops_source)
    device = torch.device("cuda:0")
    model = WanModel.from_pretrained(str(args.checkpoint_dir))
    model.eval().requires_grad_(False).to(device)
    block_count = len(model.blocks)
    latent_frames = (args.frame_num - 1) // 4 + 1
    latent_height = args.height // 8
    latent_width = args.width // 8
    patch_tokens = math.prod(
        (
            latent_frames // model.patch_size[0],
            latent_height // model.patch_size[1],
            latent_width // model.patch_size[2],
        )
    )
    seq_len = patch_tokens
    latent = torch.zeros(
        model.in_dim,
        latent_frames,
        latent_height,
        latent_width,
        dtype=torch.float32,
        device=device,
    )
    timestep = torch.tensor([500.0], dtype=torch.float32, device=device)
    context = torch.zeros(
        model.text_len,
        model.text_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    wrapper = WanForwardProfile(model, seq_len).to(device)
    inputs = (latent, timestep, context)

    full_calflops = profile_case(wrapper, inputs, calculate_flops_fn)
    original_blocks = model.blocks
    model.blocks = nn.ModuleList()
    try:
        always_on_calflops = profile_case(wrapper, inputs, calculate_flops_fn)
    finally:
        model.blocks = original_blocks

    head_dim = model.dim // model.num_heads
    self_attention = dense_attention_counts(
        batch_size=1,
        query_tokens=seq_len,
        key_value_tokens=seq_len,
        num_heads=model.num_heads,
        head_dim=head_dim,
    )
    cross_attention = dense_attention_counts(
        batch_size=1,
        query_tokens=seq_len,
        key_value_tokens=model.text_len,
        num_heads=model.num_heads,
        head_dim=head_dim,
    )
    attention_correction = block_count * (
        self_attention["flops"] + cross_attention["flops"]
    )
    full_forward_flops = full_calflops["flops"] + attention_correction
    always_on_flops = always_on_calflops["flops"]

    config = WAN_CONFIGS["t2v-1.3B"]
    t5_model = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=config.t5_dtype,
        device=device,
    ).eval().requires_grad_(False)
    t5_model.load_state_dict(
        torch.load(args.checkpoint_dir / config.t5_checkpoint, map_location="cpu")
    )
    t5_profile = profile_t5(
        model=t5_model,
        text_tokens=int(config.text_len),
        device=device,
        calculate_flops_fn=calculate_flops_fn,
    )
    del t5_model
    torch.cuda.empty_cache()
    vae = WanVAE(
        vae_pth=str(args.checkpoint_dir / config.vae_checkpoint),
        device=device,
    )
    vae_profile = profile_vae_decode(
        model=vae.model,
        scale=vae.scale,
        latent_shape=(1, model.in_dim, latent_frames, latent_height, latent_width),
        device=device,
        calculate_flops_fn=calculate_flops_fn,
    )
    component_profiles = {"t5": t5_profile, "vae_decode": vae_profile}
    component_tflops = extract_component_tflops(
        {"component_profiles": component_profiles}
    )

    baseline_timing = read_json(args.baseline_timing)
    candidate_timing = read_json(args.candidate_timing)
    baseline_process = read_json(args.baseline_process_timing)
    candidate_process = read_json(args.candidate_process_timing)
    payload = {
        "schema_version": 2,
        "tool": {
            "name": "calflops",
            **calflops_metadata,
        },
        "scope": (
            "Wan2.1 DiT forward plus separately profiled UMT5 and VAE decode. "
            "The DiT count contains Calflops-observed operators plus a manual "
            "dense FlashAttention-core correction."
        ),
        "counting_convention": {
            "mac_to_flop": 2,
            "tflop_divisor": TFLOP_DIVISOR,
            "tflops": "operation count",
            "tflops_per_second": "estimated operation count divided by measured CUDA time",
        },
        "input": {
            "task": "t2v-1.3B",
            "video_shape_fhw": [args.frame_num, args.height, args.width],
            "latent_shape_cfhw": [
                model.in_dim,
                latent_frames,
                latent_height,
                latent_width,
            ],
            "patch_tokens": patch_tokens,
            "seq_len": seq_len,
            "text_tokens": model.text_len,
            "hidden_dim": model.dim,
            "num_heads": model.num_heads,
            "head_dim": head_dim,
            "transformer_blocks": block_count,
            "dtype": "bfloat16 autocast",
        },
        "per_model_forward": {
            "calflops_full": full_calflops,
            "calflops_always_on_without_blocks": always_on_calflops,
            "manual_flash_attention": {
                "self_attention_per_block": self_attention,
                "cross_attention_per_block": cross_attention,
                "block_count": block_count,
                "total_flops": attention_correction,
                "total_tflops": attention_correction / TFLOP_DIVISOR,
            },
            "estimated_full_flops": full_forward_flops,
            "estimated_full_tflops": full_forward_flops / TFLOP_DIVISOR,
            "estimated_always_on_flops": always_on_flops,
            "estimated_always_on_tflops": always_on_flops / TFLOP_DIVISOR,
        },
        "component_profiles": component_profiles,
        "runs": {
            "baseline": summarize_run(
                baseline_timing,
                baseline_process,
                always_on_flops=always_on_flops,
                full_forward_flops=full_forward_flops,
                block_count=block_count,
                component_tflops=component_tflops,
            ),
            "teacache_threshold0": summarize_run(
                candidate_timing,
                candidate_process,
                always_on_flops=always_on_flops,
                full_forward_flops=full_forward_flops,
                block_count=block_count,
                component_tflops=component_tflops,
            ),
        },
        "warnings": [
            "Calflops does not observe the custom FlashAttention CUDA kernel; dense attention core FLOPs are added analytically.",
            "TFLOP/s is an achieved estimate under this small smoke shape, not the GPU vendor peak specification.",
            "Pipeline and process latency are single-run cold-start measurements; use repeated warm runs for benchmarking claims.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
