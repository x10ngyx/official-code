#!/usr/bin/env python3
"""Profile the fixed Wan2.1-1.3B VBench200 DiT shape with Calflops."""

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
CALFLOPS_EVALUATION_DIR = REPOSITORY_DIR / "CalflopsEvaluation"
EXP_ROOT = Path("/all/yiran07-disk3/huteng_data/exp").resolve()
TFLOP_DIVISOR = 1_000_000_000_000

sys.path.insert(0, str(CALFLOPS_EVALUATION_DIR))
sys.path.insert(0, str(REPOSITORY_DIR / "ComponentMetrics"))
from calflops_eval import dense_attention_counts  # noqa: E402
from calflops_loader import load_calflops  # noqa: E402
from component_flops import profile_t5, profile_vae_decode  # noqa: E402


class WanForwardProfile(nn.Module):
    """Expose Wan's list-based forward as a tensor-only Calflops case."""

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
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.wan_model(
                [latent],
                t=timestep,
                context=[context],
                seq_len=self.seq_len,
            )


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_external_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as exc:
        raise ValueError(f"output must be below {EXP_ROOT}: {resolved}") from exc
    return resolved


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan21-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calflops-source", type=Path)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frame-num", type=int, default=81)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.wan21_root = args.wan21_root.expanduser().resolve(strict=True)
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve(strict=True)
    args.output = require_external_output(args.output)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    if (args.width, args.height, args.frame_num) != (832, 480, 81):
        raise ValueError(
            "the locked Wan2.1-1.3B benchmark profile is 832x480 with 81 frames"
        )
    if "1.3B" not in str(args.checkpoint_dir):
        raise ValueError("checkpoint path must identify Wan2.1-T2V-1.3B")
    if not torch.cuda.is_available():
        raise RuntimeError("Wan Calflops profiling requires CUDA")
    calculate_flops_fn, calflops_metadata = load_calflops(args.calflops_source)

    sys.path.insert(0, str(args.wan21_root))
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wan.modules.t5 import umt5_xxl
    from wan.modules.vae import WanVAE

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
    wrapper = WanForwardProfile(model, patch_tokens).to(device)
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
        query_tokens=patch_tokens,
        key_value_tokens=patch_tokens,
        num_heads=model.num_heads,
        head_dim=head_dim,
    )
    cross_attention = dense_attention_counts(
        batch_size=1,
        query_tokens=patch_tokens,
        key_value_tokens=model.text_len,
        num_heads=model.num_heads,
        head_dim=head_dim,
    )
    attention_correction = block_count * (
        self_attention["flops"] + cross_attention["flops"]
    )
    estimated_full_flops = full_calflops["flops"] + attention_correction
    estimated_always_on_flops = always_on_calflops["flops"]

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

    payload = {
        "schema_version": 2,
        "tool": {
            "name": "calflops",
            **calflops_metadata,
            "repository_evaluator": str(CALFLOPS_EVALUATION_DIR),
        },
        "scope": (
            "Wan2.1 DiT forward plus separately profiled UMT5 encoder and VAE "
            "decode. DiT uses Calflops-observed operators plus a manual dense "
            "FlashAttention-core correction; scheduler, MP4 export, and the "
            "TeaCache polynomial controller remain excluded."
        ),
        "counting_convention": {
            "mac_to_flop": 2,
            "tflop_divisor": TFLOP_DIVISOR,
            "tflops": "operation count, not throughput",
            "tflops_per_second": (
                "estimated operation count divided by measured DiT CUDA time"
            ),
        },
        "source": {
            "wan21_root": str(args.wan21_root),
            "wan21_generate_sha256": sha256(args.wan21_root / "generate.py"),
            "checkpoint_dir": str(args.checkpoint_dir),
        },
        "input": {
            "task": "t2v-1.3B",
            "video_shape_fhw": [args.frame_num, args.height, args.width],
            "output_fps": 16,
            "sampling_steps": 50,
            "solver": "unipc",
            "shift": 5.0,
            "cfg": 5.0,
            "seed": 42,
            "parameter_dtype": "bfloat16",
            "latent_shape_cfhw": [
                model.in_dim,
                latent_frames,
                latent_height,
                latent_width,
            ],
            "patch_tokens": patch_tokens,
            "seq_len": patch_tokens,
            "text_tokens": model.text_len,
            "hidden_dim": model.dim,
            "num_heads": model.num_heads,
            "head_dim": head_dim,
            "transformer_blocks": block_count,
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
            "estimated_full_flops": estimated_full_flops,
            "estimated_full_tflops": estimated_full_flops / TFLOP_DIVISOR,
            "estimated_always_on_flops": estimated_always_on_flops,
            "estimated_always_on_tflops": (
                estimated_always_on_flops / TFLOP_DIVISOR
            ),
            "estimated_transformer_blocks_flops": (
                estimated_full_flops - estimated_always_on_flops
            ),
        },
        "component_profiles": component_profiles,
        "warnings": [
            "Calflops does not observe the custom FlashAttention CUDA kernel; dense attention core FLOPs are added analytically.",
            "T5 is reported for two encoder calls per video and VAE for one decode; tokenizer and scheduler FLOPs are outside these component counts.",
            "The TeaCache controller is excluded from the DiT headline and from complete-method claims.",
            "TFLOP/s derived later is achieved estimated DiT throughput, not vendor peak throughput.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
