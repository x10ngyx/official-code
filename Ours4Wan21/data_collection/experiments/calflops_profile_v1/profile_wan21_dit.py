#!/usr/bin/env python3
"""Profile locked Wan2.1-1.3B DiT full/reuse costs at the collection shape."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PROJECT = SCRIPT_DIR.parents[1]
OFFICIAL_CODE = SCRIPT_DIR.parents[3]
CALFLOPS_EVALUATION = OFFICIAL_CODE / "CalflopsEvaluation"
TFLOP_DIVISOR = 1_000_000_000_000.0
THREAD_KEYS = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)

sys.path.insert(0, str(DATA_PROJECT / "src"))
sys.path.insert(0, str(CALFLOPS_EVALUATION))
sys.path.insert(0, str(OFFICIAL_CODE / "ComponentMetrics"))
from calflops_eval import dense_attention_counts  # noqa: E402
from calflops_loader import load_calflops  # noqa: E402
from component_flops import profile_t5, profile_vae_decode  # noqa: E402
from ours4wan21_data.manifest import PROTOCOL  # noqa: E402
from ours4wan21_data.paths import require_result_path  # noqa: E402
from ours4wan21_data.source_lock import WAN21_COMMIT, validate_wan21_source  # noqa: E402


class WanForwardProfile(nn.Module):
    def __init__(self, model: nn.Module, seq_len: int) -> None:
        super().__init__()
        self.wan_model = model
        self.seq_len = seq_len

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            return self.wan_model(
                [latent], t=timestep, context=[context], seq_len=self.seq_len
            )


def require_threads() -> dict[str, str | None]:
    values = {key: os.environ.get(key) for key in THREAD_KEYS}
    invalid = {key: value for key, value in values.items() if value != "1"}
    if invalid:
        raise RuntimeError(f"all BLAS thread limits must equal one: {invalid}")
    return values


def profile_case(wrapper: nn.Module, inputs: tuple[torch.Tensor, ...], calculate_flops: Any) -> dict[str, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    flops, macs, params = calculate_flops(
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan21-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--calflops-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    threads = require_threads()
    wan_root = args.wan21_root.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint_dir.expanduser().resolve(strict=True)
    output = require_result_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if "1.3B" not in checkpoint.name:
        raise ValueError("checkpoint directory name must retain the 1.3B marker")
    if not torch.cuda.is_available():
        raise RuntimeError("Wan2.1 Calflops profiling requires CUDA")
    source_hashes = validate_wan21_source(wan_root)
    calculate_flops, calflops_metadata = load_calflops(args.calflops_source)

    sys.path.insert(0, str(wan_root))
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wan.modules.t5 import umt5_xxl
    from wan.modules.vae import WanVAE

    device = torch.device("cuda:0")
    model = WanModel.from_pretrained(str(checkpoint))
    model.eval().requires_grad_(False).to(device)
    block_count = len(model.blocks)
    latent_frames, latent_height, latent_width = 21, 60, 104
    patch_tokens = math.prod((
        latent_frames // model.patch_size[0],
        latent_height // model.patch_size[1],
        latent_width // model.patch_size[2],
    ))
    latent = torch.zeros(
        model.in_dim, latent_frames, latent_height, latent_width,
        dtype=torch.float32, device=device,
    )
    timestep = torch.tensor([500.0], dtype=torch.float32, device=device)
    context = torch.zeros(model.text_len, model.text_dim, dtype=torch.bfloat16, device=device)
    wrapper = WanForwardProfile(model, patch_tokens).to(device)
    inputs = (latent, timestep, context)
    full = profile_case(wrapper, inputs, calculate_flops)
    original_blocks = model.blocks
    model.blocks = nn.ModuleList()
    try:
        always_on = profile_case(wrapper, inputs, calculate_flops)
    finally:
        model.blocks = original_blocks

    head_dim = model.dim // model.num_heads
    self_attention = dense_attention_counts(
        batch_size=1, query_tokens=patch_tokens, key_value_tokens=patch_tokens,
        num_heads=model.num_heads, head_dim=head_dim,
    )
    cross_attention = dense_attention_counts(
        batch_size=1, query_tokens=patch_tokens, key_value_tokens=model.text_len,
        num_heads=model.num_heads, head_dim=head_dim,
    )
    attention_correction = block_count * (self_attention["flops"] + cross_attention["flops"])
    estimated_full = full["flops"] + attention_correction
    estimated_always_on = always_on["flops"]
    if not 0.0 < estimated_always_on < estimated_full:
        raise RuntimeError("invalid full/always-on profile ordering")
    config = WAN_CONFIGS["t2v-1.3B"]
    t5_model = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=config.t5_dtype,
        device=device,
    ).eval().requires_grad_(False)
    t5_model.load_state_dict(
        torch.load(checkpoint / config.t5_checkpoint, map_location="cpu")
    )
    t5_profile = profile_t5(
        model=t5_model,
        text_tokens=int(config.text_len),
        device=device,
        calculate_flops_fn=calculate_flops,
    )
    del t5_model
    torch.cuda.empty_cache()
    vae = WanVAE(
        vae_pth=str(checkpoint / config.vae_checkpoint),
        device=device,
    )
    vae_profile = profile_vae_decode(
        model=vae.model,
        scale=vae.scale,
        latent_shape=(1, model.in_dim, latent_frames, latent_height, latent_width),
        device=device,
        calculate_flops_fn=calculate_flops,
    )
    component_profiles = {"t5": t5_profile, "vae_decode": vae_profile}
    payload = {
        "schema": "ours4wan21_calflops_profile_v2",
        "schema_version": 2,
        "protocol": PROTOCOL,
        "tool": {
            "name": "calflops",
            **calflops_metadata,
            "evaluation_package": str(CALFLOPS_EVALUATION.resolve()),
            "manual_attention_formula": "calflops_eval.dense_attention_counts",
        },
        "scope": (
            "Wan2.1 DiT forward plus separately profiled UMT5 encoder and VAE "
            "decode. DiT uses Calflops-observed modules plus manual dense "
            "FlashAttention-core correction; scheduler, MP4/file I/O, and "
            "controller/FFT overhead remain excluded."
        ),
        "counting_convention": {
            "mac_to_flop": 2,
            "tflop_divisor": TFLOP_DIVISOR,
            "tflops": "operation count, not throughput",
        },
        "source": {
            "wan21_root": str(wan_root),
            "wan21_commit": WAN21_COMMIT,
            "locked_file_sha256": source_hashes,
            "checkpoint_dir": str(checkpoint),
        },
        "input": {
            "video_shape_fhw": [81, 480, 832],
            "latent_shape_cfhw": [model.in_dim, latent_frames, latent_height, latent_width],
            "patch_tokens": patch_tokens,
            "seq_len": patch_tokens,
            "text_tokens": model.text_len,
            "hidden_dim": model.dim,
            "num_heads": model.num_heads,
            "head_dim": head_dim,
            "transformer_blocks": block_count,
        },
        "per_model_forward": {
            "transformer_blocks": block_count,
            "calflops_full": full,
            "calflops_always_on_without_blocks": always_on,
            "manual_flash_attention": {
                "self_attention_per_block": self_attention,
                "cross_attention_per_block": cross_attention,
                "block_count": block_count,
                "total_flops": attention_correction,
                "total_tflops": attention_correction / TFLOP_DIVISOR,
            },
            "estimated_full_flops": estimated_full,
            "estimated_full_tflops": estimated_full / TFLOP_DIVISOR,
            "estimated_always_on_flops": estimated_always_on,
            "estimated_always_on_tflops": estimated_always_on / TFLOP_DIVISOR,
            "estimated_transformer_blocks_flops": estimated_full - estimated_always_on,
        },
        "component_profiles": component_profiles,
        "thread_environment": threads,
        "warnings": [
            "Custom FlashAttention cores are added analytically because Calflops hooks cannot observe them.",
            "T5 is reported for two encoder calls per video and VAE for one decode; tokenizer and scheduler FLOPs are outside these component counts.",
            "Controller and spectral-filter FLOPs are excluded and should be disclosed separately from DiT TFLOPs.",
            "TFLOP/s later derived from CUDA spans is diagnostic achieved throughput, not vendor peak throughput.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
