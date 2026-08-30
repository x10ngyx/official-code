#!/usr/bin/env python3
"""Profile the fixed Wan2.2 T2V-A14B DiT paths with Calflops."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
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
EXPECTED_CALFLOPS_COMMIT = "027e89a24daf23ee7ed79ca4abee3fb59b5b23cd"
EXPECTED_CALFLOPS_VERSION = "0.3.2"

sys.path.insert(0, str(CALFLOPS_EVALUATION_DIR))
sys.path.insert(0, str(REPOSITORY_DIR / "ComponentMetrics"))
from calflops_eval import dense_attention_counts  # noqa: E402
from component_flops import profile_t5, profile_vae_decode  # noqa: E402


class WanForwardProfile(nn.Module):
    """Expose Wan's list-based forward as one tensor-only Calflops case."""

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


def profile_stage(
    *,
    model_class: type[nn.Module],
    checkpoint_dir: Path,
    subfolder: str,
    stage: str,
    timestep_value: float,
    latent: torch.Tensor,
    contexts: dict[str, torch.Tensor],
    seq_len: int,
    device: torch.device,
    calculate_flops_fn: Any,
) -> tuple[dict[str, Any], dict[str, int]]:
    model = model_class.from_pretrained(str(checkpoint_dir), subfolder=subfolder)
    model.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    block_count = len(model.blocks)
    if block_count < 1:
        raise RuntimeError(f"{stage} model has no Transformer blocks")
    timestep = torch.tensor([timestep_value], dtype=torch.float32, device=device)
    wrapper = WanForwardProfile(model, seq_len).to(device)

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
    branch_profiles = {}
    original_blocks = model.blocks
    for branch in ("cond", "uncond"):
        if branch not in contexts:
            raise ValueError(f"missing {branch} context for {stage} profile")
        inputs = (latent, timestep, contexts[branch])
        full_calflops = profile_case(wrapper, inputs, calculate_flops_fn)
        model.blocks = nn.ModuleList()
        try:
            always_on_calflops = profile_case(
                wrapper, inputs, calculate_flops_fn
            )
        finally:
            model.blocks = original_blocks
        estimated_full_flops = full_calflops["flops"] + attention_correction
        estimated_always_on_flops = always_on_calflops["flops"]
        branch_profiles[branch] = {
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
        }
    for key in ("estimated_full_flops", "estimated_always_on_flops"):
        if branch_profiles["cond"][key] != branch_profiles["uncond"][key]:
            raise RuntimeError(
                f"{stage} cond/uncond Calflops mismatch for {key}: "
                f"{branch_profiles['cond'][key]} != {branch_profiles['uncond'][key]}"
            )
    metadata = {
        "hidden_dim": int(model.dim),
        "num_heads": int(model.num_heads),
        "head_dim": int(head_dim),
        "text_tokens": int(model.text_len),
        "transformer_blocks": int(block_count),
    }
    result = {
        "checkpoint_subfolder": subfolder,
        "representative_timestep": timestep_value,
        "cfg_branch_equivalence": {
            "status": "pass",
            "checked_fields": [
                "estimated_full_flops",
                "estimated_always_on_flops",
            ],
            "reason": (
                "conditional and unconditional inputs have identical tensor "
                "shapes and produced identical independent Calflops profiles"
            ),
        },
        "branches": branch_profiles,
        "model": metadata,
    }
    del wrapper, model, timestep
    torch.cuda.empty_cache()
    return result, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan22-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--calflops-source",
        type=Path,
        help=(
            "Optional calculate-flops.pytorch checkout at locked commit 027e89a. "
            "Use when calflops==0.3.2 is not installed."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frame-num", type=int, default=45)
    return parser.parse_args()


def load_calflops(source: Path | None) -> tuple[Any, dict[str, Any]]:
    configured = source or (
        Path(os.environ["CALFLOPS_SOURCE"]) if os.environ.get("CALFLOPS_SOURCE") else None
    )
    source_metadata: dict[str, Any] = {}
    if configured is not None:
        resolved = configured.expanduser().resolve(strict=True)
        if not (resolved / "calflops" / "__init__.py").is_file():
            raise FileNotFoundError(f"not a Calflops source checkout: {resolved}")
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=resolved,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        if commit != EXPECTED_CALFLOPS_COMMIT:
            raise ValueError(
                f"Calflops commit mismatch: expected {EXPECTED_CALFLOPS_COMMIT}, got {commit}"
            )
        sys.path.insert(0, str(resolved))
        source_metadata = {"source_checkout": str(resolved), "commit": commit}
    try:
        from calflops import calculate_flops
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "calflops is unavailable; install calflops==0.3.2 or pass "
            "--calflops-source /path/to/calculate-flops.pytorch@027e89a"
        ) from exc
    if source_metadata:
        version = EXPECTED_CALFLOPS_VERSION
    else:
        try:
            version = importlib.metadata.version("calflops")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    if version != EXPECTED_CALFLOPS_VERSION:
        raise ValueError(
            f"Calflops version mismatch: expected {EXPECTED_CALFLOPS_VERSION}, got {version}"
        )
    return calculate_flops, {"version": version, **source_metadata}


def main() -> None:
    args = parse_args()
    args.wan22_root = args.wan22_root.expanduser().resolve(strict=True)
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve(strict=True)
    args.output = require_external_output(args.output)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    if (args.width, args.height, args.frame_num) != (832, 480, 45):
        raise ValueError(
            "the locked Wan2.2-T2V-A14B profile is 832x480 with 45 frames"
        )
    prepared_manifest_path = args.wan22_root / ".teacache4wan22_prepared.json"
    if not prepared_manifest_path.is_file():
        raise FileNotFoundError(
            f"Wan2.2 source is not prepared by TeaCache4Wan22: {args.wan22_root}"
        )
    prepared_manifest = json.loads(
        prepared_manifest_path.read_text(encoding="utf-8")
    )
    if prepared_manifest.get("status") != "pass" or prepared_manifest.get(
        "mode"
    ) != "prepared":
        raise ValueError("invalid TeaCache4Wan22 prepared-source manifest")
    if not torch.cuda.is_available():
        raise RuntimeError("Wan2.2 Calflops profiling requires CUDA")

    calculate_flops_fn, calflops_metadata = load_calflops(args.calflops_source)

    sys.path.insert(0, str(args.wan22_root))
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wan.modules.t5 import umt5_xxl
    from wan.modules.vae2_1 import Wan2_1_VAE

    device = torch.device("cuda:0")
    latent_frames = (args.frame_num - 1) // 4 + 1
    latent_height = args.height // 8
    latent_width = args.width // 8
    patch_size = (1, 2, 2)
    patch_tokens = math.prod(
        (
            latent_frames // patch_size[0],
            latent_height // patch_size[1],
            latent_width // patch_size[2],
        )
    )
    latent = torch.zeros(
        16,
        latent_frames,
        latent_height,
        latent_width,
        dtype=torch.float32,
        device=device,
    )
    contexts = {
        branch: torch.zeros(
            512,
            4096,
            dtype=torch.bfloat16,
            device=device,
        )
        for branch in ("cond", "uncond")
    }

    stages = {}
    stage_metadata = {}
    for stage, subfolder, timestep in (
        ("high", "high_noise_model", 900.0),
        ("low", "low_noise_model", 100.0),
    ):
        stages[stage], stage_metadata[stage] = profile_stage(
            model_class=WanModel,
            checkpoint_dir=args.checkpoint_dir,
            subfolder=subfolder,
            stage=stage,
            timestep_value=timestep,
            latent=latent,
            contexts=contexts,
            seq_len=patch_tokens,
            device=device,
            calculate_flops_fn=calculate_flops_fn,
        )
    if stage_metadata["high"] != stage_metadata["low"]:
        raise RuntimeError(
            "Wan2.2 high/low DiT architectures differ; aggregation assumptions are invalid"
        )
    for branch in ("cond", "uncond"):
        for key in ("estimated_full_flops", "estimated_always_on_flops"):
            high_value = stages["high"]["branches"][branch][key]
            low_value = stages["low"]["branches"][branch][key]
            if high_value != low_value:
                raise RuntimeError(
                    f"Wan2.2 high/low Calflops mismatch for {branch}/{key}: "
                    f"{high_value} != {low_value}"
                )

    config = WAN_CONFIGS["t2v-A14B"]
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
    vae = Wan2_1_VAE(
        vae_pth=str(args.checkpoint_dir / config.vae_checkpoint),
        device=device,
    )
    vae_profile = profile_vae_decode(
        model=vae.model,
        scale=vae.scale,
        latent_shape=(1, 16, latent_frames, latent_height, latent_width),
        device=device,
        calculate_flops_fn=calculate_flops_fn,
    )
    component_profiles = {"t5": t5_profile, "vae_decode": vae_profile}

    payload = {
        "schema": "teacache4wan22_calflops_profile_v2",
        "schema_version": 2,
        "tool": {
            "name": "calflops",
            **calflops_metadata,
            "repository_evaluator": str(CALFLOPS_EVALUATION_DIR),
        },
        "scope": (
            "Wan2.2 high/low DiT forward plus separately profiled UMT5 encoder "
            "and VAE decode. DiT uses Calflops-observed operators plus manual "
            "dense FlashAttention-core correction; scheduler, MP4 export, and "
            "the TeaCache controller/residual add remain excluded."
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
            "wan22_root": str(args.wan22_root),
            "prepared_manifest_path": str(prepared_manifest_path),
            "prepared_manifest_sha256": sha256(prepared_manifest_path),
            "prepared_manifest": prepared_manifest,
            "checkpoint_dir": str(args.checkpoint_dir),
        },
        "input": {
            "task": "t2v-A14B",
            "video_shape_fhw": [args.frame_num, args.height, args.width],
            "output_fps": 16,
            "sampling_steps": 50,
            "solver": "dpm++",
            "shift": 12.0,
            "guide_scale_low_high": [3.0, 4.0],
            "boundary": 0.875,
            "seed": 42,
            "parameter_dtype": "bfloat16",
            "latent_shape_cfhw": [
                16,
                latent_frames,
                latent_height,
                latent_width,
            ],
            "patch_tokens": patch_tokens,
            "seq_len": patch_tokens,
            "stage_steps": {"high": 32, "low": 18},
            **stage_metadata["high"],
        },
        "stages": stages,
        "component_profiles": component_profiles,
        "stage_equivalence": {
            "status": "pass",
            "checked_branches": ["cond", "uncond"],
            "checked_fields": [
                "estimated_full_flops",
                "estimated_always_on_flops",
            ],
        },
        "warnings": [
            "Calflops does not observe the custom FlashAttention CUDA kernel; dense attention core FLOPs are added analytically.",
            "T5 is reported for two encoder calls per video and VAE for one decode; tokenizer and scheduler FLOPs are outside these component counts.",
            "TeaCache controller and residual-add FLOPs are excluded from the DiT headline and from complete-method claims.",
            "TFLOP/s derived later is achieved estimated DiT throughput, not vendor peak throughput.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
