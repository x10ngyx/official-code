#!/usr/bin/env python3
"""Write the reproducibility manifest for the threshold-zero smoke test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan21-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--entrypoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    payload = {
        "schema_version": 1,
        "purpose": "Wan2.2 conda compatibility and TeaCache threshold=0 equivalence smoke test",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "calflops": importlib.metadata.version("calflops"),
        "calflops_evaluation": importlib.metadata.version(
            "cache-calflops-evaluation"
        ),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_device_count": torch.cuda.device_count(),
        "cuda_current_device": device,
        "cuda_device_name": torch.cuda.get_device_name(device) if device is not None else None,
        "wan21_root": str(args.wan21_root.resolve()),
        "wan21_commit": git_head(args.wan21_root),
        "wan21_generate_sha256": sha256(args.wan21_root / "generate.py"),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "entrypoint": str(args.entrypoint.resolve()),
        "entrypoint_sha256": sha256(args.entrypoint),
        "configuration": {
            "task": "t2v-1.3B",
            "size": "832*480",
            "frame_num": 5,
            "sample_solver": "unipc",
            "sample_steps": 4,
            "sample_shift": 5.0,
            "sample_guide_scale": 5.0,
            "base_seed": 42,
            "offload_model": True,
            "t5_cpu": True,
            "teacache_threshold": 0.0,
            "use_ret_steps": False,
            "prompt": args.prompt,
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
