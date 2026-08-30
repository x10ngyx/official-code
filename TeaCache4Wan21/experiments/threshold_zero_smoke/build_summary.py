#!/usr/bin/env python3
"""Build the fixed-protocol TeaCache4Wan21 threshold-zero smoke summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flops", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--baseline-vbench", type=Path, required=True)
    parser.add_argument("--candidate-vbench", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    flops = read(args.flops)
    metrics = read(args.metrics)
    baseline_vbench = read(args.baseline_vbench)
    candidate_vbench = read(args.candidate_vbench)
    baseline = flops["runs"]["baseline"]
    candidate = flops["runs"]["teacache_threshold0"]
    payload = {
        "schema_version": 2,
        "protocol": {
            "model": "Wan2.1-T2V-1.3B",
            "video": {"width": 832, "height": 480, "frames": 81, "fps": 16},
            "sampling": {"steps": 50, "solver": "unipc", "shift": 5, "cfg": 5, "seed": 42},
            "offload_model": False,
            "t5_cpu": False,
        },
        "latency": {
            "headline": "pipeline_generate_wall_seconds",
            "baseline": baseline["pipeline_generate_wall_seconds"],
            "candidate": candidate["pipeline_generate_wall_seconds"],
            "components": {
                "baseline": {
                    "t5_cuda_seconds": baseline["t5_cuda_seconds"],
                    "dit_cuda_seconds": baseline["dit_cuda_seconds"],
                    "vae_decode_cuda_seconds": baseline["vae_decode_cuda_seconds"],
                },
                "candidate": {
                    "t5_cuda_seconds": candidate["t5_cuda_seconds"],
                    "dit_cuda_seconds": candidate["dit_cuda_seconds"],
                    "vae_decode_cuda_seconds": candidate["vae_decode_cuda_seconds"],
                },
            },
        },
        "tflops": {
            "headline": "estimated DiT TFLOPs",
            "baseline_dit": baseline["estimated_dit_tflops_per_video"],
            "candidate_dit": candidate["estimated_dit_tflops_per_video"],
            "t5": candidate["estimated_t5_tflops_per_video"],
            "vae_decode": candidate["estimated_vae_decode_tflops_per_video"],
        },
        "video_quality": {
            "protocol": metrics["protocol_id"],
            "metrics": metrics["metrics"],
            "baseline_vbench_score": baseline_vbench["vbench_score"],
            "candidate_vbench_score": candidate_vbench["vbench_score"],
            "vbench_protocol": candidate_vbench["protocol"],
            "vbench_warning": candidate_vbench["warning"],
        },
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
