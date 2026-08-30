#!/usr/bin/env python3
"""Build the fixed-protocol TeaCache4Wan22 smoke summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--baseline-vbench", type=Path, required=True)
    parser.add_argument("--candidate-vbench", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    performance = read(args.performance)
    metrics = read(args.metrics)
    baseline_vbench = read(args.baseline_vbench)
    candidate_vbench = read(args.candidate_vbench)
    baseline = performance["conditions"]["baseline"]
    candidate = performance["conditions"]["teacache"]
    payload = {
        "schema_version": 2,
        "protocol": performance["protocol"],
        "latency": {
            "headline": "pipeline_generate_wall_seconds",
            "baseline": baseline["pipeline_generate_wall_seconds"],
            "candidate": candidate["pipeline_generate_wall_seconds"],
            "comparison": performance["comparison"],
            "components": {
                "baseline": {
                    "t5_cuda_seconds": baseline["t5_cuda_seconds"],
                    "dit_cuda_seconds": baseline["dit_forward_cuda_seconds"],
                    "vae_decode_cuda_seconds": baseline["vae_decode_cuda_seconds"],
                },
                "candidate": {
                    "t5_cuda_seconds": candidate["t5_cuda_seconds"],
                    "dit_cuda_seconds": candidate["dit_forward_cuda_seconds"],
                    "vae_decode_cuda_seconds": candidate["vae_decode_cuda_seconds"],
                },
            },
        },
        "tflops": {
            "headline": "estimated DiT TFLOPs",
            "baseline_dit_total": baseline["estimated_dit_total_tflops"],
            "candidate_dit_total": candidate["estimated_dit_total_tflops"],
            "t5_per_video": candidate["estimated_t5_tflops_per_video"],
            "vae_decode_per_video": candidate[
                "estimated_vae_decode_tflops_per_video"
            ],
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
