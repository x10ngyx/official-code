#!/usr/bin/env python3
"""Build one headline report from performance, VideoMetrics, and VBench outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXP_ROOT = Path("/mnt/hdd/xiongyuxiang/tmp/exp").resolve()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def require_external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as exc:
        raise ValueError(f"path must be below {EXP_ROOT}: {resolved}") from exc
    return resolved


def mean_metric(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get("metrics", {}).get(name, {}).get("mean")
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing VideoMetrics mean: {name}")
    return float(value)


def aggregate_scores(payload: dict[str, Any]) -> dict[str, float]:
    values = payload.get("aggregate_scores")
    if not isinstance(values, dict):
        raise ValueError("missing VBench aggregate_scores")
    required = ("quality_score", "semantic_score", "total_score")
    return {name: float(values[name]) for name in required}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result_dir = require_external(args.result_dir)
    output_json = result_dir / "benchmark_report.json"
    output_markdown = result_dir / "benchmark_report.md"
    if output_json.exists() or output_markdown.exists():
        raise FileExistsError(f"refusing to overwrite report under {result_dir}")

    config = read_json(result_dir / "run_config.json")
    performance = read_json(result_dir / "performance" / "summary.json")
    metrics = read_json(result_dir / "evaluation" / "video_metrics" / "summary.json")
    vbench_reference = read_json(
        result_dir
        / "evaluation"
        / "vbench_reference"
        / "vbench200_aggregate_scores.json"
    )
    vbench_candidate = read_json(
        result_dir
        / "evaluation"
        / "vbench_candidate"
        / "vbench200_aggregate_scores.json"
    )
    baseline = performance["conditions"]["baseline"]
    teacache = performance["conditions"]["teacache"]
    baseline_latency = baseline["end_to_end_inference_latency_seconds"]
    teacache_latency = teacache["end_to_end_inference_latency_seconds"]
    baseline_tflops = baseline["estimated_dit_tflops_per_video"]
    teacache_tflops = teacache["estimated_dit_tflops_per_video"]

    payload = {
        "schema_version": 1,
        "model": "Wan2.1-T2V-1.3B",
        "dataset": "Vbench200",
        "threshold": config["threshold"],
        "use_ret_steps": config["use_ret_steps"],
        "protocol": config["protocol"],
        "paired_fidelity": {
            "protocol_id": metrics.get("protocol_id"),
            "psnr_rgb_db": mean_metric(metrics, "psnr_rgb_db"),
            "ssim_rgb": mean_metric(metrics, "ssim_rgb"),
            "lpips_alex_v0_1_spatial": mean_metric(
                metrics, "lpips_alex_v0_1_spatial"
            ),
        },
        "vbench200_subset_scores": {
            "official_full_vbench_score": False,
            "baseline": aggregate_scores(vbench_reference),
            "teacache": aggregate_scores(vbench_candidate),
        },
        "performance": {
            "latency_definition": performance["latency_definition"],
            "baseline": {
                "inference_latency_seconds_mean": baseline_latency["mean"],
                "inference_latency_seconds_p50": baseline_latency["p50"],
                "inference_latency_seconds_p90": baseline_latency["p90"],
                "estimated_dit_tflops_per_video_mean": baseline_tflops["mean"],
                "estimated_achieved_dit_tflops_per_second": baseline[
                    "estimated_achieved_dit_tflops_per_second_ratio_of_sums"
                ],
            },
            "teacache": {
                "inference_latency_seconds_mean": teacache_latency["mean"],
                "inference_latency_seconds_p50": teacache_latency["p50"],
                "inference_latency_seconds_p90": teacache_latency["p90"],
                "estimated_dit_tflops_per_video_mean": teacache_tflops["mean"],
                "estimated_achieved_dit_tflops_per_second": teacache[
                    "estimated_achieved_dit_tflops_per_second_ratio_of_sums"
                ],
            },
            "comparison": performance["comparison"],
        },
        "source_files": {
            "performance": str(result_dir / "performance" / "summary.json"),
            "video_metrics": str(
                result_dir / "evaluation" / "video_metrics" / "summary.json"
            ),
            "vbench_reference": str(
                result_dir
                / "evaluation"
                / "vbench_reference"
                / "vbench200_aggregate_scores.json"
            ),
            "vbench_candidate": str(
                result_dir
                / "evaluation"
                / "vbench_candidate"
                / "vbench200_aggregate_scores.json"
            ),
        },
    }
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fidelity = payload["paired_fidelity"]
    comparison = payload["performance"]["comparison"]
    baseline_perf = payload["performance"]["baseline"]
    teacache_perf = payload["performance"]["teacache"]
    baseline_vbench = payload["vbench200_subset_scores"]["baseline"]
    teacache_vbench = payload["vbench200_subset_scores"]["teacache"]
    output_markdown.write_text(
        "# TeaCache4Wan21 Vbench200 report\n\n"
        f"Threshold: `{payload['threshold']}`; retention steps: "
        f"`{payload['use_ret_steps']}`.\n\n"
        "| Metric | Baseline | TeaCache |\n"
        "| --- | ---: | ---: |\n"
        f"| Inference latency mean (s/video) | {baseline_perf['inference_latency_seconds_mean']:.6f} | {teacache_perf['inference_latency_seconds_mean']:.6f} |\n"
        f"| Estimated DiT TFLOPs/video | {baseline_perf['estimated_dit_tflops_per_video_mean']:.6f} | {teacache_perf['estimated_dit_tflops_per_video_mean']:.6f} |\n"
        f"| Estimated achieved DiT TFLOP/s | {baseline_perf['estimated_achieved_dit_tflops_per_second']:.6f} | {teacache_perf['estimated_achieved_dit_tflops_per_second']:.6f} |\n"
        f"| Vbench200 Quality | {baseline_vbench['quality_score']:.6f} | {teacache_vbench['quality_score']:.6f} |\n"
        f"| Vbench200 Semantic | {baseline_vbench['semantic_score']:.6f} | {teacache_vbench['semantic_score']:.6f} |\n"
        f"| Vbench200 Total | {baseline_vbench['total_score']:.6f} | {teacache_vbench['total_score']:.6f} |\n\n"
        "Paired fidelity of TeaCache against baseline: "
        f"PSNR `{fidelity['psnr_rgb_db']:.6f}` dB, "
        f"SSIM `{fidelity['ssim_rgb']:.6f}`, "
        f"LPIPS `{fidelity['lpips_alex_v0_1_spatial']:.6f}`.\n\n"
        f"Latency speedup: `{comparison['latency_speedup_baseline_over_teacache']:.6f}x`; "
        f"DiT FLOPs speedup: `{comparison['dit_flops_speedup_ratio_of_sums']:.6f}x`.\n\n"
        "Latency includes text encoding, denoising, and VAE decode, but excludes "
        "model loading, MP4 export, and metric evaluation. VBench values are "
        "Vbench200 subset scores, not full-suite leaderboard scores.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
