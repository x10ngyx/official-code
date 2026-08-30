#!/usr/bin/env python3
"""Merge latency, TFLOPs, PSNR/SSIM/LPIPS, and Vbench200 scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXP_ROOT = Path("/all/yiran07-disk3/huteng_data/exp").resolve()


def external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as exc:
        raise ValueError(f"result must be below {EXP_ROOT}: {resolved}") from exc
    return resolved


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return payload


def metric(payload: dict[str, Any], name: str) -> float:
    value = payload.get("metrics", {}).get(name, {}).get("mean")
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing VideoMetrics mean: {name}")
    return float(value)


def vbench(payload: dict[str, Any]) -> dict[str, float]:
    values = payload.get("aggregate_scores")
    if not isinstance(values, dict):
        raise ValueError("missing VBench aggregate_scores")
    return {name: float(values[name]) for name in ("quality_score", "semantic_score", "total_score")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = external(args.result_dir)
    output_json, output_md, output_csv = root / "benchmark_report.json", root / "benchmark_report.md", root / "benchmark_report.csv"
    if any(path.exists() for path in (output_json, output_md, output_csv)):
        raise FileExistsError("refusing to overwrite an existing benchmark report")
    config = load(root / "run_config.json")
    performance = load(root / "performance" / "summary.json")
    metrics = load(root / "evaluation" / "video_metrics" / "summary.json")
    reference_vbench = vbench(load(root / "evaluation" / "vbench_reference" / "vbench200_aggregate_scores.json"))
    candidate_vbench = vbench(load(root / "evaluation" / "vbench_candidate" / "vbench200_aggregate_scores.json"))
    baseline, candidate = performance["conditions"]["baseline"], performance["conditions"]["seacache"]
    fidelity = {
        "protocol_id": metrics.get("protocol_id"), "psnr_rgb_db": metric(metrics, "psnr_rgb_db"),
        "ssim_rgb": metric(metrics, "ssim_rgb"), "lpips_alex_v0_1_spatial": metric(metrics, "lpips_alex_v0_1_spatial"),
    }
    payload = {
        "schema": "seacache_vbench200_report_v2", "method": "SeaCache4Wan22",
        "model": "Wan2.2-T2V-A14B", "dataset": "Vbench200", "threshold": config["threshold"],
        "use_ret_steps": config["use_ret_steps"], "protocol": config["protocol"],
        "paired_fidelity_seacache_against_baseline": fidelity,
        "vbench200_subset_scores": {"official_full_vbench_score": False, "baseline": reference_vbench, "seacache": candidate_vbench},
        "performance": {
            "latency_definition": performance["latency_definition"], "flops_definition": performance["flops_definition"],
            "baseline": {
                "inference_time_seconds_mean": baseline["pipeline_generate_wall_seconds"]["mean"],
                "inference_time_seconds_p50": baseline["pipeline_generate_wall_seconds"]["p50"],
                "inference_time_seconds_p90": baseline["pipeline_generate_wall_seconds"]["p90"],
                "estimated_dit_tflops_per_video_mean": baseline["estimated_dit_tflops_per_video"]["mean"],
                "t5_cuda_seconds_mean": baseline["t5_cuda_seconds"]["mean"],
                "dit_cuda_seconds_mean": baseline["dit_forward_cuda_seconds"]["mean"],
                "vae_decode_cuda_seconds_mean": baseline["vae_decode_cuda_seconds"]["mean"],
                "estimated_t5_tflops_per_video": baseline["estimated_t5_tflops_per_video"],
                "estimated_vae_decode_tflops_per_video": baseline["estimated_vae_decode_tflops_per_video"],
            },
            "seacache": {
                "inference_time_seconds_mean": candidate["pipeline_generate_wall_seconds"]["mean"],
                "inference_time_seconds_p50": candidate["pipeline_generate_wall_seconds"]["p50"],
                "inference_time_seconds_p90": candidate["pipeline_generate_wall_seconds"]["p90"],
                "estimated_dit_tflops_per_video_mean": candidate["estimated_dit_tflops_per_video"]["mean"],
                "t5_cuda_seconds_mean": candidate["t5_cuda_seconds"]["mean"],
                "dit_cuda_seconds_mean": candidate["dit_forward_cuda_seconds"]["mean"],
                "vae_decode_cuda_seconds_mean": candidate["vae_decode_cuda_seconds"]["mean"],
                "estimated_t5_tflops_per_video": candidate["estimated_t5_tflops_per_video"],
                "estimated_vae_decode_tflops_per_video": candidate["estimated_vae_decode_tflops_per_video"],
            },
            "comparison": performance["comparison"],
        },
        "source_files": {
            "performance": str(root / "performance" / "summary.json"),
            "video_metrics": str(root / "evaluation" / "video_metrics" / "summary.json"),
            "vbench_reference": str(root / "evaluation" / "vbench_reference" / "vbench200_aggregate_scores.json"),
            "vbench_candidate": str(root / "evaluation" / "vbench_candidate" / "vbench200_aggregate_scores.json"),
        },
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    b, c, comparison = payload["performance"]["baseline"], payload["performance"]["seacache"], payload["performance"]["comparison"]
    output_md.write_text(
        "# SeaCache4Wan22 Vbench200 report\n\n"
        f"Threshold: `{payload['threshold']}`; retention steps: `{payload['use_ret_steps']}`.\n\n"
        "| Metric | Baseline | SeaCache |\n| --- | ---: | ---: |\n"
        f"| Inference time mean (s/video) | {b['inference_time_seconds_mean']:.6f} | {c['inference_time_seconds_mean']:.6f} |\n"
        f"| Estimated DiT TFLOPs/video | {b['estimated_dit_tflops_per_video_mean']:.6f} | {c['estimated_dit_tflops_per_video_mean']:.6f} |\n"
        f"| T5 CUDA time (s/video) | {b['t5_cuda_seconds_mean']:.6f} | {c['t5_cuda_seconds_mean']:.6f} |\n"
        f"| DiT CUDA time (s/video) | {b['dit_cuda_seconds_mean']:.6f} | {c['dit_cuda_seconds_mean']:.6f} |\n"
        f"| VAE decode CUDA time (s/video) | {b['vae_decode_cuda_seconds_mean']:.6f} | {c['vae_decode_cuda_seconds_mean']:.6f} |\n"
        f"| Estimated T5 TFLOPs/video | {b['estimated_t5_tflops_per_video']:.6f} | {c['estimated_t5_tflops_per_video']:.6f} |\n"
        f"| Estimated VAE decode TFLOPs/video | {b['estimated_vae_decode_tflops_per_video']:.6f} | {c['estimated_vae_decode_tflops_per_video']:.6f} |\n"
        f"| Vbench200 quality | {reference_vbench['quality_score']:.6f} | {candidate_vbench['quality_score']:.6f} |\n"
        f"| Vbench200 semantic | {reference_vbench['semantic_score']:.6f} | {candidate_vbench['semantic_score']:.6f} |\n"
        f"| Vbench200 total | {reference_vbench['total_score']:.6f} | {candidate_vbench['total_score']:.6f} |\n\n"
        f"Paired fidelity: PSNR `{fidelity['psnr_rgb_db']:.6f}` dB, SSIM `{fidelity['ssim_rgb']:.6f}`, LPIPS `{fidelity['lpips_alex_v0_1_spatial']:.6f}`.\n\n"
        f"Latency speedup: `{comparison['latency_speedup_ratio_of_sums']:.6f}x`; estimated DiT FLOPs speedup: `{comparison['dit_flops_speedup_ratio_of_sums']:.6f}x`.\n\n"
        "VBench values are Vbench200 subset scores, not full-suite leaderboard scores. TFLOPs is a trace-weighted estimated DiT operation count.\n",
        encoding="utf-8",
    )
    fields = ["condition", "inference_time_seconds_mean", "estimated_dit_tflops_per_video_mean", "vbench_quality", "vbench_semantic", "vbench_total", "psnr_rgb_db", "ssim_rgb", "lpips_alex_v0_1_spatial"]
    with output_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"condition": "baseline", "inference_time_seconds_mean": b["inference_time_seconds_mean"], "estimated_dit_tflops_per_video_mean": b["estimated_dit_tflops_per_video_mean"], "vbench_quality": reference_vbench["quality_score"], "vbench_semantic": reference_vbench["semantic_score"], "vbench_total": reference_vbench["total_score"], "psnr_rgb_db": "", "ssim_rgb": "", "lpips_alex_v0_1_spatial": ""})
        writer.writerow({"condition": "seacache", "inference_time_seconds_mean": c["inference_time_seconds_mean"], "estimated_dit_tflops_per_video_mean": c["estimated_dit_tflops_per_video_mean"], "vbench_quality": candidate_vbench["quality_score"], "vbench_semantic": candidate_vbench["semantic_score"], "vbench_total": candidate_vbench["total_score"], **{key: fidelity[key] for key in ("psnr_rgb_db", "ssim_rgb", "lpips_alex_v0_1_spatial")}})
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
