#!/usr/bin/env python3
"""Audit and summarize the completed MagCache4Wan22 VBench200 suite."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path


THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
TARGETS = {
    "target_1p8x_gpu1": {"label": "1.8x", "R": 0.2, "E": 0.072, "K": 2, "full": 52, "reuse": 48},
    "target_2p4x_gpu2": {"label": "2.4x", "R": 0.2, "E": 0.199, "K": 4, "full": 38, "reuse": 62},
    "target_3p0x_gpu3": {"label": "3.0x", "R": 0.1, "E": 0.198, "K": 5, "full": 29, "reuse": 71},
}
PROTOCOL = {
    "task": "t2v-A14B", "batch_size": 1, "size_wh": [832, 480], "frame_num": 45, "fps": 16,
    "sampling_steps": 50, "sample_solver": "dpm++", "shift": 12.0,
    "guide_scale_low_high": [3.0, 4.0], "boundary": 0.875, "seed": 42,
    "param_dtype": "torch.bfloat16", "offload_model": True, "t5_cpu": False,
    "distributed": False, "prompt_extension": False,
}
BASELINE_VBENCH = Path(
    "/all/yiran07-disk3/huteng_data/exp/"
    "wan22_seacache_vbench200_thr024_038_055_persistent_batch2wave_gpu0123_20260903_002834/"
    "shared_evaluation/vbench_reference/vbench200_aggregate_scores.json"
)
BASELINE_VBENCH_SHA256 = "71f7d10a6ab361031397d3b650a8979dd9060d18211ec6f724545b69c86c13df"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def close(left: float, right: float) -> None:
    if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8):
        raise AssertionError((left, right))


def committed(record: dict) -> dict:
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise AssertionError(f"SHA mismatch: {path}")
    return read(path)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: audit_results.py SUITE_ROOT")
    if Path(sys.prefix).name != "wan2.2" or any(os.environ.get(name) != "1" for name in THREAD_VARS):
        raise RuntimeError("use wan2.2 with all four BLAS thread variables set to 1")
    root = Path(sys.argv[1]).resolve(strict=True)
    complete = read(root / "COMPLETE.json")
    if complete.get("status") != "complete" or read(root / "status.json") != {"status": "complete", "phase": "complete"}:
        raise AssertionError("suite is not complete")
    if committed(complete["balanced_assignment"]) != read(root / "balanced_assignment.json"):
        raise AssertionError("assignment record mismatch")
    if sha256(BASELINE_VBENCH) != BASELINE_VBENCH_SHA256:
        raise AssertionError("shared baseline VBench record changed")
    baseline_vbench_record = read(BASELINE_VBENCH)
    if (len(baseline_vbench_record["raw_dimension_scores"]) != 16
            or len(baseline_vbench_record["normalized_dimension_scores"]) != 16):
        raise AssertionError("shared baseline VBench dimension mismatch")
    baseline_vbench_score = baseline_vbench_record["aggregate_scores"]["total_score"]

    summaries = []
    baseline_links = []
    checks = Counter()
    for target, spec in TARGETS.items():
        directory = root / target
        target_complete = read(directory / "COMPLETE.json")
        report_record = complete["reports"][target]
        report = committed(report_record)
        if target_complete != {"status": "complete", "report": report_record} or report.get("status") != "complete":
            raise AssertionError(f"uncommitted report: {target}")
        performance = report["performance"]["magcache_000"]
        if performance["pair_count"] != 200 or performance["method"] != {
            "threshold": spec["E"], "K": spec["K"], "retention_ratio": spec["R"], "split_steps": 32,
            "ratio_source": "official_builtin_40step_nearest_interp", "variant": "official_unmodified",
        }:
            raise AssertionError(f"performance method mismatch: {target}")
        for mode in ("baseline", "magcache"):
            for key, total in performance["sums"][mode].items():
                close(float(total), sum(float(row[mode][key]) for row in performance["per_video"]))
        close(performance["speedup"], performance["sums"]["baseline"]["generate_seconds"] /
              performance["sums"]["magcache"]["generate_seconds"])
        close(performance["dit_flops_speedup"], performance["sums"]["baseline"]["estimated_dit_tflops"] /
              performance["sums"]["magcache"]["estimated_dit_tflops"])
        checks["performance_rows"] += 200

        expected_ids = [f"vbench200_{index:03d}" for index in range(1, 201)]
        gpu_counts = Counter()
        for sample_id in expected_ids:
            run = directory / "runs" / "magcache_000" / sample_id
            manifest = read(run / "manifest.json")
            timing = read(run / "timing.json")
            trace = read(run / "trace.json")
            video = run / "videos" / f"{sample_id}.mp4"
            if manifest.get("status") != "complete" or timing.get("status") != "success" or not video.is_file():
                raise AssertionError(f"incomplete sample: {run}")
            if manifest["sample_id"] != sample_id or any(manifest["protocol"][key] != value for key, value in PROTOCOL.items()):
                raise AssertionError(f"protocol mismatch: {run}")
            actions = Counter(row["action"] for row in trace["calls"])
            if len(trace["calls"]) != 100 or (actions["recompute"], actions["reuse"]) != (spec["full"], spec["reuse"]):
                raise AssertionError(f"schedule mismatch: {run}")
            if timing["full_compute_forward_calls"] != spec["full"] or timing["reuse_forward_calls"] != spec["reuse"]:
                raise AssertionError(f"timing schedule mismatch: {run}")
            lifecycle = timing["pipeline_lifecycle"]
            if not lifecycle["persistent_pipeline"] or timing["pipeline_init_wall_seconds"] != 0:
                raise AssertionError(f"lifecycle mismatch: {run}")
            gpu_counts[str(lifecycle["physical_gpu"])] += 1
            checks["candidate_manifests"] += 1
            checks["candidate_traces"] += 1
            checks["candidate_videos"] += 1

        links = sorted((directory / "videos" / "baseline").glob("*.mp4"))
        if [path.stem for path in links] != expected_ids or not all(path.is_symlink() for path in links):
            raise AssertionError(f"baseline links invalid: {target}")
        baseline_links.append({path.name: str(path.resolve()) for path in links})

        metrics = report["video_metrics"]["magcache_000"]
        if metrics["video_count"] != 200 or metrics["frame_count_total"] != 9000:
            raise AssertionError(f"VideoMetrics coverage mismatch: {target}")
        for key in ("psnr_rgb_db", "ssim_rgb", "lpips_alex_v0_1_spatial"):
            if not math.isfinite(float(metrics["metrics"][key]["mean"])):
                raise AssertionError(f"non-finite metric: {target} {key}")
        value = report["vbench"]["magcache_000"]
        if len(value["raw_dimension_scores"]) != 16 or len(value["normalized_dimension_scores"]) != 16:
            raise AssertionError(f"VBench dimension mismatch: {target}")
        close(report["vbench_score"]["magcache_000"], value["aggregate_scores"]["total_score"])
        checks["paired_frames"] += 9000
        checks["candidate_vbench_records"] += 1

        base = performance["sums"]["baseline"]
        cand = performance["sums"]["magcache"]
        summaries.append({
            "target": spec["label"], "R": spec["R"], "E": spec["E"], "K": spec["K"],
            "full_calls": spec["full"], "reuse_calls": spec["reuse"], "physical_gpu_counts": dict(gpu_counts),
            "baseline_generate_seconds_mean": base["generate_seconds"] / 200,
            "candidate_generate_seconds_mean": cand["generate_seconds"] / 200,
            "latency_speedup": performance["speedup"],
            "candidate_t5_cuda_seconds_mean": cand["t5_cuda_seconds"] / 200,
            "candidate_dit_cuda_seconds_mean": cand["dit_cuda_seconds"] / 200,
            "candidate_vae_cuda_seconds_mean": cand["vae_decode_cuda_seconds"] / 200,
            "baseline_dit_tflops_mean": base["estimated_dit_tflops"] / 200,
            "candidate_dit_tflops_mean": cand["estimated_dit_tflops"] / 200,
            "dit_flops_speedup": performance["dit_flops_speedup"],
            "t5_tflops_mean": cand["estimated_t5_tflops_per_video"] / 200,
            "vae_tflops_mean": cand["estimated_vae_decode_tflops_per_video"] / 200,
            "psnr_rgb_db": metrics["metrics"]["psnr_rgb_db"]["mean"],
            "ssim_rgb": metrics["metrics"]["ssim_rgb"]["mean"],
            "lpips": metrics["metrics"]["lpips_alex_v0_1_spatial"]["mean"],
            "baseline_vbench_score": baseline_vbench_score,
            "candidate_vbench_score": report["vbench_score"]["magcache_000"],
            "vbench_delta": report["vbench_score"]["magcache_000"] - baseline_vbench_score,
            "quality_evaluation_gpu": report["quality_evaluation_gpu"],
        })

    if baseline_links[0] != baseline_links[1] or baseline_links[0] != baseline_links[2]:
        raise AssertionError("targets do not share identical baseline links")
    validation = {
        "status": "pass", "scope": "MagCache4Wan22 VBench200 three-target final audit",
        "checks": dict(checks), "baseline_links_per_target": 200, "baseline_generated_videos": 0,
        "shared_baseline_targets_identical": True, "target_reports": 3,
        "shared_baseline_vbench": {"path": str(BASELINE_VBENCH), "sha256": BASELINE_VBENCH_SHA256},
        "source_complete": {"path": str(root / "COMPLETE.json"), "sha256": sha256(root / "COMPLETE.json")},
    }
    summary = {"schema": "magcache4wan22_vbench200_final_summary_v1", "status": "complete", "targets": summaries}
    atomic(root / "FINAL_SUMMARY.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    atomic(root / "FINAL_VALIDATION.json", json.dumps(validation, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# MagCache4Wan22 VBench200 final results", "",
        "All three targets contain 200 candidate videos and reuse the same audited 200-video baseline. Latency speedup uses corrected shared-baseline reporting times and complete candidate generate wall time. VideoMetrics uses RGB with 9000 paired frames per target; VBench is the fixed Vbench200 16-dimension aggregate.", "",
        "| Target | R | E | K | Full/reuse | Generate s | Speedup | DiT TFLOPs | DiT reduction | RGB PSNR | SSIM | LPIPS | VBench | Delta vs baseline |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['target']} | {row['R']:.1f} | {row['E']:.3f} | {row['K']} | {row['full_calls']}/{row['reuse_calls']} | "
            f"{row['candidate_generate_seconds_mean']:.3f} | {row['latency_speedup']:.6f}x | {row['candidate_dit_tflops_mean']:.3f} | "
            f"{row['dit_flops_speedup']:.6f}x | {row['psnr_rgb_db']:.6f} | {row['ssim_rgb']:.6f} | {row['lpips']:.6f} | "
            f"{row['candidate_vbench_score'] * 100:.4f}% | {row['vbench_delta'] * 100:+.4f} pp |"
        )
    lines += ["", f"Shared baseline: {summaries[0]['baseline_generate_seconds_mean']:.6f} seconds/video, "
              f"{summaries[0]['baseline_dit_tflops_mean']:.3f} DiT TFLOPs/video, and {baseline_vbench_score * 100:.4f}% VBench.", "",
              "Validation: 600 manifests, timings, traces and candidate videos; 27,000 paired frames; three candidate plus one reused baseline 16-dimension VBench records; zero newly generated baseline videos."]
    atomic(root / "RESULTS.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
