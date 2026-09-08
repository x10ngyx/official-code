#!/usr/bin/env python3
"""Merge the three locked TeaCache threshold reports into one suite table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


EXP_ROOT = Path("/all/yiran07-disk3/huteng_data/exp").resolve()
LOCKED_THRESHOLDS = [0.29, 0.45, 0.68]


def external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXP_ROOT)
    except ValueError as exc:
        raise ValueError(f"path must be below {EXP_ROOT}: {resolved}") from exc
    return resolved


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, action="append", required=True)
    args = parser.parse_args()
    suite_root = external(args.suite_root)
    roots = [external(path) for path in args.result_dir]
    if len(roots) != len(LOCKED_THRESHOLDS):
        raise ValueError("the formal suite requires exactly three result directories")
    outputs = [
        suite_root / "suite_report.json", suite_root / "suite_report.csv",
        suite_root / "suite_report.md", suite_root / "README.md",
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite an existing suite report")

    rows: list[dict[str, Any]] = []
    baseline_targets: set[Path] = set()
    baseline_vbench: dict[str, float] | None = None
    baseline_performance: dict[str, Any] | None = None
    for expected_threshold, root in zip(LOCKED_THRESHOLDS, roots):
        status = load(root / "status.json")
        report = load(root / "benchmark_report.json")
        if status.get("status") != "complete":
            raise ValueError(f"threshold result is not complete: {root}")
        observed_threshold = float(report.get("threshold"))
        if not math.isclose(observed_threshold, expected_threshold, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"threshold order mismatch: expected {expected_threshold}, got {observed_threshold}"
            )
        baseline_targets.add((root / "baseline").resolve(strict=True))
        current_baseline_vbench = report["vbench200_subset_scores"]["baseline"]
        if baseline_vbench is None:
            baseline_vbench = current_baseline_vbench
        elif current_baseline_vbench != baseline_vbench:
            raise ValueError("reused baseline VBench scores differ across thresholds")
        performance = report["performance"]
        current_baseline_performance = performance["baseline"]
        if baseline_performance is None:
            baseline_performance = current_baseline_performance
        elif current_baseline_performance != baseline_performance:
            raise ValueError("reused baseline performance differs across thresholds")
        candidate = performance["teacache"]
        fidelity = report["paired_fidelity_teacache_against_baseline"]
        candidate_vbench = report["vbench200_subset_scores"]["teacache"]
        rows.append({
            "threshold": observed_threshold,
            "latency_seconds_mean": candidate["inference_time_seconds_mean"],
            "latency_speedup_ratio_of_sums": performance["comparison"]["latency_speedup_ratio_of_sums"],
            "estimated_dit_tflops_per_video_mean": candidate["estimated_dit_tflops_per_video_mean"],
            "dit_flops_speedup_ratio_of_sums": performance["comparison"]["dit_flops_speedup_ratio_of_sums"],
            "psnr_rgb_db": fidelity["psnr_rgb_db"],
            "ssim_rgb": fidelity["ssim_rgb"],
            "lpips_alex_v0_1_spatial": fidelity["lpips_alex_v0_1_spatial"],
            "vbench_quality": candidate_vbench["quality_score"],
            "vbench_semantic": candidate_vbench["semantic_score"],
            "vbench_total": candidate_vbench["total_score"],
            "result_dir": str(root),
        })
    if len(baseline_targets) != 1:
        raise ValueError(f"thresholds do not reuse one baseline directory: {baseline_targets}")

    payload = {
        "schema": "teacache4wan22_vbench200_threshold_suite_v1",
        "method": "TeaCache4Wan22",
        "thresholds": LOCKED_THRESHOLDS,
        "baseline_source": str(next(iter(baseline_targets))),
        "baseline_vbench200_subset_scores": baseline_vbench,
        "baseline_performance": baseline_performance,
        "rows": rows,
    }
    (suite_root / "suite_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [key for key in rows[0] if key != "result_dir"] + ["result_dir"]
    with (suite_root / "suite_report.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    table_rows = "\n".join(
        f"| {row['threshold']:.2f} | {row['latency_speedup_ratio_of_sums']:.6f} | "
        f"{row['dit_flops_speedup_ratio_of_sums']:.6f} | {row['psnr_rgb_db']:.6f} | "
        f"{row['ssim_rgb']:.6f} | {row['lpips_alex_v0_1_spatial']:.6f} | "
        f"{row['vbench_total']:.6f} |"
        for row in rows
    )
    (suite_root / "suite_report.md").write_text(
        "# TeaCache4Wan22 formal VBench200 threshold suite\n\n"
        "| Threshold | Latency speedup | DiT FLOPs speedup | PSNR (dB) | SSIM | LPIPS | VBench total |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"{table_rows}\n\n"
        "All thresholds directly reuse one SeaCache no-cache baseline. VBench values are "
        "VBench200 subset scores, not full-suite leaderboard scores.\n",
        encoding="utf-8",
    )
    (suite_root / "README.md").write_text(
        "# TeaCache4Wan22 formal VBench200 suite\n\n"
        "Three fixed-threshold result directories plus JSON/CSV/Markdown aggregate reports. "
        "Each threshold directory contains generation traces, performance aggregation, "
        "PSNR/SSIM/LPIPS, and VBench200 subset scores.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
