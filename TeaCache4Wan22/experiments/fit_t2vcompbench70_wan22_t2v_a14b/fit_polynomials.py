#!/usr/bin/env python3
"""Fit and validate separate high/low quartic TeaCache polynomials."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y
    ss_res = float(np.square(error).sum())
    ss_total = float(np.square(y - y.mean()).sum())
    return {
        "count": int(y.size),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "max_abs_error": float(np.abs(error).max()),
        "r2": float(1.0 - ss_res / ss_total) if ss_total > 0 else math.nan,
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
    }


def fit_one(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.asarray([row["e_rel_l1"] for row in rows], dtype=np.float64)
    y = np.asarray([row["h_rel_l1"] for row in rows], dtype=np.float64)
    coefficients = np.polyfit(x, y, 4)
    prediction = np.polyval(coefficients, x)
    branch_metrics = {}
    for branch in ("cond", "uncond"):
        mask = np.asarray([row["branch"] == branch for row in rows])
        branch_metrics[branch] = metrics(y[mask], prediction[mask])
    category_metrics = {}
    for category in sorted({row["category"] for row in rows}):
        mask = np.asarray([row["category"] == category for row in rows])
        category_metrics[category] = metrics(y[mask], prediction[mask])
    return {
        "coefficient_order": ["x^4", "x^3", "x^2", "x", "constant"],
        "coefficients_descending": [float(value) for value in coefficients],
        "x_unique_count": int(np.unique(x).size),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "overall_metrics": metrics(y, prediction),
        "branch_metrics": branch_metrics,
        "category_metrics": category_metrics,
    }


def main() -> None:
    args = parse_args()
    manifest = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(manifest) != 70:
        raise ValueError(f"Expected 70 manifest rows, found {len(manifest)}")

    sample_paths = sorted((args.result_root / "samples").glob("*.json"))
    payloads = [load_json(path) for path in sample_paths]
    complete = [payload for payload in payloads if payload.get("status") == "complete"]
    expected_ids = {row["sample_id"] for row in manifest}
    observed_ids = {payload["prompt"]["sample_id"] for payload in complete}
    missing = sorted(expected_ids - observed_ids)
    unexpected = sorted(observed_ids - expected_ids)
    if missing or unexpected or len(complete) != 70:
        raise ValueError(
            f"Calibration set is incomplete: complete={len(complete)}, missing={missing}, unexpected={unexpected}"
        )

    records: list[dict[str, Any]] = []
    run_configs = []
    for payload in complete:
        if payload.get("record_count") != 100 or len(payload.get("records", [])) != 100:
            raise ValueError(f"Invalid record count for {payload['prompt']['sample_id']}")
        records.extend(payload["records"])
        run_configs.append(payload["run_config"])
    canonical_config = json.dumps(run_configs[0], sort_keys=True)
    if any(json.dumps(config, sort_keys=True) != canonical_config for config in run_configs[1:]):
        raise ValueError("Run configurations differ across prompts")

    paired = [row for row in records if row["paired_within_stage"]]
    eligible = [row for row in records if row["fit_eligible"]]
    if len(records) != 7000 or len(paired) != 6720 or len(eligible) != 6580:
        raise ValueError(
            f"Unexpected aggregate counts: records={len(records)}, paired={len(paired)}, eligible={len(eligible)}"
        )

    primary = {
        stage: fit_one([row for row in eligible if row["stage"] == stage])
        for stage in ("high", "low")
    }
    all_within_stage = {
        stage: fit_one([row for row in paired if row["stage"] == stage])
        for stage in ("high", "low")
    }
    result = {
        "method": "ordinary quartic least squares on raw pooled cond/uncond observations",
        "formula": "y_hat = a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0",
        "x_definition": "adjacent relative-L1 of non-retention timestep embedding e",
        "y_definition": "adjacent relative-L1 of full-compute post-block/pre-head hidden state H",
        "primary_fit_scope": "within-stage online gate-eligible transitions; excludes each stage's first point and global final forced-recompute step 49",
        "use_ret_steps": False,
        "prompt_count": 70,
        "raw_forward_record_count": len(records),
        "within_stage_transition_count": len(paired),
        "primary_fit_record_count": len(eligible),
        "run_config": run_configs[0],
        "primary": primary,
        "diagnostic_all_within_stage_transitions": all_within_stage,
    }
    (args.result_root / "coefficients.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    grouped: dict[tuple[str, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        grouped[(row["stage"], row["branch"], row["global_step"], row["timestep"])].append(row)
    point_path = args.result_root / "fit_points_summary.csv"
    with point_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "stage",
                "branch",
                "global_step",
                "timestep",
                "fit_eligible",
                "count",
                "e_rel_l1_mean",
                "e_rel_l1_std",
                "h_rel_l1_mean",
                "h_rel_l1_std",
                "block_residual_rel_l1_mean",
                "block_residual_rel_l1_std",
            ],
        )
        writer.writeheader()
        for key in sorted(grouped, key=lambda item: (item[2], item[1])):
            rows = grouped[key]
            residual = np.asarray(
                [row["block_residual_rel_l1"] for row in rows if row["block_residual_rel_l1"] is not None],
                dtype=np.float64,
            )
            x = np.asarray([row["e_rel_l1"] for row in rows], dtype=np.float64)
            y = np.asarray([row["h_rel_l1"] for row in rows], dtype=np.float64)
            writer.writerow(
                {
                    "stage": key[0],
                    "branch": key[1],
                    "global_step": key[2],
                    "timestep": key[3],
                    "fit_eligible": all(row["fit_eligible"] for row in rows),
                    "count": len(rows),
                    "e_rel_l1_mean": f"{x.mean():.17g}",
                    "e_rel_l1_std": f"{x.std():.17g}",
                    "h_rel_l1_mean": f"{y.mean():.17g}",
                    "h_rel_l1_std": f"{y.std():.17g}",
                    "block_residual_rel_l1_mean": f"{residual.mean():.17g}" if residual.size else "",
                    "block_residual_rel_l1_std": f"{residual.std():.17g}" if residual.size else "",
                }
            )

    lines = [
        "# TeaCache Wan2.2 T2V-A14B quartic fit",
        "",
        "Primary fits use only runtime gate-eligible within-stage transitions under `use_ret_steps=False`.",
        "Cond/uncond observations are pooled into one polynomial per stage, with branch metrics reported separately.",
        "",
    ]
    for stage in ("high", "low"):
        fit = primary[stage]
        coeff = fit["coefficients_descending"]
        lines.extend(
            [
                f"## {stage}",
                "",
                "`[a4, a3, a2, a1, a0] = [" + ", ".join(f"{value:.17g}" for value in coeff) + "]`",
                "",
                f"Records: {fit['overall_metrics']['count']}; unique x: {fit['x_unique_count']}; "
                f"R²: {fit['overall_metrics']['r2']:.8f}; RMSE: {fit['overall_metrics']['rmse']:.8g}; "
                f"MAE: {fit['overall_metrics']['mae']:.8g}.",
                "",
            ]
        )
    (args.result_root / "FIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
