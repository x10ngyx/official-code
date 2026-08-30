#!/usr/bin/env python3
"""Independently audit the fixed-protocol Wan2.2 TeaCache quartic fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
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


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
EXPECTED_CATEGORY_COUNTS = {
    "action_binding": 10,
    "consistent_attribute_binding": 10,
    "dynamic_attribute_binding": 10,
    "generative_numeracy": 10,
    "motion_binding": 10,
    "object_interactions": 10,
    "spatial_relationships": 10,
}
EXPECTED_STAGE_COUNTS = {"high": 4340, "low": 2240}
EXPECTED_UNIQUE_X = {"high": 31, "low": 16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = prediction - y
    ss_total = float(np.square(y - y.mean()).sum())
    return {
        "count": int(y.size),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "max_abs_error": float(np.abs(error).max()),
        "r2": float(1.0 - np.square(error).sum() / ss_total),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def cross_validated_predictions(
    rows: list[dict[str, Any]], group_key: str
) -> np.ndarray:
    x = np.asarray([row["e_rel_l1"] for row in rows], dtype=np.float64)
    y = np.asarray([row["h_rel_l1"] for row in rows], dtype=np.float64)
    groups = np.asarray([row[group_key] for row in rows])
    prediction = np.empty_like(y)
    for group in sorted(set(groups)):
        test = groups == group
        require(int((~test).sum()) >= 5, f"Too few training rows after excluding {group!r}.")
        coefficients = np.polyfit(x[~test], y[~test], 4)
        prediction[test] = np.polyval(coefficients, x[test])
    return prediction


def audit_stage(
    stage: str, rows: list[dict[str, Any]], stored: dict[str, Any]
) -> dict[str, Any]:
    x = np.asarray([row["e_rel_l1"] for row in rows], dtype=np.float64)
    y = np.asarray([row["h_rel_l1"] for row in rows], dtype=np.float64)
    coefficients = np.polyfit(x, y, 4)
    stored_coefficients = np.asarray(
        stored["coefficients_descending"], dtype=np.float64
    )
    coefficient_exact_match = bool(np.array_equal(coefficients, stored_coefficients))
    coefficient_max_abs_difference = float(
        np.max(np.abs(coefficients - stored_coefficients))
    )
    require(
        np.allclose(
            coefficients, stored_coefficients, rtol=1e-12, atol=1e-12
        ),
        f"{stage} coefficients do not match an independent np.polyfit recomputation.",
    )
    prediction = np.polyval(coefficients, x)
    recomputed_metrics = metrics(y, prediction)
    for key in ("count", "mae", "rmse", "max_abs_error", "r2"):
        require(
            math.isclose(
                float(recomputed_metrics[key]),
                float(stored["overall_metrics"][key]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            f"{stage} metric {key} does not match the stored fit.",
        )

    require(len(rows) == EXPECTED_STAGE_COUNTS[stage], f"Unexpected {stage} row count.")
    unique_x = np.unique(x)
    require(
        unique_x.size == EXPECTED_UNIQUE_X[stage],
        f"Unexpected {stage} unique-x count.",
    )
    grouped_x: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped_x[int(row["global_step"])].append(float(row["e_rel_l1"]))
    max_within_step_x_spread = max(max(values) - min(values) for values in grouped_x.values())
    require(
        max_within_step_x_spread == 0.0,
        f"{stage} timestep-only gate feature varies across prompts or CFG branches.",
    )

    unique_prediction = np.polyval(coefficients, unique_x)
    require(np.isfinite(unique_prediction).all(), f"{stage} has non-finite in-domain predictions.")
    require((unique_prediction >= 0).all(), f"{stage} has negative in-domain predictions.")

    branch_metrics: dict[str, Any] = {}
    branches = np.asarray([row["branch"] for row in rows])
    for branch in ("cond", "uncond"):
        mask = branches == branch
        branch_error = prediction[mask] - y[mask]
        branch_metrics[branch] = {
            **metrics(y[mask], prediction[mask]),
            "mean_error": float(branch_error.mean()),
        }

    prompt_ids = np.asarray([row["sample_id"] for row in rows])
    prompt_mae = []
    for sample_id in sorted(set(prompt_ids)):
        mask = prompt_ids == sample_id
        prompt_mae.append(float(np.abs(prediction[mask] - y[mask]).mean()))

    categories = np.asarray([row["category"] for row in rows])
    leave_category_out_prediction = cross_validated_predictions(rows, "category")
    leave_category_out_by_category = {}
    for category in sorted(set(categories)):
        mask = categories == category
        leave_category_out_by_category[category] = metrics(
            y[mask], leave_category_out_prediction[mask]
        )

    centered_x = (x - x.mean()) / x.std()
    return {
        "row_count": len(rows),
        "unique_x_count": int(unique_x.size),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "max_within_step_x_spread": max_within_step_x_spread,
        "coefficients_descending": [float(value) for value in coefficients],
        "coefficient_recompute_match": True,
        "coefficient_exact_recompute_match": coefficient_exact_match,
        "coefficient_max_abs_difference": coefficient_max_abs_difference,
        "in_sample": recomputed_metrics,
        "leave_one_prompt_out": metrics(
            y, cross_validated_predictions(rows, "sample_id")
        ),
        "leave_one_category_out": metrics(y, leave_category_out_prediction),
        "leave_one_category_out_by_category": leave_category_out_by_category,
        "branch_metrics": branch_metrics,
        "prompt_mae_distribution": {
            "min": float(np.min(prompt_mae)),
            "median": float(np.median(prompt_mae)),
            "p90": float(np.quantile(prompt_mae, 0.9)),
            "max": float(np.max(prompt_mae)),
        },
        "in_domain_prediction": {
            "min": float(unique_prediction.min()),
            "max": float(unique_prediction.max()),
            "negative_count": int((unique_prediction < 0).sum()),
        },
        "vandermonde_condition_number": {
            "raw_x": float(np.linalg.cond(np.vander(x, 5))),
            "standardized_x": float(np.linalg.cond(np.vander(centered_x, 5))),
        },
    }


def write_reports(result_root: Path, report: dict[str, Any]) -> None:
    json_path = result_root / "FIT_AUDIT_REPORT.json"
    markdown_path = result_root / "FIT_AUDIT_REPORT.md"
    temporary_json = json_path.with_name(json_path.name + f".tmp.{os.getpid()}")
    temporary_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_json, json_path)

    high = report["stages"]["high"]
    low = report["stages"]["low"]
    lines = [
        "# Wan2.2 TeaCache fit audit",
        "",
        "Assessment: **share with caveats**. Data completeness, provenance, and all",
        "stored coefficients/metrics pass independent recomputation. The fit is valid as",
        "a fixed-protocol TeaCache gate calibration, but it is not a threshold recommendation",
        "or an end-to-end quality/speed claim.",
        "",
        "## Verified",
        "",
        "- 70/70 prompts are complete, with seven categories of 10 prompts and no duplicate records.",
        "- 7,000 raw CFG-forward records reduce to 6,580 runtime gate-eligible records.",
        "- High/low quartic coefficients exactly match an independent NumPy recomputation.",
        "- The noncanonical collection tree matches the approved prepared tree on all 96",
        "  `e/H/(H-Z)` statistics for the rerun reference prompt (zero error).",
        "- All in-domain polynomial outputs are finite and non-negative.",
        "",
        "## Fit metrics",
        "",
        "| Stage | Rows | Unique x | In-sample R² | MAE | RMSE | LOPO R² | LOCO R² |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| high | {high['row_count']} | {high['unique_x_count']} | "
            f"{high['in_sample']['r2']:.6f} | {high['in_sample']['mae']:.6f} | "
            f"{high['in_sample']['rmse']:.6f} | {high['leave_one_prompt_out']['r2']:.6f} | "
            f"{high['leave_one_category_out']['r2']:.6f} |"
        ),
        (
            f"| low | {low['row_count']} | {low['unique_x_count']} | "
            f"{low['in_sample']['r2']:.6f} | {low['in_sample']['mae']:.6f} | "
            f"{low['in_sample']['rmse']:.6f} | {low['leave_one_prompt_out']['r2']:.6f} | "
            f"{low['leave_one_category_out']['r2']:.6f} |"
        ),
        "",
        "LOPO leaves out one prompt; LOCO leaves out one complete T2V-CompBench category.",
        "",
        "## Required caveats",
        "",
        "- High-stage explanatory power is weak (`R²≈0.278`; LOCO `R²≈0.267`),",
        "  including a particularly weak held-out generative-numeracy segment. Treat the",
        "  quartic as the intended TeaCache heuristic, not an accurate per-forward predictor.",
        "- `x` is timestep-only, so the effective support is 31 high-stage and 16 low-stage",
        "  values; the larger row counts estimate target variability across prompts/CFG branches.",
        "- Raw-power quartic coefficients are ill-conditioned. Runtime use is safe only inside",
        "  the locked protocol/domain with double-precision Horner evaluation; do not extrapolate.",
        "- Cond/uncond are deliberately pooled for one shared gate. Their mean residual errors",
        "  have opposite signs, so the mapping estimates their balanced average.",
        "- Threshold selection still requires matched-seed end-to-end speed and fidelity tests.",
    ]
    temporary_markdown = markdown_path.with_name(
        markdown_path.name + f".tmp.{os.getpid()}"
    )
    temporary_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary_markdown, markdown_path)


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    manifest_path = result_root / "prompts.jsonl"
    coefficient_path = result_root / "coefficients.json"
    selection_path = result_root / "prompt_selection.json"
    equivalence_path = result_root / "SOURCE_EQUIVALENCE_REPORT.json"
    for path in (manifest_path, coefficient_path, selection_path, equivalence_path):
        require(path.is_file(), f"Missing audit input: {path}")

    manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(manifest) == 70, "Expected exactly 70 manifest rows.")
    require(
        len({row["sample_id"] for row in manifest}) == 70,
        "Manifest sample IDs are not unique.",
    )
    require(
        dict(sorted(Counter(row["category"] for row in manifest).items()))
        == EXPECTED_CATEGORY_COUNTS,
        "Manifest category counts are not seven categories of 10.",
    )
    selection = load_json(selection_path)
    require(
        selection.get("manifest_sha256") == sha256(manifest_path),
        "Prompt selection does not match prompts.jsonl.",
    )
    source_metadata = {
        row["filename"]: row for row in selection.get("source_files", [])
    }
    require(len(source_metadata) == 7, "Expected seven prompt source files.")
    source_lines = {}
    for filename, metadata in source_metadata.items():
        source_path = result_root / "source_prompts" / filename
        require(source_path.is_file(), f"Missing prompt source file: {filename}")
        require(
            sha256(source_path) == metadata["sha256"],
            f"Prompt source SHA256 mismatch: {filename}",
        )
        lines = [
            line.strip()
            for line in source_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        require(
            len(lines) == metadata["prompt_count"],
            f"Prompt source row count mismatch: {filename}",
        )
        source_lines[filename] = lines
    for row in manifest:
        lines = source_lines[row["source_file"]]
        require(
            lines[row["source_index_0based"]] == row["prompt"],
            f"Manifest prompt does not match its source row: {row['sample_id']}",
        )

    sample_paths = sorted((result_root / "samples").glob("*.json"))
    require(len(sample_paths) == 70, "Expected exactly 70 sample files.")
    payloads = [load_json(path) for path in sample_paths]
    require(
        all(payload.get("status") == "complete" for payload in payloads),
        "Not all sample payloads are complete.",
    )
    require(
        all(len(payload.get("records", [])) == 100 for payload in payloads),
        "Every sample must contain 100 records.",
    )
    require(
        {payload["prompt"]["sample_id"] for payload in payloads}
        == {row["sample_id"] for row in manifest},
        "Sample files do not cover the manifest exactly.",
    )
    canonical_config = json.dumps(payloads[0]["run_config"], sort_keys=True)
    require(
        all(
            json.dumps(payload["run_config"], sort_keys=True) == canonical_config
            for payload in payloads
        ),
        "Run configurations differ across samples.",
    )

    shard_paths = sorted((result_root / "shards").glob("shard_*.json"))
    require(
        [path.name for path in shard_paths]
        == [f"shard_{index}.json" for index in range(4)],
        "Expected exactly shard_0.json through shard_3.json.",
    )
    shard_payloads = [load_json(path) for path in shard_paths]
    require(
        all(payload.get("status") == "complete" for payload in shard_payloads),
        "Not all shard metadata records are complete.",
    )
    selected_ids = [
        sample_id
        for payload in shard_payloads
        for sample_id in payload["selected_sample_ids"]
    ]
    require(
        len(selected_ids) == 70
        and len(set(selected_ids)) == 70
        and set(selected_ids) == {row["sample_id"] for row in manifest},
        "Shard selections do not partition the prompt manifest exactly.",
    )
    require(
        all(
            payload.get("manifest_sha256") == sha256(manifest_path)
            and json.dumps(payload.get("run_config"), sort_keys=True)
            == canonical_config
            for payload in shard_payloads
        ),
        "Shard manifest or run configuration differs from the samples.",
    )
    collection_sources = {
        (
            payload["wan_model_py_sha256"],
            payload["wan_text2video_py_sha256"],
        )
        for payload in shard_payloads
    }
    require(len(collection_sources) == 1, "Calibration shards used different source trees.")
    model_sha, text2video_sha = next(iter(collection_sources))
    collection_source = {
        "wan/modules/model.py": model_sha,
        "wan/text2video.py": text2video_sha,
    }
    failure_file_count = len(list((result_root / "failures").glob("*.json")))
    require(failure_file_count == 0, "Calibration result contains failure records.")

    records = [row for payload in payloads for row in payload["records"]]
    record_keys = [
        (row["sample_id"], row["stage"], row["branch"], row["global_step"])
        for row in records
    ]
    require(len(records) == 7000, "Expected 7,000 raw records.")
    require(len(record_keys) == len(set(record_keys)), "Duplicate record grain detected.")
    eligible = [row for row in records if row["fit_eligible"]]
    require(len(eligible) == 6580, "Expected 6,580 gate-eligible records.")
    exclusion_counts = Counter(row["fit_exclusion"] for row in records)
    require(
        exclusion_counts
        == {None: 6580, "stage_first_no_previous": 280, "global_final_forced_recompute": 140},
        "Fit exclusions do not match the runtime gate boundary rules.",
    )
    require(
        all(
            row[key] is not None
            and isinstance(row[key], (int, float))
            and math.isfinite(float(row[key]))
            for row in eligible
            for key in ("e_rel_l1", "h_rel_l1", "block_residual_rel_l1")
        ),
        "Eligible records contain null or non-finite scalar values.",
    )

    raw = load_json(coefficient_path)
    require(raw.get("primary_fit_record_count") == 6580, "Raw fit count mismatch.")
    stages = {
        stage: audit_stage(
            stage,
            [row for row in eligible if row["stage"] == stage],
            raw["primary"][stage],
        )
        for stage in ("high", "low")
    }

    equivalence = load_json(equivalence_path)
    require(equivalence.get("status") == "pass", "Source equivalence did not pass.")
    require(
        equivalence.get("calibration_source_sha256") == collection_source,
        "Source equivalence does not describe the collection source.",
    )
    lock = load_json(PROJECT_ROOT / "upstream_lock.json")
    require(
        equivalence.get("reference_source_sha256")
        in lock["wan22"]["approved_calibration_source_sha256"].values(),
        "Source equivalence reference is not an approved locked source.",
    )
    for metric in ("e_rel_l1", "h_rel_l1", "block_residual_rel_l1"):
        summary = equivalence["metric_summary"][metric]
        require(summary["compared"] == 96, f"Source equivalence count failed for {metric}.")
        require(
            summary["max_abs_error"] == 0.0 and summary["max_relative_error"] == 0.0,
            f"Source equivalence is not exact for {metric}.",
        )

    report = {
        "schema": "teacache4wan22_fit_audit_v1",
        "status": "pass",
        "assessment": "share_with_caveats",
        "scope": "fixed-protocol polynomial calibration only; excludes threshold, fidelity, and speed validation",
        "inputs": {
            "audit_script_sha256": sha256(Path(__file__).resolve()),
            "prompts_jsonl_sha256": sha256(manifest_path),
            "prompt_selection_sha256": sha256(selection_path),
            "raw_coefficients_sha256": sha256(coefficient_path),
            "source_equivalence_report_sha256": sha256(equivalence_path),
            "sample_count": len(sample_paths),
            "sample_sha256": {path.name: sha256(path) for path in sample_paths},
        },
        "data_quality": {
            "prompt_count": len(manifest),
            "category_counts": EXPECTED_CATEGORY_COUNTS,
            "raw_record_count": len(records),
            "eligible_record_count": len(eligible),
            "duplicate_record_count": len(record_keys) - len(set(record_keys)),
            "eligible_nonfinite_scalar_count": 0,
            "run_config_count": 1,
            "fit_scope_counts": {
                "eligible": exclusion_counts[None],
                "stage_first_no_previous": exclusion_counts["stage_first_no_previous"],
                "global_final_forced_recompute": exclusion_counts["global_final_forced_recompute"],
            },
            "shard_count": len(shard_payloads),
            "collection_source_sha256": collection_source,
            "failure_file_count": failure_file_count,
        },
        "source_equivalence": {
            "status": equivalence["status"],
            "metric_summary": equivalence["metric_summary"],
        },
        "stages": stages,
        "caveats": [
            "High-stage fit has weak explanatory power and is a heuristic rather than a precise predictor.",
            "The timestep-only x feature has 31 high-stage and 16 low-stage support points.",
            "Raw-power quartics are ill-conditioned and must not be extrapolated outside the locked protocol.",
            "Pooled cond/uncond fitting estimates a balanced shared-gate average.",
            "This audit does not select or validate a TeaCache threshold.",
        ],
    }
    write_reports(result_root, report)
    print(json.dumps({"status": "pass", "assessment": report["assessment"]}, indent=2))


if __name__ == "__main__":
    main()
