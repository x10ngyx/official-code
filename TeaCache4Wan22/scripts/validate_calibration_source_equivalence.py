#!/usr/bin/env python3
"""Validate scalar calibration records against an approved Wan2.2 source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


METRIC_KEYS = ("e_rel_l1", "h_rel_l1", "block_residual_rel_l1")
IDENTITY_KEYS = ("stage", "branch", "global_step", "stage_step", "timestep")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-result-root", type=Path, required=True)
    parser.add_argument("--reference-sample", type=Path, required=True)
    parser.add_argument("--reference-wan-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    lock = load_json(project_root / "upstream_lock.json")
    reference_source = args.reference_wan_source.resolve()
    validation_mode = (
        "prepared"
        if (reference_source / ".teacache4wan22_prepared.json").is_file()
        else "upstream"
    )
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "validate_prepared_tree.py"),
            "--source",
            str(reference_source),
            "--mode",
            validation_mode,
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    reference_source_hashes = {
        "wan/modules/model.py": sha256(reference_source / "wan/modules/model.py"),
        "wan/text2video.py": sha256(reference_source / "wan/text2video.py"),
    }
    approved = lock["wan22"]["approved_calibration_source_sha256"]
    if reference_source_hashes not in approved.values():
        raise ValueError("reference source is not an approved upstream/prepared Wan2.2 tree")

    shard_payloads = [
        load_json(path)
        for path in sorted((args.calibration_result_root / "shards").glob("shard_*.json"))
    ]
    source_pairs = {
        (
            payload.get("wan_model_py_sha256"),
            payload.get("wan_text2video_py_sha256"),
        )
        for payload in shard_payloads
    }
    if len(shard_payloads) != 4 or len(source_pairs) != 1:
        raise ValueError("calibration result does not contain one consistent four-shard source")
    calibration_model_sha, calibration_text_sha = next(iter(source_pairs))
    calibration_source_hashes = {
        "wan/modules/model.py": calibration_model_sha,
        "wan/text2video.py": calibration_text_sha,
    }

    reference = load_json(args.reference_sample)
    sample_id = reference.get("prompt", {}).get("sample_id")
    candidates = [
        path
        for path in (args.calibration_result_root / "samples").glob("*.json")
        if load_json(path).get("prompt", {}).get("sample_id") == sample_id
    ]
    if len(candidates) != 1:
        raise ValueError(f"could not resolve one calibration sample for {sample_id!r}")
    calibration_path = candidates[0]
    calibration = load_json(calibration_path)
    if calibration.get("prompt") != reference.get("prompt"):
        raise ValueError("calibration and reference prompt records differ")
    if calibration.get("run_config") != reference.get("run_config"):
        raise ValueError("calibration and reference run configurations differ")
    left_records = calibration.get("records", [])
    right_records = reference.get("records", [])
    if len(left_records) != 100 or len(right_records) != 100:
        raise ValueError("both source-equivalence samples must contain 100 records")

    metric_summary = {}
    for key in METRIC_KEYS:
        max_abs = 0.0
        max_rel = 0.0
        compared = 0
        for index, (left, right) in enumerate(zip(left_records, right_records)):
            if any(left.get(name) != right.get(name) for name in IDENTITY_KEYS):
                raise ValueError(f"record identity differs at row {index}")
            left_value = left.get(key)
            right_value = right.get(key)
            if left_value is None or right_value is None:
                if left_value is not None or right_value is not None:
                    raise ValueError(f"{key} missingness differs at row {index}")
                continue
            left_float = float(left_value)
            right_float = float(right_value)
            if not math.isfinite(left_float) or not math.isfinite(right_float):
                raise ValueError(f"non-finite {key} at row {index}")
            absolute = abs(left_float - right_float)
            relative = absolute / max(abs(right_float), args.atol)
            max_abs = max(max_abs, absolute)
            max_rel = max(max_rel, relative)
            compared += 1
            if not math.isclose(
                left_float,
                right_float,
                rel_tol=args.rtol,
                abs_tol=args.atol,
            ):
                raise ValueError(
                    f"{key} differs at row {index}: calibration={left_float}, reference={right_float}"
                )
        metric_summary[key] = {
            "compared": compared,
            "max_abs_error": max_abs,
            "max_relative_error": max_rel,
        }

    payload = {
        "schema": "teacache4wan22_calibration_source_equivalence_v1",
        "status": "pass",
        "wan22_commit": lock["wan22"]["commit"],
        "sample_id": sample_id,
        "rtol": args.rtol,
        "atol": args.atol,
        "calibration_source_sha256": calibration_source_hashes,
        "reference_source_sha256": reference_source_hashes,
        "calibration_sample": {
            "filename": calibration_path.name,
            "sha256": sha256(calibration_path),
        },
        "reference_sample": {
            "filename": args.reference_sample.name,
            "sha256": sha256(args.reference_sample),
        },
        "metric_summary": metric_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
