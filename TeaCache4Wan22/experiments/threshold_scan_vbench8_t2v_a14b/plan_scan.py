#!/usr/bin/env python3
"""Validate the frozen scan inputs and render deterministic worker assignments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "scan_config.json"
PROMPTS_PATH = SCRIPT_DIR / "prompts.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def load_prompts(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"prompt row {line_number} is not an object")
        rows.append(row)
    return rows


def validate_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_json(CONFIG_PATH)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported scan config schema")
    prompts = load_prompts(PROMPTS_PATH)
    sample_ids = [row.get("sample_id") for row in prompts]
    if len(prompts) != 11 or len(set(sample_ids)) != len(prompts):
        raise ValueError("the scan must contain eleven unique prompts")
    if sample_ids != config["prompt_selection"]["sample_ids"]:
        raise ValueError("prompt snapshot order differs from scan config")

    source = REPOSITORY_DIR / "Vbench200" / "prompts.jsonl"
    if sha256(source) != config["prompt_selection"]["source_sha256"]:
        raise ValueError("Vbench200 prompt source SHA256 mismatch")
    source_rows = {row["sample_id"]: row for row in load_prompts(source)}
    for row in prompts:
        if source_rows.get(row["sample_id"]) != row:
            raise ValueError(f"prompt snapshot differs from Vbench200: {row['sample_id']}")
    covered_dimensions = {
        dimension for row in prompts for dimension in row.get("dimension", [])
    }
    expected_dimensions = {
        dimension for row in source_rows.values() for dimension in row.get("dimension", [])
    }
    if covered_dimensions != expected_dimensions:
        raise ValueError(
            "the selected prompt subset must cover every VBench dimension; "
            f"missing={sorted(expected_dimensions - covered_dimensions)}"
        )

    thresholds = config.get("thresholds")
    expected = [
        0.15,
        0.175,
        0.2,
        0.225,
        0.25,
        0.275,
        0.3,
        0.325,
        0.35,
        0.375,
        0.4,
        0.45,
        0.5,
        0.6,
        0.7,
        0.8,
    ]
    if thresholds != expected:
        raise ValueError("threshold grid differs from the frozen dense-to-sparse grid")
    if config.get("target_speedups") != [1.8, 2.4, 3.0]:
        raise ValueError("target speedups differ from the requested targets")

    protocol = PROJECT_DIR / "configs" / "wan22_t2v_a14b_50step_dpmpp.json"
    coefficients = (
        PROJECT_DIR
        / "coefficients"
        / "wan22_t2v_a14b_50step_dpmpp_nonretention.json"
    )
    if sha256(protocol) != config["protocol"]["sha256"]:
        raise ValueError("protocol SHA256 mismatch")
    if sha256(coefficients) != config["coefficients"]["sha256"]:
        raise ValueError("coefficient SHA256 mismatch")
    return config, prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if args.worker_count < 1 or args.worker_count > 8:
        raise ValueError("--worker-count must be in [1, 8]")
    config, prompts = validate_inputs()
    workers = []
    for worker_index in range(args.worker_count):
        assigned = [
            row["sample_id"]
            for index, row in enumerate(prompts)
            if index % args.worker_count == worker_index
        ]
        workers.append({"worker_index": worker_index, "sample_ids": assigned})
    payload = {
        "schema_version": 1,
        "prompt_count": len(prompts),
        "threshold_count": len(config["thresholds"]),
        "measured_generation_count": len(prompts) * (1 + len(config["thresholds"])),
        "warmup_generation_count": args.worker_count,
        "worker_count": args.worker_count,
        "workers": workers,
    }
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=None if args.compact else 2,
            separators=(",", ":") if args.compact else None,
        )
    )


if __name__ == "__main__":
    main()
