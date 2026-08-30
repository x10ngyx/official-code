#!/usr/bin/env python3
"""Write a machine-readable manifest for one fixed-protocol generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from compare_runs import validate_timing_payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--coefficients", type=Path)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    protocol_path = project_root / "configs" / "wan22_t2v_a14b_50step_dpmpp.json"
    prepared_manifest_path = args.source / ".teacache4wan22_prepared.json"
    required_files = [
        protocol_path,
        prepared_manifest_path,
        args.video,
        args.timing,
        args.log,
    ]
    if args.coefficients:
        required_files.append(args.coefficients)
    if args.trace:
        required_files.append(args.trace)
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    if not math.isfinite(args.threshold) or args.threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    if (args.threshold > 0) != bool(args.coefficients and args.trace):
        raise ValueError(
            "positive thresholds require both coefficients and trace; baseline requires neither"
        )

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    prepared_payload = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    if prepared_payload.get("status") != "pass" or prepared_payload.get("mode") != "prepared":
        raise ValueError("invalid prepared-source validation manifest")
    if prepared_payload.get("protocol_sha256") != sha256(protocol_path):
        raise ValueError("prepared source and run manifest use different protocol locks")
    timing_payload = json.loads(args.timing.read_text(encoding="utf-8"))
    expected_method = "teacache" if args.threshold > 0 else "none"
    validate_timing_payload(timing_payload, expected_method)
    payload = {
        "schema": "teacache4wan22_run_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_source": str(args.source.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "threshold": args.threshold,
        "prompt": args.prompt,
        "prepared_source_manifest": {
            "path": str(prepared_manifest_path.resolve()),
            "sha256": sha256(prepared_manifest_path),
            "payload": prepared_payload,
        },
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256(protocol_path),
            "payload": protocol,
        },
        "coefficients": (
            {
                "path": str(args.coefficients.resolve()),
                "sha256": sha256(args.coefficients),
            }
            if args.coefficients
            else None
        ),
        "video": {
            "path": str(args.video.resolve()),
            "sha256": sha256(args.video),
        },
        "trace": (
            {"path": str(args.trace.resolve()), "sha256": sha256(args.trace)}
            if args.trace
            else None
        ),
        "timing": {
            "path": str(args.timing.resolve()),
            "sha256": sha256(args.timing),
            "payload": timing_payload,
        },
        "log": {"path": str(args.log.resolve()), "sha256": sha256(args.log)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
