#!/usr/bin/env python3
"""Validate an original or TeaCache-patched Wan2.2 source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("upstream", "prepared"), required=True
    )
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    lock = json.loads((project_root / "upstream_lock.json").read_text(encoding="utf-8"))
    source = args.source.resolve()
    expected_commit = lock["wan22"]["commit"]
    observed_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if observed_commit != expected_commit:
        raise RuntimeError(
            f"Wan2.2 commit mismatch: expected {expected_commit}, observed {observed_commit}"
        )

    original_hashes = lock["wan22"]["original_file_sha256"]
    patch_path = project_root / "patches" / "wan22_42bf4cf_teacache.patch"
    runtime_path = project_root / "runtime" / "teacache.py"
    timing_runtime_path = project_root / "runtime" / "inference_timing.py"
    protocol_path = project_root / "configs" / "wan22_t2v_a14b_50step_dpmpp.json"
    if args.mode == "upstream":
        observed = {relative: sha256(source / relative) for relative in original_hashes}
        if observed != original_hashes:
            raise RuntimeError(
                "Original Wan2.2 file hashes do not match the lock: "
                + json.dumps(observed, sort_keys=True)
            )
        subprocess.run(
            ["git", "apply", "--check", str(patch_path)], cwd=source, check=True
        )
    else:
        installed_runtime = source / "wan" / "teacache.py"
        installed_timing_runtime = source / "wan" / "inference_timing.py"
        if sha256(runtime_path) != sha256(installed_runtime):
            raise RuntimeError("Installed wan/teacache.py differs from the canonical runtime.")
        if sha256(timing_runtime_path) != sha256(installed_timing_runtime):
            raise RuntimeError(
                "Installed wan/inference_timing.py differs from the canonical runtime."
            )
        patched_files = ("generate.py", "wan/modules/model.py", "wan/text2video.py")
        observed_names = set(
            subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=source,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.splitlines()
        )
        if observed_names != set(patched_files):
            raise RuntimeError(
                f"Prepared tracked-file set differs from the canonical patch: {sorted(observed_names)}"
            )
        observed_patch = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--binary", "--", *patched_files],
            cwd=source,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        if sha256_bytes(observed_patch) != sha256(patch_path):
            raise RuntimeError("Prepared tracked diff differs from the canonical patch.")
        marker_requirements = {
            "generate.py": (
                "--teacache_threshold",
                "TeaCacheConfig",
                "patch_pipeline_timing",
            ),
            "wan/text2video.py": (
                "TeaCacheController",
                "teacache.write_trace",
            ),
            "wan/modules/model.py": (
                "teacache.plan_step",
                "teacache.record_recompute",
            ),
        }
        for relative, markers in marker_requirements.items():
            text = (source / relative).read_text(encoding="utf-8")
            missing = [marker for marker in markers if marker not in text]
            if missing:
                raise RuntimeError(f"Prepared file {relative} is missing markers {missing}")
        subprocess.run(["git", "diff", "--check"], cwd=source, check=True)
        for relative in (
            "generate.py",
            "wan/text2video.py",
            "wan/modules/model.py",
            "wan/teacache.py",
            "wan/inference_timing.py",
        ):
            py_compile.compile(str(source / relative), doraise=True)

    payload = {
        "schema": "teacache4wan22_prepared_tree_validation_v1",
        "mode": args.mode,
        "source": str(source),
        "wan22_commit": observed_commit,
        "patch_sha256": sha256(patch_path),
        "runtime_sha256": sha256(runtime_path),
        "timing_runtime_sha256": sha256(timing_runtime_path),
        "protocol_sha256": sha256(protocol_path),
        "status": "pass",
    }
    expected_artifacts = lock["integration_artifacts"]
    if payload["patch_sha256"] != expected_artifacts["patch_sha256"]:
        raise RuntimeError("Integration patch SHA256 differs from upstream_lock.json.")
    if payload["runtime_sha256"] != expected_artifacts["runtime_sha256"]:
        raise RuntimeError("Runtime SHA256 differs from upstream_lock.json.")
    if payload["timing_runtime_sha256"] != expected_artifacts[
        "timing_runtime_sha256"
    ]:
        raise RuntimeError("Timing runtime SHA256 differs from upstream_lock.json.")
    if payload["protocol_sha256"] != expected_artifacts["protocol_sha256"]:
        raise RuntimeError("Protocol SHA256 differs from upstream_lock.json.")
    if args.write_manifest:
        target = source / ".teacache4wan22_prepared.json"
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
