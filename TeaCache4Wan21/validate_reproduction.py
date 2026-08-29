#!/usr/bin/env python3
"""Validate the locked TeaCache4Wan2.1 source and local evaluation boundary."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent
LOCK_PATH = PROJECT_DIR / "upstream_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def validate_wan21(root: Path, lock: dict[str, object]) -> dict[str, object]:
    root = root.resolve()
    if not (root / ".git").exists():
        raise ValueError(f"Wan2.1 source is not a Git checkout: {root}")
    commit = git(root, "rev-parse", "HEAD")
    if commit != lock["commit"]:
        raise ValueError(
            f"Wan2.1 commit mismatch: expected {lock['commit']}, got {commit}"
        )
    dirty = git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError("Wan2.1 checkout contains tracked modifications")

    observed: dict[str, str] = {}
    for relative, expected in lock["compatibility_files"].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[relative] = sha256(path)
        if observed[relative] != expected:
            raise ValueError(
                f"Wan2.1 file hash mismatch for {relative}: "
                f"expected {expected}, got {observed[relative]}"
            )
    return {"root": str(root), "commit": commit, "sha256": observed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan21-root", type=Path)
    args = parser.parse_args()

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    method = lock["method"]
    vendored = PROJECT_DIR / method["vendored_path"]
    observed_method_hash = sha256(vendored)
    if observed_method_hash != method["sha256"]:
        raise ValueError(
            f"TeaCache source hash mismatch: expected {method['sha256']}, "
            f"got {observed_method_hash}"
        )
    compile(vendored.read_bytes(), str(vendored), "exec")

    entrypoint = PROJECT_DIR / lock["integration"]["entrypoint"]
    installer = PROJECT_DIR / lock["integration"]["installer"]
    entrypoint_text = entrypoint.read_text(encoding="utf-8")
    installer_text = installer.read_text(encoding="utf-8")
    compile(entrypoint_text, str(entrypoint), "exec")
    installer_tree = ast.parse(installer_text, filename=str(installer))

    baseline_block = (
        "if not wrapper_args.enable_teacache and wrapper_args.timing_json is None:\n"
        "        original.generate(args)\n"
        "        return"
    )
    if baseline_block not in entrypoint_text:
        raise ValueError("entrypoint baseline is not a direct original Wan2.1 call")
    if "from teacache import patch_pipeline_construction" not in entrypoint_text:
        raise ValueError("entrypoint does not lazily import the TeaCache installer")

    official_imports: set[str] = set()
    coefficient_tables: dict[str, object] = {}
    model_assignments: set[str] = set()
    for node in ast.walk(installer_tree):
        if isinstance(node, ast.ImportFrom) and node.module == "upstream.teacache_generate":
            official_imports.update(alias.name for alias in node.names)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "model"
                ):
                    model_assignments.add(target.attr)
                if isinstance(target, ast.Name) and target.id in {
                    "T2V_COEFFICIENTS",
                    "I2V_COEFFICIENTS",
                }:
                    coefficient_tables[target.id] = ast.literal_eval(node.value)
    required_functions = {"t2v_generate", "i2v_generate", "teacache_forward"}
    if official_imports != required_functions:
        raise ValueError(
            f"installer must import exactly the official method functions: {official_imports}"
        )
    expected_coefficients = {
        "T2V_COEFFICIENTS": {
            ("1.3B", False): [2396.76752, -1311.10545, 201.331979, -8.29855975, 0.137887774],
            ("14B", False): [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
            ("1.3B", True): [-52186.2437, 9230.41404, -528.275948, 13.6987616, -0.0499875664],
            ("14B", True): [-303318.725, 49053.7029, -2655.30556, 58.7365115, -0.315583525],
        },
        "I2V_COEFFICIENTS": {
            ("480P", False): [-302.33167, 223.948934, -52.546397, 5.8734844, -0.201973289],
            ("720P", False): [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
            ("480P", True): [257151.496, -35422.9917, 1402.86849, -13.5890334, 0.132517977],
            ("720P", True): [8107.0546, 2133.93892, -372.934672, 16.6203073, -0.0417769401],
        },
    }
    if coefficient_tables != expected_coefficients:
        raise ValueError("active TeaCache coefficient tables differ from the official source")
    required_model_state = {
        "forward",
        "enable_teacache",
        "cnt",
        "num_steps",
        "teacache_thresh",
        "accumulated_rel_l1_distance_even",
        "accumulated_rel_l1_distance_odd",
        "previous_e0_even",
        "previous_e0_odd",
        "previous_residual_even",
        "previous_residual_odd",
        "use_ref_steps",
        "coefficients",
        "ret_steps",
        "cutoff_steps",
    }
    if not required_model_state.issubset(model_assignments):
        raise ValueError(
            "active TeaCache state differs from official setup; missing "
            f"{sorted(required_model_state - model_assignments)}"
        )
    required_formulas = (
        "model.num_steps = sample_steps * 2",
        "model.ret_steps = (5 if use_ret_steps else 1) * 2",
        "sample_steps * 2 if use_ret_steps else sample_steps * 2 - 2",
    )
    for formula in required_formulas:
        if formula not in installer_text:
            raise ValueError(f"active TeaCache schedule formula mismatch: {formula}")

    license_path = PROJECT_DIR / lock["wan21"]["license_file"]
    observed_license_hash = sha256(license_path)
    if observed_license_hash != lock["wan21"]["license_file_sha256"]:
        raise ValueError(
            f"upstream license hash mismatch: expected "
            f"{lock['wan21']['license_file_sha256']}, got {observed_license_hash}"
        )

    paired_metrics = (PROJECT_DIR / lock["evaluation_boundary"]["paired_metrics"]).resolve()
    vbench = (PROJECT_DIR / lock["evaluation_boundary"]["vbench"]).resolve()
    expected_paired = (REPOSITORY_DIR / "VideoMetrics" / "run_evaluation.sh").resolve()
    expected_vbench = (REPOSITORY_DIR / "VbenchEvaluation" / "run_vbench200.sh").resolve()
    if paired_metrics != expected_paired or not paired_metrics.is_file():
        raise ValueError(f"paired-metric boundary is not the local VideoMetrics tool: {paired_metrics}")
    if vbench != expected_vbench or not vbench.is_file():
        raise ValueError(f"VBench boundary is not the local VbenchEvaluation tool: {vbench}")

    evaluation_runner = PROJECT_DIR / "experiments" / "vbench200_t2v" / "evaluate_results.sh"
    runner_text = evaluation_runner.read_text(encoding="utf-8")
    required_markers = (
        "VideoMetrics/run_evaluation.sh",
        "VbenchEvaluation/run_vbench200.sh",
    )
    for marker in required_markers:
        if marker not in runner_text:
            raise ValueError(f"evaluation runner does not use required local tool: {marker}")
    forbidden_markers = ("eval/teacache", "common_metrics")
    for marker in forbidden_markers:
        if marker in runner_text:
            raise ValueError(f"evaluation runner references forbidden TeaCache evaluator: {marker}")

    report: dict[str, object] = {
        "status": "ok",
        "teacache": {
            "commit": method["commit"],
            "source_blob": method["source_blob"],
            "sha256": observed_method_hash,
            "byte_exact": True,
        },
        "integration": {
            "entrypoint": str(entrypoint),
            "installer": str(installer),
            "baseline_uses_original_wan21": True,
            "official_functions_bound": sorted(official_imports),
            "official_coefficients_match": True,
            "official_state_and_schedule_match": True,
        },
        "evaluation": {
            "paired_metrics": str(paired_metrics),
            "vbench": str(vbench),
            "official_teacache_evaluation_used": False,
        },
        "license": {
            "path": str(license_path),
            "sha256": observed_license_hash,
        },
    }
    if args.wan21_root is not None:
        report["wan21"] = validate_wan21(args.wan21_root, lock["wan21"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
