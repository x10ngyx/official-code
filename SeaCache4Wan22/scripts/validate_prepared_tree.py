#!/usr/bin/env python3
"""Validate upstream or prepared SeaCache4Wan22 source trees."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
LOCK = json.loads((PROJECT / "upstream_lock.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _class_method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise ValueError(f"missing {class_name}.{method_name}")


def _call_targets(node: ast.AST) -> set[str]:
    return {
        ast.unparse(call.func)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    }


def validate_prepared_model_contract(model_path: Path) -> dict[str, bool]:
    """Require the behavior-reference modulation helper on both execution paths."""
    tree = ast.parse(model_path.read_text(encoding="utf-8"), filename=str(model_path))
    helper = _class_method(tree, "WanAttentionBlock", "_modulated_norm1")
    if [argument.arg for argument in helper.args.args] != ["self", "x", "e"]:
        raise ValueError("WanAttentionBlock._modulated_norm1 has an unexpected signature")
    returns = [node for node in helper.body if isinstance(node, ast.Return)]
    expected = ast.parse(
        "self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)",
        mode="eval",
    ).body
    if len(returns) != 1 or ast.dump(returns[0].value) != ast.dump(expected):
        raise ValueError("WanAttentionBlock._modulated_norm1 differs from work/Wan2.2")

    block_forward = _class_method(tree, "WanAttentionBlock", "forward")
    if "self._modulated_norm1" not in _call_targets(block_forward):
        raise ValueError("WanAttentionBlock.forward bypasses _modulated_norm1")

    model_forward = _class_method(tree, "WanModel", "forward")
    if "self.blocks[0]._modulated_norm1" not in _call_targets(model_forward):
        raise ValueError("WanModel SeaCache feature path bypasses _modulated_norm1")
    return {
        "modulated_norm1_matches_reference": True,
        "block_forward_uses_modulated_norm1": True,
        "seacache_feature_uses_modulated_norm1": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=["upstream", "prepared"])
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if git(source, "rev-parse", "HEAD") != LOCK["wan22"]["commit"]:
        raise ValueError("Wan2.2 commit does not match upstream_lock.json")
    expected = LOCK["wan22"][
        "original_file_sha256" if args.mode == "upstream" else "prepared_file_sha256"
    ]
    observed = {}
    for relative, digest in expected.items():
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[relative] = sha256(path)
        if observed[relative] != digest:
            raise ValueError(
                f"{args.mode} hash mismatch for {relative}: "
                f"expected {digest}, got {observed[relative]}"
            )
        if path.suffix == ".py":
            compile(path.read_bytes(), str(path), "exec")

    integration = LOCK["integration_artifacts"]
    for path_key, hash_key in (
        ("patch", "patch_sha256"),
        ("runtime", "runtime_sha256"),
        ("timing_runtime", "timing_runtime_sha256"),
        ("component_timing", "component_timing_sha256"),
        ("protocol", "protocol_sha256"),
    ):
        path = PROJECT / integration[path_key]
        if sha256(path) != integration[hash_key]:
            raise ValueError(f"integration artifact hash mismatch: {path}")

    forbidden = ("block_cache", "cfg_cache", "zeustimestep", "teacache")
    found = []
    prepared_model_contract = None
    if args.mode == "prepared":
        active_paths = [source / relative for relative in expected]
        active_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in active_paths
            if path.suffix == ".py"
        ).lower()
        found = [marker for marker in forbidden if marker in active_text]
        if found:
            raise ValueError(f"forbidden cache implementation markers found: {found}")
        prepared_model_contract = validate_prepared_model_contract(
            source / "wan" / "modules" / "model.py"
        )

    payload = {
        "schema": "seacache4wan22_prepared_v1",
        "status": "pass",
        "mode": args.mode,
        "source": str(source),
        "wan22_commit": LOCK["wan22"]["commit"],
        "sha256": observed,
        "forbidden_cache_markers": found,
        "prepared_model_contract": prepared_model_contract,
        "patch_sha256": integration["patch_sha256"],
        "runtime_sha256": integration["runtime_sha256"],
        "timing_runtime_sha256": integration["timing_runtime_sha256"],
        "protocol_sha256": integration["protocol_sha256"],
        "integration_artifacts": {
            hash_key: integration[hash_key]
            for hash_key in (
                "patch_sha256",
                "runtime_sha256",
                "timing_runtime_sha256",
                "component_timing_sha256",
                "protocol_sha256",
            )
        },
    }
    if args.write_manifest:
        if args.mode != "prepared":
            raise ValueError("--write-manifest requires --mode prepared")
        target = source / ".seacache4wan22_prepared.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
