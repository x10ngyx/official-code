#!/usr/bin/env python3
"""Run locked original Wan2.1, optionally injecting official TeaCache."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


EXPECTED_WAN21_GENERATE_SHA256 = (
    "f4aae5a3edafa9522ccbbc19d200928035f0a594d125b8a113c04f031eeeeeb0"
)
REPOSITORY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_DIR / "ComponentMetrics"))
from fixed_protocol import validate_wan21_t2v_1_3b_args  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wan21_root", type=Path)
    parser.add_argument("--enable_teacache", action="store_true")
    parser.add_argument("--teacache_thresh", type=float)
    parser.add_argument("--use_ret_steps", action="store_true")
    parser.add_argument("--timing_json", type=Path)
    wrapper_args, wan_args = parser.parse_known_args(argv)

    if not wrapper_args.enable_teacache and (
        wrapper_args.teacache_thresh is not None or wrapper_args.use_ret_steps
    ):
        parser.error("TeaCache options require --enable_teacache")
    if wrapper_args.enable_teacache:
        if wrapper_args.teacache_thresh is None:
            wrapper_args.teacache_thresh = 0.2
        if wrapper_args.teacache_thresh < 0:
            parser.error("--teacache_thresh must be non-negative")
    return wrapper_args, wan_args


def resolve_wan21_root(wrapper_args: argparse.Namespace) -> Path:
    configured = wrapper_args.wan21_root or os.environ.get("WAN21_ROOT")
    root = Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()
    generate_path = root / "generate.py"
    if not generate_path.is_file() or not (root / "wan" / "__init__.py").is_file():
        raise FileNotFoundError(f"not a Wan2.1 source tree: {root}")
    observed = sha256(generate_path)
    if observed != EXPECTED_WAN21_GENERATE_SHA256:
        raise ValueError(
            "Wan2.1 generate.py hash mismatch: expected "
            f"{EXPECTED_WAN21_GENERATE_SHA256}, got {observed}"
        )
    return root


def load_original_generate(wan21_root: Path) -> ModuleType:
    source = wan21_root / "generate.py"
    sys.path.insert(0, str(wan21_root))
    spec = importlib.util.spec_from_file_location("_locked_wan21_generate", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load original Wan2.1 entry point: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    wrapper_args, wan_argv = parse_wrapper_args(sys.argv[1:])
    wan21_root = resolve_wan21_root(wrapper_args)
    original = load_original_generate(wan21_root)

    saved_argv = sys.argv
    try:
        sys.argv = [str(wan21_root / "generate.py"), *wan_argv]
        args = original._parse_args()
    finally:
        sys.argv = saved_argv
    validate_wan21_t2v_1_3b_args(args)

    if not wrapper_args.enable_teacache and wrapper_args.timing_json is None:
        original.generate(args)
        return

    with ExitStack() as stack:
        if wrapper_args.enable_teacache:
            from teacache import patch_pipeline_construction

            stack.enter_context(
                patch_pipeline_construction(
                    original.wan,
                    task=args.task,
                    checkpoint_dir=args.ckpt_dir,
                    sample_steps=args.sample_steps,
                    threshold=wrapper_args.teacache_thresh,
                    use_ret_steps=wrapper_args.use_ret_steps,
                )
            )
        if wrapper_args.timing_json is not None:
            from inference_timing import patch_pipeline_timing

            stack.enter_context(
                patch_pipeline_timing(
                    original.wan,
                    task=args.task,
                    output_path=wrapper_args.timing_json,
                    implementation=(
                        "teacache" if wrapper_args.enable_teacache else "wan21"
                    ),
                )
            )
        original.generate(args)


if __name__ == "__main__":
    main()
