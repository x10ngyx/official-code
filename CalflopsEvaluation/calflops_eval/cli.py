from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adapters import load_profile_items
from .aggregation import aggregate_trace, load_trace_rows, read_json
from .profiling import profile_items


def _write_json(path: str | Path, payload: Any, *, overwrite: bool) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calflops-eval",
        description="Profile component FLOPs and aggregate cache traces.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="Profile adapter-defined components")
    profile.add_argument("--adapter", required=True, help="MODULE_OR_FILE.py:FACTORY")
    profile.add_argument("--output", required=True)
    profile.add_argument("--print-detailed", action="store_true")
    profile.add_argument("--overwrite", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="Aggregate component costs over traces")
    aggregate.add_argument("--cost-table", required=True)
    aggregate.add_argument("--mapping", required=True)
    aggregate.add_argument("--trace", action="append", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "profile":
        payload = profile_items(
            load_profile_items(args.adapter), print_detailed=args.print_detailed
        )
        _write_json(args.output, payload, overwrite=args.overwrite)
    elif args.command == "aggregate":
        payload = aggregate_trace(
            cost_table=read_json(args.cost_table),
            mapping=read_json(args.mapping),
            rows=load_trace_rows(args.trace),
        )
        _write_json(args.output, payload, overwrite=args.overwrite)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
