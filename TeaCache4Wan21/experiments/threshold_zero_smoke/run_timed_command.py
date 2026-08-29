#!/usr/bin/env python3
"""Run one command while recording precise process wall-clock latency."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after --")
    if args.output.exists() or args.log.exists():
        raise FileExistsError("refusing to overwrite process timing or log output")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started_at = dt.datetime.now(dt.timezone.utc)
    started = time.perf_counter()
    with args.log.open("x", encoding="utf-8") as handle:
        handle.write(f"command: {shlex.join(command)}\n")
        handle.flush()
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finished_at = dt.datetime.now(dt.timezone.utc)
    payload = {
        "schema_version": 1,
        "command": command,
        "command_shell": shlex.join(command),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "process_wall_seconds": time.perf_counter() - started,
        "returncode": result.returncode,
        "log": str(args.log.resolve()),
    }
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


if __name__ == "__main__":
    main()
