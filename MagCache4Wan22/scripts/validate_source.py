#!/usr/bin/env python3
"""Verify pristine Wan2.2 and the pinned MagCache package."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from common import check_environment, validate_source, write_json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    check_environment()
    payload = validate_source(args.source)
    if args.write_manifest:
        write_json(args.source / ".magcache4wan22_prepared.json", payload)
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
