#!/usr/bin/env python3
"""Frozen RGB full-reference replacement for the historical PSNR helper."""

from __future__ import annotations

import argparse
import json
import os

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

from video_metrics.compat import write_psnr_contract
from video_metrics.core import PROTOCOL_ID


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", choices=(PROTOCOL_ID,), default=PROTOCOL_ID)
    args = parser.parse_args()
    result = write_psnr_contract(args.reference, args.candidate, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "per_frame_psnr"}, indent=2))


if __name__ == "__main__":
    main()
