#!/usr/bin/env python3
"""Generate one fixed-protocol native baseline or DiCache video."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import json
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "runtime"))
from common import check_environment, checkpoint_path, external_output, validate_source
from official import load_official, method_args
from generation import create_pipeline, generate_run, make_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--sample-id", default="sample001")
    p.add_argument("--mode", choices=("baseline", "dicache"), default="dicache")
    p.add_argument("--threshold", type=float, default=.08)
    p.add_argument("--retention-ratio", type=float, default=.2)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    check_environment()
    if not args.prompt.strip() or not args.sample_id or Path(args.sample_id).name != args.sample_id or args.sample_id in {".", ".."}:
        raise ValueError("invalid prompt or sample ID")
    provenance = validate_source(args.source)
    checkpoint = checkpoint_path(args.checkpoint)
    destination = external_output(args.output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    options = method_args(args.threshold, args.retention_ratio)
    core = load_official()
    plan = make_plan(provenance, checkpoint, dict(sample_id=args.sample_id, prompt_en=args.prompt), args.mode, options)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    sys.path.insert(0, provenance["source"])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Wan2.2 generation requires CUDA")
    pipeline, seconds = create_pipeline(checkpoint)
    manifest = generate_run(pipeline, core, options, plan, destination, init_seconds=seconds)
    print(json.dumps(dict(status="complete", manifest=str(manifest))))


if __name__ == "__main__":
    main()
