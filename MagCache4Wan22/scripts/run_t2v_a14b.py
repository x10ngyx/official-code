#!/usr/bin/env python3
"""Generate one fixed-protocol baseline, official MagCache, or calibration video."""
from __future__ import annotations

import argparse
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "runtime"))
from common import (artifact, check_environment, checkpoint_path, external_output,
                    index_result, read_json, validate_config, validate_source, write_json)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--sample-id", default="sample001")
    parser.add_argument("--mode", choices=("baseline", "magcache", "calibrate"), default="magcache")
    parser.add_argument("--threshold", type=float, default=.06)
    parser.add_argument("--magcache-k", type=int, default=2)
    parser.add_argument("--retention-ratio", type=float, default=.4)
    parser.add_argument("--dry-run", action="store_true", help="verify sources and print the run plan without loading weights")
    return parser.parse_args()


def main():
    args = parse_args()
    check_environment()
    if not args.prompt.strip():
        raise ValueError("prompt must not be empty")
    if not args.sample_id or Path(args.sample_id).name != args.sample_id or args.sample_id in {".", ".."}:
        raise ValueError("sample-id must be a plain filename stem")
    source = args.source.expanduser().resolve()
    provenance = validate_source(source)
    checkpoint = checkpoint_path(args.checkpoint)
    destination = external_output(args.output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite run: {destination}")
    protocol = read_json(PROJECT / "configs/wan22_t2v_a14b_50step_dpmpp.json")
    sys.path.insert(0, str(source))
    import torch
    from wan.configs import WAN_CONFIGS
    from official import load_official, method_args

    cfg = WAN_CONFIGS["t2v-A14B"]
    validate_config(cfg)
    core = load_official()
    options = method_args(args.threshold, args.magcache_k, args.retention_ratio)
    # Deliberately use upstream MagCache's helper, including its sigma_min=.01.
    inferred_timesteps = core.get_timesteps(shift=12, num_inference_steps=50)
    split_steps = int((inferred_timesteps >= cfg.num_train_timesteps * cfg.boundary).sum())
    from generation import create_pipeline, generate_run, make_plan
    plan = make_plan(provenance, checkpoint,
                     dict(sample_id=args.sample_id, prompt_en=args.prompt), args.mode, options, split_steps)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Wan2.2 generation requires CUDA")
    pipeline, init_seconds = create_pipeline(checkpoint)
    path = generate_run(pipeline, core, options, plan, destination, init_seconds=init_seconds)
    print(json.dumps(dict(status="complete", manifest=str(path))))


if __name__ == "__main__":
    main()
