#!/usr/bin/env python3
"""CUDA/BF16 original-function parity with small random-weight WanModel instances."""
from __future__ import annotations
import argparse
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "runtime"))
from common import check_environment, external_output, index_result, validate_source, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    check_environment()
    provenance = validate_source(args.source)
    destination = external_output(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    index_result(destination)
    (destination / "README.md").write_text(
        "# Small WanModel CUDA smoke\n\nRandom-weight, reduced-size WanModel blocks; "
        "not an A14B video generation, latency/FLOPs benchmark or quality result. "
        "validation.json records exact official/adapter tensor and decision parity.\n")
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from wan.modules.model import WanModel
    from official import load_official, method_args, official_forward, record_calls, STATE_NAMES
    core = load_official()
    options = method_args()
    def make_pair():
        torch.manual_seed(42)
        pair = [WanModel(model_type="t2v", patch_size=(1, 2, 2), text_len=16,
                          in_dim=16, dim=128, ffn_dim=256, freq_dim=32, text_dim=64,
                          out_dim=16, num_heads=4, num_layers=2).eval().requires_grad_(False)
                .to(device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        for m in pair:
            torch.nn.init.normal_(m.head.head.weight, std=.02)
        return pair
    try:
        pair = make_pair()
        block_calls = [0]
        def count_block(_module, _inputs):
            block_calls[0] += 1
        for model in pair:
            for block in model.blocks:
                block.register_forward_pre_hook(count_block)
        values = [torch.randn(16, 3, 4, 4, device="cuda", dtype=torch.float32) for _ in range(50)]
        context = [torch.randn(8, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        timesteps = core.get_timesteps(shift=12, num_inference_steps=50).to("cuda")
        def generate(models):
            outputs, decisions = [], []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for step in range(50):
                    for b in range(2):
                        m = models[0 if step < 32 else 1]
                        before = block_calls[0]
                        outputs.append(m([values[step]], t=timesteps[step:step+1], context=[context[b]], seq_len=12)[0].clone())
                        decisions.append(block_calls[0] - before)
            return outputs, decisions
        # Direct upstream installation. Save/restore only around the experiment,
        # preserving all original behavior within the 100 calls.
        cls = type(pair[0])
        saved = {n: cls.__dict__.get(n) for n in STATE_NAMES}
        present = {n for n in STATE_NAMES if n in cls.__dict__}
        core.init_magcache(pair[0], core.ratios, options, 32)
        try:
            expected, expected_decisions = generate(pair)
        finally:
            for m in pair:
                for n in STATE_NAMES:
                    if n in m.__dict__:
                        delattr(m, n)
            for n in STATE_NAMES:
                if n in present:
                    setattr(cls, n, saved[n])
                elif n in cls.__dict__:
                    delattr(cls, n)
        trace = []
        with official_forward(pair, core=core, args=options, split_steps=32), record_calls(pair, trace):
            actual, actual_decisions = generate(pair)
        torch.cuda.synchronize()
        exact = sum(torch.equal(a, b) for a, b in zip(expected, actual))
        finite = all(bool(torch.isfinite(t).all()) for t in actual)
        if exact != 100 or not finite or expected_decisions != actual_decisions:
            raise AssertionError(f"CUDA parity failed: {exact}/100, finite={finite}")
        write_json(destination / "validation.json", dict(status="pass", source=provenance,
                   protocol="small_random_weight_wanmodel_cuda_bf16", device=torch.cuda.get_device_name(0),
                   equal_forward_outputs=exact, finite=finite, observed_calls=len(trace),
                   equal_block_execution_decisions=(expected_decisions == actual_decisions),
                   reuse_calls=sum(row["action"] == "reuse" for row in trace),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), calls=trace))
        print(f"CUDA original/adapter parity: {exact}/100, finite={finite}")
    except BaseException as exc:
        write_json(destination / "FAILED.json", dict(status="failed", error=repr(exc)))
        raise

if __name__ == "__main__":
    main()
