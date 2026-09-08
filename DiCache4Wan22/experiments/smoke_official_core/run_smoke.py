#!/usr/bin/env python3
"""Real CUDA attention/BF16 smoke and small-shape Calflops path validation."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "runtime"))
from common import check_environment, external_output, index_result, validate_source, write_json
from official import load_official, method_args, official_forward


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    check_environment()
    provenance = validate_source(args.source)
    root = external_output(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    index_result(root)
    (root / "README.md").write_text(
        "# Small WanModel CUDA/BF16 validation\n\nRandom weights, reduced tensor shapes and real CUDA attention. "
        "validation.json records native all-full equivalence, repeated-cache determinism and observed probe calls; "
        "small_calflops.json checks measured full/probe/head cost separation. "
        "These are implementation checks, not A14B video quality, latency or peak-memory results.\n")
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from wan.modules.model import WanModel
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    def make_model():
        m = WanModel(model_type="t2v", patch_size=(1, 2, 2), text_len=16,
                     in_dim=16, dim=128, ffn_dim=256, freq_dim=32, text_dim=64,
                     out_dim=16, num_heads=4, num_layers=4).eval().requires_grad_(False)
        m.to(device="cuda", dtype=torch.bfloat16)
        torch.nn.init.normal_(m.head.head.weight, std=.02)
        return m
    pair = [make_model(), make_model()]
    latent = torch.randn(16, 3, 4, 4, device="cuda")
    values = [latent + i*.005 for i in range(50)]
    contexts = [torch.randn(8, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    timesteps = torch.linspace(1000, 1, 50, device="cuda")
    def run():
        outputs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(50):
                for branch in range(2):
                    m = pair[0 if i < 32 else 1]
                    outputs.append(m([values[i]], t=timesteps[i:i+1], context=[contexts[branch]], seq_len=12)[0].clone())
        return outputs
    try:
        native = run()
        core = load_official()
        with official_forward(pair, core=core, args=method_args(threshold=0)):
            full = run()
        full_equal = sum(torch.equal(a, b) for a, b in zip(native, full))
        traces, cached = [], []
        for _ in range(2):
            records = []
            with official_forward(pair, core=core, args=method_args(threshold=100), records=records):
                cached.append(run())
            traces.append(records)
        repeat_equal = sum(torch.equal(a, b) for a, b in zip(*cached))
        finite = all(torch.isfinite(x).all().item() for x in full+cached[0])
        if full_equal != 100 or repeat_equal != 100 or traces[0] != traces[1] or not finite:
            raise AssertionError("CUDA/BF16 native/full or repeated-cache comparison failed")
        calls = traces[0]
        if [c["blocks_executed"] for c in calls[64:68]] != [4, 4, 1, 1]:
            raise AssertionError("new-expert initialization or probe fallback failed")
        # Exercise the actual Calflops path splitter on small real WanModel
        # instances without saving any random weights or loading A14B weights.
        sys.path.insert(0, str(PROJECT / "experiments/performance_t2v_a14b"))
        from profile_calflops import load_calflops, profile_stage
        calculate, tool = load_calflops(None)
        profile, _ = profile_stage(
            model_class=SimpleNamespace(from_pretrained=lambda *a, **k: make_model()),
            checkpoint_dir=Path("unused_random_weights"), subfolder="random", stage="high",
            timestep_value=900., latent=latent, contexts=dict(zip(("cond", "uncond"), contexts)),
            seq_len=12, device=torch.device("cuda:0"), calculate_flops_fn=calculate)
        for value in profile["branches"].values():
            if not value["estimated_always_on_flops"] < value["estimated_probe_flops"] < value["estimated_full_flops"]:
                raise AssertionError("probe must have a measured nonzero block cost")
        write_json(root / "small_calflops.json", dict(tool=tool, profile=profile))
        torch.cuda.synchronize()
        write_json(root / "validation.json", dict(status="pass", source=provenance,
                   scope="small random-weight Wan2.2 CUDA/BF16 implementation checks",
                   device=torch.cuda.get_device_name(0), native_full_equal_calls=full_equal,
                   cached_repeat_equal_calls=repeat_equal, finite=finite,
                   reuse_calls=sum(c["action"] == "reuse" for c in calls),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), calls=calls))
        print(f"CUDA/BF16 native/full={full_equal}/100; cached repeat={repeat_equal}/100; probe Calflops verified")
    except BaseException as exc:
        write_json(root / "FAILED.json", dict(status="failed", error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
