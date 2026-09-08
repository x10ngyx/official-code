#!/usr/bin/env python3
"""Prepare corrected targets and import only validated unchanged candidates."""
import argparse
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vbench200_reuse_gpu123"))
import run_gpu123 as base
from rebalance_gpu123 import target_context, validate_any_gpu, build_assignment
from common import artifact, external_output, index_result, read_json, write_json
from suite import ensure_profile, link_video

TARGETS = (
    dict(label="1p8x", target_speedup=1.8, gpu="1", threshold=.075, retention_ratio=.2),
    dict(label="2p4x", target_speedup=2.4, gpu="2", threshold=.172, retention_ratio=.2),
    dict(label="3p0x", target_speedup=3., gpu="3", threshold=.396, retention_ratio=.2),
)


def prepare(previous, output):
    output = external_output(output)
    if output.exists():
        raise FileExistsError(output)
    if any(not base.gpu_is_idle(gpu) for gpu in ("1", "2", "3")):
        raise RuntimeError("stop previous GPU1/2/3 workers before importing immutable results")
    args = argparse.Namespace(source=base.DEFAULT_SOURCE.resolve(),
        checkpoint=base.DEFAULT_CHECKPOINT.resolve(), prompts=base.DEFAULT_PROMPTS,
        calflops_profile=base.DEFAULT_PROFILE, baseline_audit=base.DEFAULT_BASELINE_AUDIT,
        worker_ready_timeout=1800)
    plans = {}
    for target in TARGETS:
        plan, _ = base.make_plan(args, target)
        plan["runner"] = "persistent balanced workers across physical GPU1/2/3"
        plan["target_label_note"] = "User-corrected thresholds .075/.172/.396; actual speeds measured after completion."
        plan["estimated_generate_seconds"] = {"1p8x": 547.05, "2p4x": 401.62, "3p0x": 341.36}[target["label"]]
        plan["balanced_experiment_source"] = {p.name: artifact(p) for p in sorted(HERE.glob("*.py"))}
        name = f"target_{target['label']}_gpu{target['gpu']}"
        plans[name] = plan
    output.mkdir(parents=True)
    index_result(output)
    for name, plan in plans.items():
        root = output / name
        root.mkdir()
        write_json(root / "plan.json", plan)
        ensure_profile(root, plan)
        write_json(root / "baseline_reuse.json", plan["baseline_reuse"])
        for prompt in plan["prompts"]:
            sid = prompt["sample_id"]
            link_video(root / "videos/baseline" / f"{sid}.mp4",
                       Path(plan["baseline_reuse"]["baseline_directory"]) / "videos" / f"{sid}.mp4")
        (root / "README.md").write_text("# DiCache VBench200 target\n\n200 candidates; runs contains new outputs and validated symlinks to unchanged prior candidates. Historical gpu suffix identifies the target/evaluation GPU, not generation placement. plan.json locks corrected settings. Baseline is reused read-only.\n")
    contexts = target_context(output)
    imported = []
    for name, context in contexts.items():
        if context["condition"]["threshold"] == .172:
            continue
        for path in sorted((previous / name / "runs/dicache_000").glob("*/manifest.json")):
            validate_any_gpu(path, context)
            link = output / name / "runs/dicache_000" / path.parent.name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(path.parent.resolve(), target_is_directory=True)
            imported.append(dict(target=name, sample_id=path.parent.name, manifest=artifact(path)))
    assignment = build_assignment(output, contexts)
    expected = {(name, p["sample_id"]) for name, ctx in contexts.items() for p in ctx["plan"]["prompts"]}
    done = {(r["target"], r["sample_id"]) for r in imported}
    remaining = [(j["target"], j["sample_id"]) for jobs in assignment["assignments"].values() for j in jobs]
    assert len(remaining) == len(set(remaining))
    assert not done.intersection(remaining) and done.union(remaining) == expected
    assert len(expected) == 600 and assignment["completed_before_rebalance"]["target_2p4x_gpu2"] == 0
    write_json(output / "import_validation.json", dict(status="pass", previous_root=str(previous),
        imported_count=len(imported), imported=imported, excluded_threshold=.072,
        corrected_threshold=.172, total_candidates=600, baseline_generated=0,
        coverage_exactly_once=True))
    write_json(output / "balanced_assignment.json", assignment)
    write_json(output / "launch.json", dict(targets=list(TARGETS), plans=plans,
        previous_root=str(previous), candidate_count=600, baseline_generated=0))
    (output / "README.md").write_text("# Corrected and balanced DiCache VBench200\n\nthresholds .075/.172/.396, retention .2, probe1. All frozen inference settings unchanged. GPU1/2/3 share remaining work by longest-processing-time assignment. import_validation.json records unchanged candidates imported from the superseded suite; all .072 candidates excluded. balanced_assignment.json records unique work ownership and source hashes. Three workers load once with staggered initialization and one full warmup each, then automatically aggregate and evaluate PSNR/SSIM/LPIPS and baseline/candidate 16-dimension VBench. Old target directory gpu suffixes only identify evaluation placement.\n")
    print({"output": str(output), "imported": len(imported), "remaining": len(remaining),
           "loads": assignment["estimated_load_seconds"]})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    prepare(a.previous.resolve(strict=True), a.output_dir.resolve())
