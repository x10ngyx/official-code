#!/usr/bin/env python3
"""Revalidate 200 shared baselines and compare all 44 DiCache scan baselines."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import json
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "runtime"))
from common import artifact, check_environment, external_output, index_result, read_json, sha256, write_json
from prompts import load_prompts


def audit(shared, scan_root, prompts):
    source = read_json(shared)
    expected = {x["sample_id"]: x for x in prompts}
    rows = {x["sample_id"]: x for x in source["rows"]}
    if len(rows) != 200 or len(source["rows"]) != 200 or set(rows) != set(expected):
        raise ValueError("shared baseline must contain all 200 unique prompts")
    protocol = read_json(PROJECT / "configs/wan22_t2v_a14b_50step_dpmpp.json")
    if any(protocol.get(k) != v for k, v in source["protocol"].items()):
        raise ValueError("shared protocol differs from frozen DiCache protocol")
    for sid, row in rows.items():
        if row["prompt_en"] != expected[sid]["prompt_en"] or row["seed"] != 42:
            raise ValueError("shared prompt/seed mismatch")
        for key in ("video", "timing"):
            record = row[key]
            if sha256(record["path"]) != record["sha256"] or Path(record["path"]).stat().st_size != record["bytes"]:
                raise ValueError("shared baseline artifact mismatch")
    configs = []
    for f in sorted(Path(source["baseline_directory"]).glob("generation_config.shard_*.json")):
        config = read_json(f)
        if config["condition"] != "baseline" or config["threshold"] is not None:
            raise ValueError("shared baseline enables caching")
        old = config["protocol"]
        mapping = dict(size="832*480", frame_num=45, fps=16, sample_steps=50,
                       sample_solver="dpm++", sample_shift=12., guide_scale_low_high=[3., 4.],
                       boundary=.875, seed=42, dtype="bfloat16", offload_model=True, t5_cpu=False)
        if any(old.get(k) != v for k, v in mapping.items()):
            raise ValueError("shared generation configuration mismatch")
        if Path(config["checkpoint_dir"]).resolve() != Path(source["checkpoint"]).resolve():
            raise ValueError("shared checkpoint mismatch")
        configs.append(artifact(f))
    if len(configs) != 4:
        raise ValueError("expected four baseline shard configurations")
    comparisons = []
    for f in sorted(scan_root.glob("round_*/runs/baseline/*/manifest.json")):
        m = read_json(f)
        if m["mode"] != "baseline" or m["status"] != "complete" or m["protocol"] != protocol:
            raise ValueError("scan baseline protocol/status mismatch")
        if Path(m["checkpoint"]).resolve() != Path(source["checkpoint"]).resolve():
            raise ValueError("scan/shared checkpoint mismatch")
        ref = rows[m["sample_id"]]
        if m["prompt"] != ref["prompt_en"]:
            raise ValueError("scan/shared prompt mismatch")
        for k in ("run", "video", "timing"):
            if sha256(m[k]["path"]) != m[k]["sha256"]:
                raise ValueError("scan baseline artifact changed")
        if m["video"]["sha256"] != ref["video"]["sha256"]:
            raise ValueError(f"scan/shared video bytes differ: {f}")
        comparisons.append(dict(sample_id=m["sample_id"], round=f.relative_to(scan_root).parts[0],
                                manifest=artifact(f), video=m["video"], shared_video=ref["video"], equal=True))
    if len(comparisons) != 44 or len({x["sample_id"] for x in comparisons}) != 11:
        raise ValueError("expected all 44 baselines from four rounds and 11 prompts")
    return dict(source, dicache_scan_comparison=dict(compared=44, byte_identical=44, unique_prompts=11,
                records=comparisons), source_audit=artifact(shared), shared_generation_configs=configs,
                integration="DiCache candidate-only runner validates this audit before reuse; no baseline generation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-audit", type=Path, default=Path("/all/yiran07-disk3/huteng_data/exp/magcache4wan22_baseline_reuse_audit_20260906/baseline_reuse_manifest.json"))
    parser.add_argument("--scan-root", type=Path, default=Path("/all/yiran07-disk3/huteng_data/exp/dicache4wan22_r02_threshold_scan_gpu123_20260907_194213"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    check_environment()
    result = audit(args.shared_audit, args.scan_root, load_prompts(PROJECT.parent / "Vbench200/prompts.jsonl", "standard"))
    out = external_output(args.output_dir)
    out.mkdir(exist_ok=False)
    index_result(out)
    write_json(out / "baseline_reuse_manifest.json", result)
    write_json(out / "VALIDATION.json", dict(status="pass", shared_videos=200, shared_raw_timings=200,
                                            scan_baselines=44, byte_identical=44, distinct_prompts=11,
                                            protocol_match=True, baseline_regeneration_required=False))
    (out / "README.md").write_text("# DiCache shared baseline reuse audit\n\nbaseline_reuse_manifest.json preserves the shared 200-video index, committed timing correction, and adds all 44 scan-to-shared video equality records. VALIDATION.json summarizes the checks. Shared source bytes are read-only; matching video bytes does not imply equal timing.\n")
    print(json.dumps(dict(status="pass", output_dir=str(out), shared=200, scan_equal=44)))


if __name__ == "__main__":
    main()
