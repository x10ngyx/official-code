#!/usr/bin/env python3
"""Bounded adaptive threshold calibration; every selected speed is measured."""
from __future__ import annotations
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import csv
import fcntl
import json
import math
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "runtime"))
import suite
from batch import atomic_json
from common import artifact, check_environment, external_output, index_result, read_json
from scan import select_targets


def validate_presets(config):
    if config["retention_ratio"] != .2 or config["probe_depth"] != 1:
        raise ValueError("target scan fixes official retention=.2 and probe=1")
    select_targets({"check": dict(speedup=1., method={})},
                   config["target_speedups"], config["relative_tolerance"])
    grid = config["coarse_thresholds"]
    if (not grid or len(set(grid)) != len(grid)
            or any(not math.isfinite(x) or x <= 0 for x in grid)
            or not isinstance(config["max_rounds"], int) or config["max_rounds"] < 1
            or not math.isfinite(config["max_threshold"])
            or max(grid) > config["max_threshold"]
            or not math.isfinite(config["min_threshold_gap"])
            or config["min_threshold_gap"] <= 0):
        raise ValueError("invalid bounded scan grid")
    if "focus_speedup_range" in config:
        bounds = config["focus_speedup_range"]
        if (len(bounds) != 2 or any(not math.isfinite(x) for x in bounds)
                or not 1 < bounds[0] < bounds[1]
                or any(not bounds[0] <= x <= bounds[1] for x in config["target_speedups"])):
            raise ValueError("invalid measured speed focus range")


def selection(measured, config):
    targets = []
    for target in config["target_speedups"]:
        result = select_targets(measured, [target], target * config["relative_tolerance"])
        targets.extend(result["targets"])
    return dict(metric="ratio_of_summed_complete_generate_wall_seconds", targets=targets,
                scope="Calibration subset only; independent same-GPU baseline in each round.")


def refine(measured, config):
    """Midpoints are search proposals, never predicted gate paths or speedups.

    Inspect every observed crossing, including nonmonotone ones, and both
    neighbors of the closest point. Boundaries expand geometrically.
    """
    ordered = sorted(measured.values(), key=lambda x: x["method"]["threshold"])
    seen = [x["method"]["threshold"] for x in ordered]
    focus = config.get("focus_speedup_range")
    proposed = set()
    for row in selection(measured, config)["targets"]:
        if row["status"] == "matched":
            continue
        target = row["target_speedup"]
        nearest = min(range(len(ordered)), key=lambda i: abs(ordered[i]["speedup"] - target))
        intervals = {i for i in (nearest - 1, nearest) if 0 <= i < len(ordered) - 1}
        intervals.update(i for i in range(len(ordered) - 1)
                         if (ordered[i]["speedup"] - target) * (ordered[i + 1]["speedup"] - target) <= 0)
        if focus:
            intervals = {i for i in intervals
                         if max(ordered[i]["speedup"], ordered[i + 1]["speedup"]) >= focus[0]
                         and min(ordered[i]["speedup"], ordered[i + 1]["speedup"]) <= focus[1]}
        proposed.update((seen[i] + seen[i + 1]) / 2 for i in intervals)
        if nearest == 0 and (not focus or ordered[0]["speedup"] > target
                             and ordered[0]["speedup"] > focus[0]):
            proposed.add(seen[0] / 2)
        if nearest == len(ordered) - 1 and (not focus or ordered[-1]["speedup"] < target
                                          and ordered[-1]["speedup"] < focus[1]):
            proposed.add(min(config["max_threshold"], seen[-1] * 2))
    gap = config["min_threshold_gap"]
    result = []
    for value in sorted(round(x, 8) for x in proposed):
        if 0 < value <= config["max_threshold"] and all(abs(value - x) >= gap - 1e-12 for x in seen + result):
            result.append(value)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=PROJECT / "build/Wan2.2-42bf4cf")
    p.add_argument("--checkpoint", type=Path, default=PROJECT.parents[2] / "models/Wan2.2-T2V-A14B")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu-ids", nargs="+", required=True)
    p.add_argument("--presets", type=Path, default=Path(__file__).with_name("presets.json"))
    p.add_argument("--prompts", type=Path, default=PROJECT / "experiments/parameter_scan/prompts.jsonl")
    p.add_argument("--calflops-profile", type=Path)
    p.add_argument("--worker-launch-wave-size", type=int, default=1)
    p.add_argument("--worker-ready-timeout", type=float, default=1800)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--defer-evaluation", action="store_true")
    return p


def round_args(args, config, root, thresholds, profile=None):
    argv = ["--source", str(args.source), "--checkpoint", str(args.checkpoint),
            "--output-dir", str(root), "--prompts", str(args.prompts),
            "--gpu-ids", *args.gpu_ids, "--worker-launch-wave-size", str(args.worker_launch_wave_size),
            "--worker-ready-timeout", str(args.worker_ready_timeout),
            "--thresholds", *map(str, thresholds), "--retention-ratios", ".2",
            "--target-speedups", *map(str, config["target_speedups"]), "--defer-evaluation"]
    if profile:
        argv += ["--calflops-profile", str(profile)]
    if args.resume and root.exists():
        argv += ["--resume"]
    return argv


def write_readout(root, measured, chosen):
    fields = ["condition_id", "threshold", "retention_ratio", "speedup", "round", "local_condition"]
    with (root / "scan.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for key, value in measured.items():
            writer.writerow(dict(condition_id=key, threshold=value["method"]["threshold"],
                                 retention_ratio=.2, speedup=value["speedup"],
                                 round=value["round"], local_condition=value["local_condition"]))
    lines = ["# DiCache 三档阈值标定", "", "按配对完整 generate 时间的总和比值选择；仅代表这组11条标定prompt。",
             "", "|目标|实测|阈值|retention|状态|", "|---:|---:|---:|---:|---|"]
    for row in chosen["targets"]:
        lines.append(f'|{row["target_speedup"]:.1f}×|{row["speedup"]:.6f}×|{row["method"]["threshold"]}|.2|{row["status"]}|')
    lines += ["", "quality.json 保存所选候选的 PSNR/SSIM/LPIPS 和 VBench；不存在时质量评测尚未完成。",
              "每轮重新测同卡baseline；每个复用调用的DiT FLOPs包含一个probe block。",
              "unmatched 是最接近的已测候选，不能当作已达标配置。"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def execute(args, config, root):
    measured = {}
    thresholds = sorted(config["coarse_thresholds"])
    profile = args.calflops_profile
    rounds = []
    for number in range(config["max_rounds"]):
        if not thresholds:
            break
        directory = root / f"round_{number:02d}"
        atomic_json(root / "status.json", dict(status="running", phase="generation", round=number, thresholds=thresholds))
        # suite revalidates all existing manifests, hashes and GPU pairing on resume.
        suite.main(scan=True, argv=round_args(args, config, directory, thresholds, profile))
        if profile is None:
            profile = Path(read_json(directory / "profile_artifact.json")["path"])
        plan = read_json(directory / "plan.json")
        rounds.append((directory, plan))
        for condition, value in read_json(directory / "performance.json").items():
            measured[f"round_{number:02d}/{condition}"] = dict(value, round=number, local_condition=condition)
        chosen = selection(measured, config)
        atomic_json(root / "performance.json", measured)
        atomic_json(root / "target_selection.json", chosen)
        write_readout(root, measured, chosen)
        thresholds = refine(measured, config)
    atomic_json(root / "SCAN_COMPLETE.json", dict(status="complete", selection=artifact(root / "target_selection.json"),
                all_targets_matched=all(x["status"] == "matched" for x in chosen["targets"])))
    if args.defer_evaluation:
        atomic_json(root / "status.json", dict(status="pending", phase="selected_quality"))
        return
    quality = {}
    atomic_json(root / "status.json", dict(status="running", phase="selected_quality"))
    for number, (directory, plan) in enumerate(rounds):
        ids = {measured[x["condition_id"]]["local_condition"] for x in chosen["targets"]
               if measured[x["condition_id"]]["round"] == number}
        if ids:
            subset = dict(plan, conditions=[c for c in plan["conditions"] if c["id"] == "baseline" or c["id"] in ids])
            quality[f"round_{number:02d}"] = suite.evaluate(directory, subset)
    atomic_json(root / "quality.json", quality)
    atomic_json(root / "COMPLETE.json", dict(status="complete", selection=artifact(root / "target_selection.json"),
                quality=artifact(root / "quality.json"), all_targets_matched=all(x["status"] == "matched" for x in chosen["targets"])))
    atomic_json(root / "status.json", dict(status="complete", phase="complete"))


def main(argv=None):
    args = parser().parse_args(argv)
    check_environment()
    config = read_json(args.presets)
    validate_presets(config)
    root = external_output(args.output_dir)
    initial = suite.build_plan(suite.parser(True).parse_args(round_args(
        args, config, root / "round_00", sorted(config["coarse_thresholds"]), args.calflops_profile)), True)
    plan = dict(schema="dicache4wan22_adaptive_scan_v1", config=config,
                presets=artifact(args.presets), initial_round=initial,
                max_rounds=config["max_rounds"], subsequent_rounds="adaptive; new same-GPU baselines each round",
                selection_tolerance="relative per target", quality="selected candidates and their paired baselines only")
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    if root.exists() and not args.resume:
        raise FileExistsError("existing scan requires --resume")
    root.mkdir(parents=True, exist_ok=True)
    index_result(root)
    with (root / ".adaptive.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "plan.json").exists() and read_json(root / "plan.json") != plan:
            raise ValueError("scan resume plan mismatch")
        atomic_json(root / "plan.json", plan)
        (root / "README.md").write_text("# DiCache adaptive threshold scan\n\nplan.json locks inputs and search limits. round_*/ stores same-GPU paired generation, component profiles and traces. scan.csv / REPORT.md / target_selection.json report measured candidates. quality.json contains selected-candidate quality. SCAN_COMPLETE means search exhausted or matched; COMPLETE additionally requires selected quality. Neither marker implies every target matched: inspect all_targets_matched.\n")
        try:
            execute(args, config, root)
        except BaseException as exc:
            atomic_json(root / "status.json", dict(status="failed", error=repr(exc)))
            raise
    print(json.dumps(dict(output_dir=str(root), **read_json(root / "status.json"))))


if __name__ == "__main__":
    main()
