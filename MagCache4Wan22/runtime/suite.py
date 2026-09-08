"""Shared persistent batch orchestration and resumable evaluation for grid/single runs."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from common import (PROJECT, REPOSITORY, THREAD_VARS, artifact, check_environment,
                    checkpoint_path, external_output, index_result, read_json, sha256,
                    validate_source, write_json)
from batch import (atomic_json, job_directory, job_plan, launch_workers, validate_completed)
from prompts import load_prompts
from scan import grid_conditions, preset_conditions, select_targets, select_preset_targets
from aggregate_performance import aggregate, validate_profile


def parser(scan=False):
    p = argparse.ArgumentParser(description="Persistent MagCache parameter scan" if scan else "Persistent MagCache paired batch")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    default_prompts = (PROJECT / "experiments/parameter_scan/prompts.jsonl" if scan
                       else REPOSITORY / "Vbench200/prompts.jsonl")
    p.add_argument("--prompts", type=Path, default=default_prompts)
    p.add_argument("--vbench-mode", choices=("standard", "custom"), default="standard")
    p.add_argument("--gpu-ids", nargs="+", default=["0"])
    p.add_argument("--worker-launch-wave-size", type=int, default=1)
    p.add_argument("--worker-ready-timeout", type=float, default=1800)
    p.add_argument("--warmup-videos", type=int, default=1)
    p.add_argument("--calflops-profile", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--defer-evaluation", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    if scan:
        p.add_argument("--thresholds", nargs="+", type=float)
        p.add_argument("--ks", nargs="+", type=int)
        p.add_argument("--retention-ratios", nargs="+", type=float)
        p.add_argument("--target-presets", type=Path, help="JSON with a fixed R/K and E grid for each target")
        p.add_argument("--target-speedups", nargs="+", type=float)
        p.add_argument("--target-tolerance", type=float, help="absolute speedup tolerance; overrides preset relative tolerance")
        p.add_argument("--no-deduplicate", action="store_true")
    else:
        p.add_argument("--threshold", type=float, default=.06)
        p.add_argument("--magcache-k", type=int, default=2)
        p.add_argument("--retention-ratio", type=float, default=.4)
    return p


def build_plan(args, scan):
    if (not args.gpu_ids or len(set(args.gpu_ids)) != len(args.gpu_ids)
            or any(not str(g).isdigit() for g in args.gpu_ids)):
        raise ValueError("GPU IDs must be distinct physical indices")
    if not 1 <= args.worker_launch_wave_size <= len(args.gpu_ids):
        raise ValueError("invalid worker launch wave size")
    if args.worker_ready_timeout <= 0 or args.warmup_videos < 0:
        raise ValueError("invalid warmup or ready timeout")
    source = validate_source(args.source)
    checkpoint = checkpoint_path(args.checkpoint)
    rows = load_prompts(args.prompts, args.vbench_mode)
    if len(rows) < len(args.gpu_ids):
        raise ValueError("more workers than prompts")
    groups = []
    if scan and args.target_presets:
        if any(v is not None for v in (args.thresholds, args.ks, args.retention_ratios)):
            raise ValueError("target presets cannot be combined with a manual R/K/E grid")
        conditions, groups = preset_conditions(read_json(args.target_presets), args.target_speedups,
                                              args.target_tolerance, not args.no_deduplicate)
    elif scan:
        conditions = grid_conditions(args.thresholds or [.04, .06, .1, .2, .4, .8],
                                     args.ks or [2, 3, 4, 6], args.retention_ratios or [.4, .2, .1, .05],
                                     not args.no_deduplicate)
    else:
        conditions = grid_conditions([args.threshold], [args.magcache_k], [args.retention_ratio])
    baseline = dict(id="baseline", mode="baseline", threshold=.06, K=2, retention_ratio=.4)
    profile = artifact(args.calflops_profile) if args.calflops_profile else None
    targets = (args.target_speedups or [1.8, 2.4, 3.0]) if scan else []
    tolerance = args.target_tolerance if scan and args.target_tolerance is not None else .1
    if groups:
        targets, tolerance = [g["target"] for g in groups], None
    elif scan:
        select_targets({"validation_only": dict(speedup=1., method={})}, targets, tolerance)
    plan = dict(schema="magcache4wan22_batch_plan_v1", kind="parameter_scan" if scan else "paired_batch",
                source=source, checkpoint=str(checkpoint), prompts=rows,
                prompt_source=artifact(args.prompts), gpu_ids=args.gpu_ids,
                protocol=read_json(PROJECT / "configs/wan22_t2v_a14b_50step_dpmpp.json"),
                conditions=[baseline] + conditions, profile_input=profile,
                warmup_videos=args.warmup_videos, vbench_mode=args.vbench_mode,
                target_speedups=targets, target_tolerance=tolerance,
                generated_videos=len(rows) * (1 + len(conditions)),
                unreported_warmup_videos=len(args.gpu_ids) * args.warmup_videos,
                runner="one persistent pipeline per worker; round-robin prompts; same-GPU paired conditions")
    if groups:
        plan.update(target_groups=groups, target_presets=artifact(args.target_presets))
    return plan


def link_video(link, target):
    link.parent.mkdir(parents=True, exist_ok=True)
    target = Path(target).resolve()
    if link.is_symlink() and link.resolve() == target:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(link)
    link.symlink_to(target)


def run_logged(root, command, name, gpu):
    env = dict(os.environ, PYTHON_BIN=sys.executable, CUDA_VISIBLE_DEVICES=gpu,
               **{k: "1" for k in THREAD_VARS})
    with (root / (name + ".log")).open("a") as log:
        subprocess.run([str(x) for x in command], env=env, check=True,
                       stdout=log, stderr=subprocess.STDOUT)


def ensure_profile(root, plan):
    marker = root / "profile_artifact.json"
    if marker.exists():
        record = read_json(marker)
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("profile SHA mismatch")
    elif plan["profile_input"]:
        record = plan["profile_input"]
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("provided profile SHA mismatch")
    else:
        path = root / "profile/calflops.json"
        if not path.exists():
            run_logged(root, [sys.executable, PROJECT / "experiments/performance_t2v_a14b/profile_calflops.py",
                             "--wan22-root", plan["source"]["source"], "--checkpoint-dir", plan["checkpoint"],
                             "--output", path], "profile", plan["gpu_ids"][0])
        record = artifact(path)
    profile = read_json(record["path"])
    validate_profile(profile)
    if (profile["source"]["prepared_manifest"] != plan["source"]
            or Path(profile["source"]["checkpoint_dir"]).resolve() != Path(plan["checkpoint"])):
        raise ValueError("profile does not match this source/checkpoint")
    if not marker.exists():
        write_json(marker, record)
    return profile


def measured_performance(root, plan, profile):
    manifests = {}
    for condition in plan["conditions"]:
        paths = []
        for index, row in enumerate(plan["prompts"]):
            path = job_directory(root, row, condition) / "manifest.json"
            manifest = validate_completed(path, job_plan(plan, row, condition), profile,
                                          plan["gpu_ids"][index % len(plan["gpu_ids"])], plan["warmup_videos"])
            if condition["mode"] == "magcache":
                actions = [r["action"] for r in read_json(manifest["trace"]["path"])["calls"]]
                if actions != condition["schedule"]:
                    raise ValueError("completed trace differs from planned official schedule")
            paths.append(path)
            link_video(root / "videos" / condition["id"] / (row["sample_id"] + ".mp4"), manifest["video"]["path"])
        manifests[condition["id"]] = paths
    performance = {}
    for condition in plan["conditions"][1:]:
        name = condition["id"]
        value = aggregate(manifests["baseline"], manifests[name], profile)
        value["profile"] = read_json(root / "profile_artifact.json")
        value["equivalent_parameters"] = condition["equivalent_parameters"]
        atomic_json(root / "performance" / (name + ".json"), value)
        performance[name] = value
    atomic_json(root / "performance.json", performance)
    if plan["target_speedups"]:
        selection = (select_preset_targets(performance, plan["target_groups"]) if plan.get("target_groups")
                     else select_targets(performance, plan["target_speedups"], plan["target_tolerance"]))
        atomic_json(root / "target_selection.json", selection)
    return performance


def evaluation_stage(root, name, command_builder, result_filename, gpu):
    marker = root / "evaluation" / (name + ".json")
    if marker.exists():
        record = read_json(marker)
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"evaluation artifact changed: {name}")
        return read_json(record["path"])
    stage_root = root / "evaluation" / name
    stage_root.mkdir(parents=True, exist_ok=True)
    # Fresh attempt directories let resume recover interrupted external evaluators.
    attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=stage_root))
    (attempt / "README.md").write_text("# Quality evaluation attempt\n\noutputs/ holds evaluator results; the parent completion record commits the final score SHA.\n")
    output = attempt / "outputs"
    run_logged(root, command_builder(output), "evaluation_" + name, gpu)
    record = artifact(output / result_filename)
    value = read_json(record["path"])
    write_json(marker, record)
    return value


def evaluate(root, plan):
    quality, vbench, scores = {}, {}, {}
    mapping = root / "prompt_map.json"
    if not mapping.exists():
        write_json(mapping, {r["sample_id"] + ".mp4": r["prompt_en"] for r in plan["prompts"]})
    gpu = plan["gpu_ids"][0]
    for condition in plan["conditions"]:
        name = condition["id"]
        if name != "baseline":
            quality[name] = evaluation_stage(root, "video_metrics_" + name, lambda out: [
                sys.executable, REPOSITORY / "VideoMetrics/evaluate.py", "--reference-dir", root / "videos/baseline",
                "--candidate-dir", root / "videos" / name, "--expected-frames", 45,
                "--metrics", "psnr", "ssim", "lpips", "--device", "cuda:0", "--output-dir", out], "summary.json", gpu)
        if plan["vbench_mode"] == "standard":
            builder = lambda out: ["bash", REPOSITORY / "VbenchEvaluation/run_vbench200.sh",
                                   root / "videos" / name, out, "1", "--allow-missing"]
            filename = "vbench200_aggregate_scores.json"
        else:
            builder = lambda out: ["bash", REPOSITORY / "VbenchEvaluation/run_custom_vbench.sh",
                                   root / "videos" / name, out, mapping]
            filename = "vbench_custom_aggregate_scores.json"
        value = evaluation_stage(root, "vbench_" + name, builder, filename, gpu)
        vbench[name] = value
        scores[name] = (value["aggregate_scores"]["total_score"] if plan["vbench_mode"] == "standard"
                        else value["vbench_score"])
    return dict(video_metrics=quality, vbench=vbench, vbench_score=scores,
                vbench_protocol=("vbench200_16_dimensions" if plan["vbench_mode"] == "standard"
                                 else "vbench_custom_input_raw_mean_v1"))


def main(scan=False, argv=None):
    args = parser(scan).parse_args(argv)
    check_environment()
    plan = build_plan(args, scan)
    root = external_output(args.output_dir)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    if root.exists() and not args.resume:
        raise FileExistsError(f"use --resume for an existing suite: {root}")
    root.mkdir(parents=True, exist_ok=True)
    index_result(root)
    with (root / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another controller owns this output directory") from None
        if (root / "plan.json").exists():
            if read_json(root / "plan.json") != plan:
                raise ValueError("resume plan mismatch (source, prompts, parameters, GPUs, warmup or profile)")
        else:
            write_json(root / "plan.json", plan)
        (root / "README.md").write_text(
            "# MagCache persistent batch / scan\n\nplan.json locks the full protocol, prompts and parameters. "
            "runs/<condition>/<sample>/ holds measured video, timing, trace and SHA manifests; videos/ indexes MP4s. "
            "worker_status/ records persistent workers; profile/ stores real-shape Calflops. performance/ contains "
            "validated paired component time/FLOPs; target_selection.json uses measured latency only. "
            "evaluation/ holds resumable VideoMetrics/VBench attempts. report.json and COMPLETE.json require all quality stages. "
            "incomplete_attempts/ preserves interrupted video attempts before regeneration.\n")
        try:
            profile = ensure_profile(root, plan)
            if not (root / "GENERATION_COMPLETE.json").exists():
                atomic_json(root / "status.json", dict(status="running", phase="generation"))
                launch_workers(root, plan, args.resume, args.worker_launch_wave_size, args.worker_ready_timeout)
            performance = measured_performance(root, plan, profile)
            atomic_json(root / "GENERATION_COMPLETE.json", dict(status="complete", performance=artifact(root / "performance.json")))
            if args.defer_evaluation and not (root / "COMPLETE.json").exists():
                atomic_json(root / "status.json", dict(status="pending", phase="quality_evaluation"))
                print(json.dumps(dict(status="quality_pending", output_dir=str(root))))
                return
            atomic_json(root / "status.json", dict(status="running", phase="quality_evaluation"))
            quality = evaluate(root, plan)
            report = dict(schema="magcache4wan22_batch_report_v1", status="complete", plan=plan,
                          performance=performance, **quality)
            if plan["target_speedups"]:
                report["target_selection"] = read_json(root / "target_selection.json")
            atomic_json(root / "report.json", report)
            atomic_json(root / "COMPLETE.json", dict(status="complete", report=artifact(root / "report.json")))
            atomic_json(root / "status.json", dict(status="complete", phase="complete"))
            print(json.dumps(dict(status="complete", output_dir=str(root))))
        except BaseException as exc:
            atomic_json(root / "status.json", dict(status="failed", error=repr(exc)))
            raise
