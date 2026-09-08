"""One native numerical generation path shared by single and persistent runners."""
from __future__ import annotations
import json
import time
from contextlib import ExitStack
from pathlib import Path
from common import artifact, index_result, read_json, validate_config, write_json


def generate_native(pipeline, prompt):
    # Same arguments as the other Wan22 methods; no prompt extension or RNG edits.
    return pipeline.generate(prompt, size=(832, 480), frame_num=45,
                             shift=12, sample_solver="dpm++", sampling_steps=50,
                             guide_scale=(3.0, 4.0), seed=42, offload_model=True)


def create_pipeline(checkpoint):
    import torch
    import wan
    from wan.configs import WAN_CONFIGS
    cfg = WAN_CONFIGS["t2v-A14B"]
    validate_config(cfg)
    started = time.perf_counter()
    pipeline = wan.WanT2V(config=cfg, checkpoint_dir=str(checkpoint), device_id=0,
                         rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False,
                         t5_cpu=False, convert_model_dtype=True)
    torch.cuda.synchronize()
    return pipeline, time.perf_counter() - started


def make_plan(provenance, checkpoint, row, mode, options, split_steps=32):
    from common import PROJECT
    return dict(schema="magcache4wan22_run_v1", mode=mode, prompt=row["prompt_en"],
                sample_id=row["sample_id"], checkpoint=str(checkpoint),
                protocol=read_json(PROJECT / "configs/wan22_t2v_a14b_50step_dpmpp.json"),
                method=dict(threshold=options.magcache_thresh, K=options.magcache_K,
                            retention_ratio=options.retention_ratio, split_steps=split_steps,
                            ratio_source=("measured_full_forward_calibration" if mode == "calibrate"
                                          else "official_builtin_40step_nearest_interp"),
                            variant="official_unmodified"), source=provenance)


def generate_run(pipeline, core, options, plan, destination, *, init_seconds=0., lifecycle=None):
    from official import official_forward, record_calls
    from inference_timing import _PipelineProfiler
    from wan.utils.utils import save_video
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    index_result(destination)
    (destination / "README.md").write_text(
        "# MagCache measured run\n\nrun.json locks inputs; videos/ holds MP4; timing.json contains "
        "complete generate and component times; trace.json records official CFG decisions. "
        "manifest.json commits successful outputs with SHA256.\n")
    write_json(destination / "run.json", plan)
    mode = plan["mode"]
    split_steps = plan["method"]["split_steps"]
    records = []
    try:
        models = [pipeline.high_noise_model, pipeline.low_noise_model]
        with ExitStack() as stack:
            if mode != "baseline":
                stack.enter_context(official_forward(
                    models, core=core, args=options, split_steps=split_steps,
                    calibration_dir=(destination / "calibration") if mode == "calibrate" else None))
                stack.enter_context(record_calls(models, records, calibration=mode == "calibrate"))
            profiler = _PipelineProfiler(pipeline, init_wall_seconds=init_seconds,
                                         output_path=destination / "timing.json",
                                         implementation=mode)
            profiler.install()
            video = generate_native(pipeline, plan["prompt"])
        timing = read_json(destination / "timing.json")
        if lifecycle is not None:
            timing["pipeline_lifecycle"] = lifecycle
            (destination / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        if timing["model_forward_call_count"] != 100:
            raise ValueError("fixed protocol must execute 100 CFG forwards")
        if mode != "baseline":
            if len(records) != 100:
                raise ValueError("incomplete official MagCache trace")
            for observed, measured in zip(records, timing["calls"]):
                for field in ("call_index", "step_index", "model_stage", "cfg_branch", "blocks_executed"):
                    if observed[field] != measured[field]:
                        raise ValueError(f"trace/timing mismatch for {field}")
            write_json(destination / "trace.json", dict(schema="magcache4wan22_trace_v1",
                       mode=mode, method=plan["method"], calls=records))
        videos = destination / "videos"
        videos.mkdir()
        video_path = videos / (plan["sample_id"] + ".mp4")
        save_video(tensor=video[None], save_file=str(video_path), fps=16,
                   nrow=1, normalize=True, value_range=(-1, 1))
        manifest = dict(**plan, status="complete", timing=artifact(destination / "timing.json"),
                        video=artifact(video_path), run=artifact(destination / "run.json"))
        if mode != "baseline":
            manifest["trace"] = artifact(destination / "trace.json")
        if mode == "calibrate":
            manifest["calibration"] = [artifact(p) for p in sorted((destination / "calibration").glob("*.json"))]
        write_json(destination / "manifest.json", manifest)
        return destination / "manifest.json"
    except BaseException as exc:
        write_json(destination / "FAILED.json", dict(status="failed", error=repr(exc)))
        raise

