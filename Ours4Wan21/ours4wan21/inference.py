"""Fixed resident Wan21 inference with one native warmup and persistent pipeline."""
import argparse
import json
from pathlib import Path
import sys
import time

from .contracts import (MODEL_ROOT, MODES, OFFICIAL, PROJECT, PROTOCOL,
                        create_result, dump, sha256, state_names)
from .policy import Policy, resolve_budget
from .runtime import apply_policy
from .overhead import predictor_fields, aggregate
from .shared import benchmark


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wan21-root', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, default=MODEL_ROOT / 'Wan2.1-T2V-1.3B')
    p.add_argument('--policy-checkpoint', type=Path)
    p.add_argument('--state-mode', choices=MODES)
    p.add_argument('--baseline', action='store_true')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--skip-budget', type=int)
    group.add_argument('--target-speedup', type=float)
    p.add_argument('--calibration', type=Path)
    prompts = p.add_mutually_exclusive_group(required=True)
    prompts.add_argument('--prompt')
    prompts.add_argument('--prompts', type=Path, help='JSONL: sample_id and prompt_en or prompt')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--flops-profile', type=Path, required=True,
                   help='81-frame Wan21 ComponentMetrics/Calflops profile including T5/VAE')
    return p.parse_args()


def main():
    args = parse_args()
    if Path(sys.prefix).name != 'wan2.2':
        raise ValueError('use conda environment wan2.2')
    if args.state_mode is not None:
        state_names(args.state_mode)
    if args.baseline:
        if any(v is not None for v in (args.policy_checkpoint, args.skip_budget, args.target_speedup, args.calibration)):
            raise ValueError('baseline must not have a policy or budget')
        budget = policy = None
    else:
        if args.policy_checkpoint is None:
            raise ValueError('--policy-checkpoint is required for candidate inference')
        budget = resolve_budget(skip_budget=args.skip_budget, target_speedup=args.target_speedup,
                                calibration=args.calibration)
    # Read-only use of the shared Wan21 source lock, residency and counters.
    benchmark()
    from protocol import checkpoint, prepare_resident, source_lock
    from generation import load_prompts
    from metrics import flops_for_calls
    from reporting import extract_component_latency, extract_component_tflops
    sys.path.insert(0, str(OFFICIAL / 'SeaCache4Wan21'))
    from inference_timing import _PipelineProfiler
    source_lock(args.wan21_root)
    model_dir = checkpoint(args.checkpoint_dir)
    rows = load_prompts(args)
    profile = json.loads(args.flops_profile.read_text())
    if profile['input']['video_shape_fhw'] != [81, 480, 832]:
        raise ValueError('FLOPs profile must be Wan21 81 frames / 832x480')
    components = extract_component_tflops(profile)
    if profile['input']['transformer_blocks'] != 30:
        raise ValueError('FLOPs profile must describe the 30-block Wan21-1.3B DiT')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('select exactly one GPU with CUDA_VISIBLE_DEVICES')
    if torch.cuda.get_device_properties(0).total_memory < 44 * 1024**3:
        raise ValueError('resident Wan21 inference requires a 48GB-class GPU')
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    if not args.baseline:
        policy = Policy(args.policy_checkpoint, device='cuda:0', state_mode=args.state_mode)
    out = create_result(args.output_dir, '# Ours4Wan21 inference\n\nrun.json freezes protocol, policy and sources. videos/, traces/, timings/ contain measured samples; components.json contains full generate and T5/DiT/VAE timings/TFLOPs. Native warmup is excluded. COMPLETE.json means generation complete; quality is evaluated separately by evaluate.py.')
    sys.path.insert(0, str(args.wan21_root.resolve()))
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import cache_video
    generation = dict(size=(832, 480), frame_num=81, shift=5., sample_solver='unipc',
                      sampling_steps=50, guide_scale=5., seed=42, offload_model=False)
    try:
        started = time.perf_counter()
        pipe = wan.WanT2V(config=WAN_CONFIGS['t2v-1.3B'], checkpoint_dir=str(model_dir),
            device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False)
        prepare_resident(pipe)
        if pipe.param_dtype != torch.bfloat16 or pipe.t5_cpu:
            raise ValueError('pipeline violates BF16/T5-GPU protocol')
        torch.cuda.synchronize()
        init_seconds = time.perf_counter() - started
        with torch.no_grad():
            warmup = pipe.generate(rows[0]['prompt'], **generation)
        del warmup
        torch.cuda.synchronize()
        controller = apply_policy(pipe, policy, budget) if policy is not None else None
        for folder in ('videos', 'traces', 'timings'):
            (out / folder).mkdir()
        method = 'baseline' if args.baseline else 'ours'
        source_paths = [*sorted((PROJECT / 'ours4wan21').glob('*.py')),
                        OFFICIAL / 'SeaCache4Wan21/seacache.py',
                        OFFICIAL / 'SeaCache4Wan21/wan21_integration.py']
        dump(out / 'run.json', dict(protocol=PROTOCOL, method=method, prompts=rows,
            checkpoint_dir=str(model_dir), gpu=torch.cuda.get_device_name(0),
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
            warmup='one native full generation', pipeline_init_seconds=init_seconds,
            policy_checkpoint=str(args.policy_checkpoint) if policy else None,
            policy_sha256=sha256(args.policy_checkpoint) if policy else None,
            state_mode=policy.mode if policy else None, skip_budget=budget,
            target_speedup=args.target_speedup,
            calibration_sha256=sha256(args.calibration) if args.calibration else None,
            flops_profile_sha256=sha256(args.flops_profile),
            predictor_flops_profile=policy.flops_profile if policy else None,
            predictor_warmup='one FP32 network forward before measured inference' if policy else None,
            source_hashes={str(p): sha256(p) for p in source_paths}))
        measurements = []
        for row in rows:
            sid = row['sample_id']
            profiler = _PipelineProfiler(pipe, init_wall_seconds=init_seconds,
                output_path=out / 'timings' / f'{sid}.json',
                implementation='wan21' if args.baseline else 'ours')
            profiler.install()
            with torch.no_grad():
                video = pipe.generate(row['prompt'], **generation)
            if controller:
                trace = controller.summary()
            else:
                trace = dict(schema='ours4wan21_baseline_trace_v1',
                             total_steps=50, step_reuse=0, step_recompute=50)
            dump(out / 'traces' / f'{sid}.json', trace)
            timing = json.loads((out / 'timings' / f'{sid}.json').read_text())
            if timing['status'] != 'success':
                raise RuntimeError('failed measured generation')
            overhead = {}
            if policy:
                timing['predictor'] = policy.overhead_summary()
                timing['latent_feature'] = controller.feature_overhead()
                overhead = predictor_fields(timing, trace)
                overhead['latent_feature_wall_seconds'] = timing['latent_feature']['wall_seconds']
                dump(out / 'timings' / f'{sid}.json', timing)
            tflops = flops_for_calls(timing['calls'], profile, method)
            measurements.append(dict(sample_id=sid,
                generate_seconds=timing['pipeline_generate_wall_seconds'],
                dit_tflops=tflops, **extract_component_latency(timing), **components, **overhead))
            cache_video(tensor=video[None], save_file=str(out / 'videos' / f'{sid}.mp4'),
                        fps=16, nrow=1, normalize=True, value_range=(-1, 1))
            del video
        dump(out / 'components.json', dict(rows=measurements,
            predictor_overhead=aggregate(measurements) if policy else None,
            latent_feature_wall_seconds_total=sum(r.get("latent_feature_wall_seconds", 0.) for r in measurements),
            latency_scope='complete pipeline.generate; excludes model load, native warmup, MP4/trace writing',
            tflops_scope='DiT Calflops full/always-on count by executed blocks; T5/VAE separately profiled; predictor MLP separately counted per actual query. SEA FFT, residual add and scheduler excluded. Predictor time is nested in DiT/generate; never add it again.'))
        dump(out / 'COMPLETE.json', dict(status='generation_complete', videos=len(rows)))
    except BaseException as exc:
        dump(out / 'FAILED.json', dict(error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
