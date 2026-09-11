"""Persistent fixed-protocol baseline/CNN generation with shared instrumentation."""
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
import time
import torch
from .contracts import PROTOCOL, budget
from .shared import OFFICIAL, MODELS, load, read, write, sha256, under, result_dir, verify_sources, implementation_hashes
from .metrics import load_profile, performance
from .policy import Policy, resolve_budget, DEFAULT_CALIBRATION
from .runtime import apply_policy


def validate_jobs(jobs):
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('nonempty job list required')
    ids = set()
    for job in jobs:
        sid = job.get('sample_id')
        if (not isinstance(sid,str) or not sid or Path(sid).name != sid
                or sid in ('.','..') or sid in ids or not isinstance(job.get('prompt'),str)
                or not job['prompt'].strip()):
            raise ValueError('unique safe sample_id and nonempty prompt required')
        ids.add(sid)
    return jobs


def generate(pipeline, controller=None):
    args = dict(size=(832,480), frame_num=45, shift=12., sample_solver='dpm++',
                sampling_steps=50, guide_scale=(3.,4.), seed=42, offload_model=True)
    if controller is not None:
        from wan.seacache import SeaCacheConfig
        args['seacache_config'] = SeaCacheConfig(threshold=1.)
    return args


def run(args):
    verify_sources()
    jobs = validate_jobs(read(args.jobs))
    source = args.wan22_root.resolve()
    subprocess.run([sys.executable, str(OFFICIAL/'SeaCache4Wan22/scripts/validate_prepared_tree.py'),
                    '--source', str(source), '--mode', 'prepared'], check=True, stdout=subprocess.DEVNULL)
    checkpoint = under(args.checkpoint_dir, MODELS)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    profile = load_profile(args.profile)
    if Path(profile['source']['checkpoint_dir']).resolve() != checkpoint:
        raise ValueError('Calflops model checkpoint mismatch')
    source_manifest = read(source/'.seacache4wan22_prepared.json')
    profile_source = profile['source']['prepared_manifest']
    for key in ('wan22_commit','patch_sha256','runtime_sha256','timing_runtime_sha256','protocol_sha256'):
        if profile_source.get(key) != source_manifest.get(key):
            raise ValueError('profile/source lock mismatch: ' + key)
    if args.policy:
        if args.target_speedup is not None and args.calibration is None:
            args.calibration = DEFAULT_CALIBRATION
        args.skip_budget = resolve_budget(args.skip_budget,args.target_speedup,args.calibration)
    elif any(x is not None for x in (args.skip_budget,args.target_speedup,args.calibration)):
        raise ValueError('budget options require a policy checkpoint')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('expose exactly one CUDA GPU')
    sys.path.insert(0,str(source))
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import save_video
    cfg = WAN_CONFIGS['t2v-A14B']
    if cfg.param_dtype != torch.bfloat16 or cfg.boundary != .875 or cfg.num_train_timesteps != 1000:
        raise ValueError('Wan22 config mismatch')
    output = result_dir(args.output, '# Ours4Wan22 inference\n\nrun.json freezes inputs; videos/, timings/, traces/, features/ preserve each trajectory; manifest.json is written only after all samples succeed.')
    plan = dict(schema='ours4wan22_run_v1', protocol=PROTOCOL, mode='ours' if args.policy else 'baseline',
                checkpoint_dir=str(checkpoint), source_manifest=source_manifest,
                source_lock=read(Path(__file__).resolve().parents[1]/'source_lock.json'),
                implementation_sha256=implementation_hashes(),
                profile_sha256=sha256(args.profile), jobs=jobs, skip_budget=args.skip_budget,
                target_speedup=args.target_speedup,
                calibration_sha256=sha256(args.calibration) if args.calibration else None,
                calibration=read(args.calibration) if args.calibration else None,
                calibration_path=str(args.calibration.resolve()) if args.calibration else None,
                policy_sha256=sha256(args.policy) if args.policy else None,
                gpu_name=torch.cuda.get_device_name(0), gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                warmup_videos=1)
    write(output/'run.json',plan)
    try:
        started = time.perf_counter()
        pipeline = wan.WanT2V(config=cfg, checkpoint_dir=str(checkpoint), device_id=0, rank=0,
                t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=False, convert_model_dtype=True)
        torch.cuda.synchronize()
        init_seconds = time.perf_counter()-started
        policy = Policy(args.policy) if args.policy else None
        # Full same-protocol warmup; no video save, no measured timing or score.
        warm_context = apply_policy(pipeline,policy,args.skip_budget) if policy else nullcontext(None)
        with warm_context as controller:
            warm = pipeline.generate(jobs[0]['prompt'], **generate(pipeline,controller))
        del warm
        profiler_class = load('timing', 'SeaCache4Wan22/runtime/inference_timing.py')._PipelineProfiler
        rows = []
        for job in jobs:
            sid = job['sample_id']
            context = apply_policy(pipeline,policy,args.skip_budget) if policy else nullcontext(None)
            with context as controller:
                profiler = profiler_class(pipeline, init_wall_seconds=init_seconds,
                    output_path=output/'timings'/f'{sid}.json', implementation=plan['mode'])
                profiler.install()
                video = pipeline.generate(job['prompt'], **generate(pipeline,controller))
            timing = read(output/'timings'/f'{sid}.json')
            trace = controller.trace() if controller else None
            if controller:
                timing['predictor'] = policy.overhead_summary()
                timing['latent_feature'] = trace['latent_feature_overhead']
                trace['latent_feature_file'] = f'../features/{sid}.pt'
                write(output/'traces'/f'{sid}.json',trace)
                (output/'features').mkdir(exist_ok=True)
                torch.save(torch.stack(controller.saved_features), output/'features'/f'{sid}.pt')
                write(output/'timings'/f'{sid}.json',timing)
            row = dict(sample_id=sid, **performance(timing,profile,trace))
            (output/'videos').mkdir(exist_ok=True)
            path = output/'videos'/f'{sid}.mp4'
            save_video(tensor=video[None], save_file=str(path), fps=16, nrow=1, normalize=True, value_range=(-1,1))
            del video
            probe = subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(path)], text=True)
            import json
            info = json.loads(probe)
            stream = info['streams'][0]
            from fractions import Fraction
            if (stream['width'],stream['height'],int(stream['nb_frames']),Fraction(stream['r_frame_rate'])) != (832,480,45,16):
                raise ValueError('exported video protocol mismatch')
            write(output/'ffprobe'/f'{sid}.json',info)
            row['video_sha256'] = sha256(path)
            row['timing_sha256'] = sha256(output/'timings'/f'{sid}.json')
            if trace:
                row['trace_sha256'] = sha256(output/'traces'/f'{sid}.json')
                row['feature_sha256'] = sha256(output/'features'/f'{sid}.pt')
            rows.append(row)
            write(output/'performance.json',dict(status='partial',rows=rows))
        write(output/'performance.json',dict(status='complete',rows=rows))
        write(output/'manifest.json',dict(**plan,status='complete',rows=rows))
    except BaseException as exc:
        write(output/'FAILED.json',dict(error=repr(exc)))
        raise
    return output
