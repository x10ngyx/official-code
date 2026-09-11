"""Persistent single-GPU generation, atomic cells and strict resume checks."""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from common import (PACKAGES, PROJECT, WORKSPACE, PROTOCOL, cell_path, completed,
                    digest, external, job_plan, read_json, sha, source_hashes,
                    steps, validate_timing, validate_trace, write_json)


def mse_frames(reference, candidate):
    import numpy as np
    if reference.shape != candidate.shape or len(reference.shape) != 4:
        raise ValueError('RGB shapes must match T,C,H,W')
    values = []
    for a,b in zip(reference, candidate):
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('nonfinite RGB')
        values.append(float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2)))
    return values


def performance(timing, profile, skip_step):
    sys.path.insert(0, str(PACKAGES/'ComponentMetrics'))
    from reporting import extract_component_latency, extract_component_tflops
    validate_timing(timing, skip_step)
    if profile['input']['video_shape_fhw'] != [81,480,832]:
        raise ValueError('FLOPs profile shape mismatch')
    full = profile['per_model_forward']['estimated_full_flops']
    always = profile['per_model_forward']['estimated_always_on_flops']
    if not 0 <= always <= full or profile['input']['transformer_blocks'] != 30:
        raise ValueError('invalid FLOPs paths')
    return dict(generate_seconds=timing['pipeline_generate_wall_seconds'],
                estimated_dit_tflops=sum(full if r['blocks_executed']==30 else always
                                         for r in timing['calls'])/1e12,
                **extract_component_latency(timing), **extract_component_tflops(profile),
                scope='instrumented full generate incl. diagnostic hashing/pooling/proxies; DiT counts exclude diagnostic operations')


def collect(args, rows):
    if Path(sys.prefix).name != 'wan2.2':
        raise ValueError('activate conda environment wan2.2')
    import numpy as np
    import torch
    sys.path.insert(0, str(PACKAGES/'Wan21Benchmark'))
    from protocol import source_lock, checkpoint, prepare_resident
    from runtime import SingleSkipController, install
    from inference_timing import _PipelineProfiler
    source_lock(args.wan21_root)
    ckpt = checkpoint(args.checkpoint_dir)
    profile = read_json(args.profile)
    expected_profile = dict(task='t2v-1.3B', video_shape_fhw=[81,480,832],
                            sampling_steps=50, solver='unipc', shift=5.0,
                            cfg=5.0, seed=42, parameter_dtype='bfloat16')
    if any(profile['input'].get(k) != v for k,v in expected_profile.items()):
        raise ValueError('profile does not match fixed inference protocol')
    if profile['source']['wan21_generate_sha256'] != sha(Path(args.wan21_root)/'generate.py'):
        raise ValueError('profile Wan source mismatch')
    output = external(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'.worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Keep lock alive until the function exits. Never allow two writers to one run.
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('select exactly one GPU with CUDA_VISIBLE_DEVICES')
    device = torch.cuda.get_device_properties(0)
    if device.total_memory < 44*1024**3:
        raise ValueError('48GB-class GPU required by resident protocol')
    torch.cuda.set_device(0)
    gpu_uuid = str(getattr(device,'uuid','unavailable'))
    if gpu_uuid == 'unavailable':
        selected = os.environ.get('CUDA_VISIBLE_DEVICES','0')
        gpu_uuid = subprocess.check_output(['nvidia-smi','-i',selected,
                    '--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        if not gpu_uuid.startswith(('GPU-','MIG-')) or '\n' in gpu_uuid:
            raise ValueError('cannot establish one physical GPU identity')
    gpu_identity = dict(name=device.name, uuid=gpu_uuid,
                        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    weights = {str(p.relative_to(ckpt)): sha(p) for p in sorted(ckpt.rglob('*')) if p.is_file()}
    contract = dict(protocol=PROTOCOL, prompts=rows, sources=source_hashes(), weights=weights,
                    wan_sources={str(p.relative_to(args.wan21_root)):sha(p)
                                 for p in sorted(Path(args.wan21_root).rglob('*.py'))},
                    profile_sha256=sha(args.profile), gpu=gpu_identity,
                    torch_version=torch.__version__, cuda=torch.version.cuda,
                    rgb='VAE output clamped and mapped from [-1,1] to float32 [0,1], TCHW, before MP4',
                    coarse_steps_1based=steps('coarse'), remaining_steps_1based=steps('remaining'))
    contract_hash = digest(contract)
    if (output/'run.json').exists():
        if read_json(output/'run.json')['contract_hash'] != contract_hash:
            raise ValueError('resume contract differs (source/model/profile/prompts/GPU/software); use a new result directory')
    else:
        write_json(output/'run.json', dict(contract_hash=contract_hash, **contract))
        if Path(args.profile).resolve() != (output/'component_profile.json').resolve():
            shutil.copyfile(args.profile, output/'component_profile.json')
        (output/'README.md').write_text('# Single-skip dataset v2\n\nrun.json and manifests/ freeze provenance and ordered identities. shared_baselines/ and shards/shard_00/candidates/ contain full trajectory bundles with 50 FP16 input latents. COMPLETE.json marks generation; completed/ contains canonical-quality-finalized candidates with trajectory/step/branch rows. published/current/tables/ exposes immutable JSONL/CSV snapshots after publish. Raw RGB terminal MSE is separate from canonical MP4 VideoMetrics. sessions/ contains excluded native-equivalence checks; _incomplete/ archives interrupted cells.\n')
    from dataset import initialize_manifest, save_capture, probe_video
    dataset_records = initialize_manifest(output,rows)
    link = PROJECT/'experiment_results'/output.name
    if link.is_symlink():
        if link.resolve()!=output:
            raise ValueError('result symlink name collision')
    elif link.exists():
        raise ValueError('result link path exists')
    else:
        link.symlink_to(output, target_is_directory=True)
    jobs = job_plan(rows, args.stage)
    done = set()
    for step,row in jobs:
        key = (step,row['sample_id'])
        if completed(cell_path(output,*key), contract_hash):
            done.add(key)
    if args.stage == 'remaining':
        for step in steps('coarse'):
            for row in rows:
                if not completed(cell_path(output,step,row['sample_id']),contract_hash):
                    raise ValueError('finish coarse stage before remaining')
    pending = [(s,r) for s,r in jobs if (s,r['sample_id']) not in done]
    if not pending:
        print('Requested generation stage already complete; verified hashes.')
        return
    sys.path.insert(0, str(Path(args.wan21_root).resolve()))
    import wan
    from wan.configs import WAN_CONFIGS
    import imageio.v2 as imageio
    started = time.perf_counter()
    pipe = wan.WanT2V(config=WAN_CONFIGS['t2v-1.3B'], checkpoint_dir=str(ckpt),
                      device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False,
                      use_usp=False, t5_cpu=False)
    prepare_resident(pipe)
    if pipe.param_dtype != torch.bfloat16 or len(pipe.model.blocks)!=30:
        raise ValueError('unexpected DiT dtype or architecture')
    torch.cuda.synchronize()
    init_seconds = time.perf_counter()-started
    generation = dict(size=(832,480), frame_num=81, shift=5, sample_solver='unipc',
                      sampling_steps=50, guide_scale=5, seed=42, offload_model=False)
    # Strict native-vs-instrumented all-compute check on the actual remote GPU.
    # Both are excluded warmup/validation generations, not dataset trajectories.
    with torch.no_grad():
        native = pipe.generate(rows[0]['prompt'], **generation).cpu()
        control = SingleSkipController(0)
        install(pipe, control)
        instrumented = pipe.generate(rows[0]['prompt'], **generation).cpu()
    if not torch.equal(native, instrumented):
        raise ValueError('native/instrumented full-compute outputs differ; abort before data collection')
    del native, instrumented
    control.input_latents.clear()
    write_json(output/'sessions'/f'{time.time_ns()}.json', dict(
        native_equivalence='bitwise_equal_float32_video', excluded_generations=2,
        pipeline_init_seconds=init_seconds, gpu=gpu_identity, stage=args.stage))
    produced = 0
    feature_cache = {}
    for skip_step,row in pending:
        if args.max_new_cells is not None and produced >= args.max_new_cells:
            break
        sid = row['sample_id']
        target = cell_path(output,skip_step,sid)
        base = cell_path(output,0,sid)
        if skip_step:
            if not (base/'COMPLETE.json').exists():
                raise ValueError('missing baseline')
            if sid not in feature_cache:
                feature_cache[sid] = torch.load(base/'features.pt',map_location='cpu',weights_only=True)
            base_features = feature_cache[sid]
        else:
            base_features = None
        if target.exists():
            archive = output/'_incomplete'/f'{target.parent.name}_{sid}_{uuid.uuid4().hex}'
            archive.parent.mkdir(exist_ok=True)
            target.rename(archive)
        target.mkdir(parents=True)
        ctrl = SingleSkipController(skip_step, base_features)
        install(pipe, ctrl)
        profiler = _PipelineProfiler(pipe,init_wall_seconds=init_seconds,
                                     output_path=target/'timing.json',implementation='single_skip_wan21')
        profiler.install()
        print(f'BEGIN step={skip_step:02d} prompt={sid}', flush=True)
        with torch.no_grad():
            video = pipe.generate(row['prompt'], **generation)
        trace = ctrl.summary()
        validate_trace(trace, skip_step)
        timing = read_json(target/'timing.json')
        perf = performance(timing, profile, skip_step)
        write_json(target/'performance.json',perf)
        trace = save_capture(ctrl,target,base,dataset_records[(skip_step,sid)])
        features = ctrl.features()
        if skip_step:
            if not torch.equal(features['g1'][skip_step], base_features['g1'][skip_step]):
                raise ValueError('pre-action G1 differs from baseline')
            if features['proxies'][skip_step] != base_features['proxies'][skip_step]:
                raise ValueError('pre-action proxies differ from baseline')
        rgb = video.detach().float().clamp(-1,1).add(1).mul(.5).permute(1,0,2,3).contiguous().cpu().numpy()
        del video
        if rgb.shape!=(81,3,480,832) or not np.isfinite(rgb).all():
            raise ValueError('invalid decoded RGB output')
        np.save(target/'rgb_f32.npy', rgb, allow_pickle=False)
        frame_mse = mse_frames(np.load(base/'rgb_f32.npy',mmap_mode='r'), rgb) if skip_step else [0.]*81
        # MP4 is presentation/VBench only. Primary MSE never reads lossy MP4.
        with imageio.get_writer(str(target/'video.mp4'), fps=16, codec='libx264',
                                quality=8, macro_block_size=None, ffmpeg_params=['-threads','1']) as writer:
            for frame in rgb:
                writer.append_data((frame.transpose(1,2,0)*255).round().astype(np.uint8))
        (target/('candidate.mp4' if skip_step else 'baseline.mp4')).symlink_to('video.mp4')
        write_json(target/'ffprobe.json',probe_video(target/'video.mp4'))
        del rgb
        torch.save(features, target/'features.pt')
        write_json(target/'metrics.json',dict(sample_id=sid,skip_step_1based=skip_step,
                   skip_step_index_0based=skip_step-1 if skip_step else None,
                   terminal_rgb_mse=float(np.mean(frame_mse)),per_frame_rgb_mse=frame_mse,
                   baseline_rgb_sha256=sha(base/'rgb_f32.npy'),
                   **perf))
        files = {str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
        write_json(target/'COMPLETE.json',dict(contract_hash=contract_hash,files=files,
                   status='generation_and_primary_mse_complete',quality_evaluation='pending'))
        produced += 1
        from evaluate import summarize
        summarize(output)
        print(f'DONE step={skip_step:02d} prompt={sid} MSE={np.mean(frame_mse):.9g}',flush=True)
    for stage in ['coarse','all']:
        cells = job_plan(rows,stage)
        if all((cell_path(output,s,r['sample_id'])/'COMPLETE.json').is_file() for s,r in cells):
            write_json(output/f'{stage.upper()}_GENERATION_COMPLETE.json',dict(
                contract_hash=contract_hash, cells=len(cells), quality_evaluation='separate'))
