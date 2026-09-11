"""Persistent resident Wan21 worker, with sealed per-video restart boundaries."""
from common import ROOT
import argparse
import fcntl
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path
import sys
import time

from ours4wan21.contracts import OFFICIAL, PROJECT, PROTOCOL, dump, sha256
from ours4wan21.online_common import prepare_directory, read, seal, verified, require_environment


def job_identity(job, manifest):
    return dict(job=job,protocol=PROTOCOL,wan_checkpoint=manifest['paths']['wan_checkpoint'],
                profile_sha256=manifest['inputs']['flops_profile']['sha256'],
                source_hashes=manifest['source_hashes'])


def generate_jobs(run, jobs_path):
    require_environment()
    import torch
    from ours4wan21.policy import Policy
    from ours4wan21.runtime import apply_policy
    from ours4wan21.shared import benchmark
    from ours4wan21.overhead import predictor_fields
    from common import verify_sources
    from ours4wan21.inference import physical_gpu_uuid
    run=Path(run); manifest=read(run/'config.json'); verify_sources(manifest); jobs=read(jobs_path)
    jobs=[j for j in jobs if not verified(j['output'],job_identity(j,manifest))]
    if not jobs:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('worker requires exactly one visible GPU')
    torch.cuda.set_device(0);torch.set_num_threads(1)
    prop=torch.cuda.get_device_properties(0)
    if prop.total_memory < 44*1024**3:
        raise ValueError('requires a 48GB-class GPU with all Wan21 components resident')
    if torch.cuda.mem_get_info(0)[0] < 40*1024**3:
        raise ValueError('generation GPU is not sufficiently free; choose an idle GPU')
    uuid=physical_gpu_uuid(torch)
    if any(j.get('expected_gpu_uuid',uuid)!=uuid for j in jobs):
        raise ValueError('candidate GPU differs from the reused native baseline GPU')
    benchmark()
    from protocol import source_lock,checkpoint,prepare_resident
    from metrics import flops_for_calls
    from reporting import extract_component_latency,extract_component_tflops
    sys.path.insert(0,str(OFFICIAL/'SeaCache4Wan21'))
    from inference_timing import _PipelineProfiler
    root=Path(manifest['paths']['wan21_root']); source_lock(root)
    model_dir=checkpoint(Path(manifest['paths']['wan_checkpoint']))
    sys.path.insert(0,str(root))
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import cache_video
    profile=read(manifest['paths']['flops_profile'])
    components=extract_component_tflops(profile)
    generation=dict(size=(832,480),frame_num=81,shift=5.,sample_solver='unipc',
                    sampling_steps=50,guide_scale=5.,seed=42,offload_model=False)
    # Serialize CPU loading and native warmup across worker starts.
    with (run/'initialization.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        began=time.perf_counter()
        pipe=wan.WanT2V(config=WAN_CONFIGS['t2v-1.3B'],checkpoint_dir=str(model_dir),
            device_id=0,rank=0,t5_fsdp=False,dit_fsdp=False,use_usp=False,t5_cpu=False)
        prepare_resident(pipe)
        if pipe.param_dtype!=torch.bfloat16 or pipe.t5_cpu:
            raise ValueError('invalid resident BF16/T5 configuration')
        torch.cuda.synchronize();initialization=time.perf_counter()-began
        with torch.no_grad():
            warm=pipe.generate(jobs[0]['prompt'],**generation)
        del warm
        torch.cuda.synchronize()
        native_forward,native_generate=pipe.model.forward,pipe.generate
    policies={}
    for job in jobs:
        identity=job_identity(job,manifest)
        out=Path(job['output'])
        if not prepare_directory(out,identity):
            continue
        pipe.model.forward=native_forward;pipe.generate=native_generate
        controller=policy=None
        if job['kind']!='baseline':
            ckpt=job['checkpoint']
            if sha256(ckpt['path'])!=ckpt['sha256']:
                raise ValueError('policy checkpoint changed during dispatch')
            if ckpt['sha256'] not in policies:
                policies[ckpt['sha256']]=Policy(ckpt['path'],device='cuda:0',state_mode=job['state_mode'])
            policy=policies[ckpt['sha256']]
            policy.set_sampling(job.get('sampling_seed'))
            controller=apply_policy(pipe,policy,job['skip_budget'])
        profiler=_PipelineProfiler(pipe,init_wall_seconds=initialization,
            output_path=out/'timing.json',implementation='ours' if policy else 'wan21')
        profiler.install()
        with torch.no_grad():
            video=pipe.generate(job['prompt'],**generation)
        trace=controller.summary() if controller else dict(total_steps=50,step_reuse=0,step_recompute=50)
        timing=read(out/'timing.json')
        if timing['status']!='success':
            raise ValueError('measured generation failed')
        overhead={}
        if policy:
            timing['predictor']=policy.overhead_summary()
            timing['latent_feature']=controller.feature_overhead()
            overhead=predictor_fields(timing,trace)
            overhead['latent_feature_wall_seconds']=timing['latent_feature']['wall_seconds']
            dump(out/'timing.json',timing)
        measured=dict(generate_seconds=timing['pipeline_generate_wall_seconds'],
            dit_tflops=flops_for_calls(timing['calls'],profile,'ours' if policy else 'baseline'),
            **extract_component_latency(timing),**components,**overhead)
        if not math.isfinite(measured['generate_seconds']) or measured['generate_seconds']<=0:
            raise ValueError('invalid complete generate latency')
        dump(out/'trace.json',trace)
        dump(out/'measurement.json',measured)
        dump(out/'generation.json',dict(identity=identity,gpu_name=prop.name,gpu_uuid=uuid,
            protocol=PROTOCOL,measurement_scope='complete generate, excluding native warmup, model load and export',
            predictor_scope='predictor network separate; nested in DiT/generate; not added twice'))
        cache_video(tensor=video[None],save_file=str(out/'video.mp4'),fps=16,nrow=1,
                    normalize=True,value_range=(-1,1))
        del video
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
            '-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate',
            '-of','json',str(out/'video.mp4')],text=True))
        stream=probe['streams'][0]
        if ((stream['width'],stream['height'],int(stream['nb_read_frames']))!=(832,480,81)
                or Fraction(stream['r_frame_rate'])!=16):
            raise ValueError('encoded video violates resolution/frame-count/FPS protocol')
        dump(out/'ffprobe.json',probe)
        seal(out,['identity.json','generation.json','video.mp4','trace.json','measurement.json','timing.json','ffprobe.json'],identity=identity)
        print('completed '+job['output'],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--jobs",type=Path,required=True);a=p.parse_args()
    generate_jobs(ROOT,a.jobs)
