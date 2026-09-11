"""One resident pipeline per physical GPU, five increasing then five fixed paths."""
import argparse
import json
import sys
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent
OFFICIAL=HERE.parents[2]
sys.path.insert(0,str(OFFICIAL/"SeaCache4Wan21/experiments/linear_increase_matched_v1"))
sys.path.insert(0,str(OFFICIAL/'Ours4Wan21'))
from ours4wan21.contracts import WORKSPACE, PROTOCOL, dump, sha256
from schedule import ScheduledController

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    root=a.root;cfg=json.loads((root/'config.json').read_text());jobs=cfg['jobs'][str(a.gpu)];row=jobs[0]['prompt']
    import torch
    from ours4wan21.shared import benchmark
    benchmark()
    from protocol import checkpoint,prepare_resident,source_lock
    from metrics import flops_for_calls
    from reporting import extract_component_latency,extract_component_tflops
    from ours4wan21.inference import physical_gpu_uuid
    sys.path.insert(0,str(OFFICIAL/'SeaCache4Wan21'))
    from inference_timing import _PipelineProfiler
    from wan21_integration import apply_seacache
    if torch.cuda.device_count()!=1:raise ValueError('single visible GPU required')
    torch.cuda.set_device(0);torch.set_num_threads(1)
    uuid=physical_gpu_uuid(torch)
    if uuid!=cfg['gpu_uuids'][str(a.gpu)] or torch.cuda.get_device_properties(0).total_memory<44*1024**3:raise ValueError('GPU mismatch')
    source=WORKSPACE/'data/source/Wan2.1-65386b2';source_lock(source);sys.path.insert(0,str(source))
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import cache_video
    model=checkpoint(WORKSPACE/'models/Wan2.1-T2V-1.3B')
    profile=json.loads(Path(cfg['flops_profile']).read_text())
    gen=dict(size=(832,480),frame_num=81,shift=5.,sample_solver='unipc',sampling_steps=50,guide_scale=5.,seed=42,offload_model=False)
    start=time.perf_counter()
    pipe=wan.WanT2V(config=WAN_CONFIGS['t2v-1.3B'],checkpoint_dir=str(model),device_id=0,rank=0,t5_fsdp=False,dit_fsdp=False,use_usp=False,t5_cpu=False)
    prepare_resident(pipe)
    if pipe.param_dtype!=torch.bfloat16 or pipe.t5_cpu:raise ValueError('wrong dtype/residency')
    torch.cuda.synchronize();init=time.perf_counter()-start
    # Warm the same increasing controller; discarded, excluded from measurements.
    warm_path=jobs[0]['threshold_path']
    apply_seacache(pipe,task='t2v-1.3B',threshold=warm_path[0],trace_path=None,use_ret_steps=False)
    pipe.model.seacache_controller=ScheduledController(warm_path)
    with torch.no_grad():warmup=pipe.generate(row['prompt'],**gen)
    del warmup;torch.cuda.synchronize()
    for job in jobs:
        row=job['prompt'];sid=row['sample_id'];out=Path(job['output'])
        done=out/'COMPLETE.json'
        if done.exists():
            seal=json.loads(done.read_text())
            assert seal['config_sha256']==sha256(root/'config.json')
            assert all(sha256(out/f)==h for f,h in seal['files'].items())
            continue
        out.mkdir(parents=True,exist_ok=True)
        (out/'README.md').write_text('# Increase candidate\n\nvideo.mp4, timing.json, trace.json, measurement.json and sealed generation identity.\n')
        path=job['threshold_path']
        apply_seacache(pipe,task='t2v-1.3B',threshold=path[0],trace_path=None,use_ret_steps=False)
        pipe.model.seacache_controller=ScheduledController(path)
        dump(out/'generation.json',dict(protocol=PROTOCOL,gpu_uuid=uuid,job=job,config_sha256=sha256(root/'config.json'),warmup='one increase warmup per GPU discarded'))
        profiler=_PipelineProfiler(pipe,init_wall_seconds=init,output_path=out/'timing.json',implementation='seacache');profiler.install()
        with torch.no_grad():video=pipe.generate(row['prompt'],**gen)
        trace=pipe.model.seacache_controller.summary();dump(out/'trace.json',trace)
        t=json.loads((out/'timing.json').read_text())
        assert t['status']=='success' and len(t['calls'])==100
        metrics=dict(sample_id=sid,generate_seconds=t['pipeline_generate_wall_seconds'],dit_tflops=flops_for_calls(t['calls'],profile,'seacache'),**extract_component_latency(t),**extract_component_tflops(profile))
        cache_video(tensor=video[None],save_file=str(out/'video.mp4'),fps=16,nrow=1,normalize=True,value_range=(-1,1));del video
        dump(out/'measurement.json',metrics)
        dump(done,dict(status='generation_complete',config_sha256=sha256(root/'config.json'),files={n:sha256(out/n) for n in ['video.mp4','timing.json','trace.json','measurement.json','generation.json']}))
        print(json.dumps(dict(completed=job['target'],sample_id=sid,seconds=metrics['generate_seconds'])),flush=True)
    dump(root/f'worker_gpu{a.gpu}_complete.json',dict(status='complete'))

if __name__=='__main__':main()
