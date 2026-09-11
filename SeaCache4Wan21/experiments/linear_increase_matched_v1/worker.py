"""One resident pipeline per physical GPU, five increasing then five fixed paths."""
import argparse
import json
import sys
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent
OFFICIAL=HERE.parents[2]
sys.path.insert(0,str(OFFICIAL/'Ours4Wan21'))
from ours4wan21.contracts import WORKSPACE, PROTOCOL, dump, sha256
from schedule import ScheduledController

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    root=a.root;cfg=json.loads((root/'config.json').read_text());row=cfg['prompts'][str(a.gpu)]
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
    with torch.no_grad():warmup=pipe.generate(row['prompt_en'],**gen)
    del warmup;torch.cuda.synchronize()
    for phase in ('increase','fixed'):
        if phase=='fixed':
            while not (root/'matching.json').exists():
                if (root/'FAILED.json').exists():raise RuntimeError('parent failed')
                time.sleep(2)
            matching=json.loads((root/'matching.json').read_text())
        for i, strategy in enumerate(cfg['strategies']):
            label=f'{phase}_{i+1}';out=root/'conditions'/label/f'gpu{a.gpu}'
            if (out/'COMPLETE.json').exists():continue
            out.mkdir(parents=True,exist_ok=True)
            for name in ('videos','timings','traces'):(out/name).mkdir(exist_ok=True)
            (out/'README.md').write_text('# Measured trajectory\n\nOne paired prompt; videos/, timings/, traces/, components.json.\n')
            path=strategy['path'] if phase=='increase' else [matching['pairs'][i]['threshold']]*50
            apply_seacache(pipe,task='t2v-1.3B',threshold=path[0],trace_path=None,use_ret_steps=False)
            pipe.model.seacache_controller=ScheduledController(path)
            sid=row['sample_id']
            dump(out/'run.json',dict(protocol=PROTOCOL,gpu_uuid=uuid,prompt=row,threshold_path=path,config_sha256=sha256(root/'config.json'),matching_sha256=sha256(root/'matching.json') if phase=='fixed' else None,warmup='one native full generation per GPU excluded'))
            profiler=_PipelineProfiler(pipe,init_wall_seconds=init,output_path=out/'timings'/f'{sid}.json',implementation='seacache');profiler.install()
            with torch.no_grad():video=pipe.generate(row['prompt_en'],**gen)
            trace=pipe.model.seacache_controller.summary();dump(out/'traces'/f'{sid}.json',trace)
            t=json.loads((out/'timings'/f'{sid}.json').read_text())
            if t['status']!='success' or len(t['calls'])!=100:raise ValueError('incomplete timing')
            metrics=dict(sample_id=sid,generate_seconds=t['pipeline_generate_wall_seconds'],dit_tflops=flops_for_calls(t['calls'],profile,'seacache'),**extract_component_latency(t),**extract_component_tflops(profile))
            cache_video(tensor=video[None],save_file=str(out/'videos'/f'{sid}.mp4'),fps=16,nrow=1,normalize=True,value_range=(-1,1));del video
            dump(out/'components.json',dict(rows=[metrics]))
            dump(out/'COMPLETE.json',dict(status='generation_complete',video_sha256=sha256(out/'videos'/f'{sid}.mp4')))
            print(json.dumps(dict(completed=label,sample_id=sid,seconds=metrics['generate_seconds'])),flush=True)
    dump(root/f'worker_gpu{a.gpu}_complete.json',dict(status='complete'))

if __name__=='__main__':main()
