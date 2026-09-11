"""Resident corrected SeaCache worker; keeps native reference videos untouched."""
import argparse
import json
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from ours4wan21.contracts import WORKSPACE, OFFICIAL, PROTOCOL, create_nested_result, dump, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','prompts','output-dir','result-parent'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);args=p.parse_args()
    cfg=json.loads(args.config.read_text());rows=[json.loads(x) for x in args.prompts.read_text().splitlines()]
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
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('need single visible GPU')
    torch.cuda.set_device(0);torch.set_num_threads(1)
    if torch.cuda.get_device_properties(0).total_memory<44*1024**3:raise ValueError('need 48GB GPU')
    uuid=physical_gpu_uuid(torch)
    if uuid!=cfg['gpu_uuids'][str(args.gpu)]:raise ValueError('same-GPU reference mismatch')
    source=WORKSPACE/'data/source/Wan2.1-65386b2';source_lock(source)
    model=checkpoint(WORKSPACE/'models/Wan2.1-T2V-1.3B')
    sys.path.insert(0,str(source));import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import cache_video
    out=create_nested_result(args.output_dir,args.result_parent,
        '# SeaCache 3.6x condition\n\nrun.json freezes settings; videos/traces/timings and components.json are measured candidates. Native warmup excluded.')
    for name in ('videos','traces','timings'):(out/name).mkdir()
    profile=json.loads(Path(cfg['flops_profile']).read_text())
    gen=dict(size=(832,480),frame_num=81,shift=5.,sample_solver='unipc',sampling_steps=50,guide_scale=5.,seed=42,offload_model=False)
    start=time.perf_counter()
    pipe=wan.WanT2V(config=WAN_CONFIGS['t2v-1.3B'],checkpoint_dir=str(model),device_id=0,rank=0,
                   t5_fsdp=False,dit_fsdp=False,use_usp=False,t5_cpu=False)
    prepare_resident(pipe)
    if pipe.param_dtype!=torch.bfloat16 or pipe.t5_cpu:raise ValueError('wrong residency/dtype')
    torch.cuda.synchronize();init=time.perf_counter()-start
    with torch.no_grad():warmup=pipe.generate(rows[0]['prompt_en'],**gen)
    del warmup;torch.cuda.synchronize()
    dump(out/'run.json',dict(protocol=PROTOCOL,method='seacache',gpu_uuid=uuid,prompts=rows,
        threshold=cfg['threshold'],config_sha256=sha256(args.config),flops_profile_sha256=sha256(Path(cfg['flops_profile'])),
        warmup='one native full generation excluded from measurement'))
    measurements=[]
    for row in rows:
        sid=row['sample_id']
        apply_seacache(pipe,task='t2v-1.3B',threshold=cfg['threshold'],trace_path=None,use_ret_steps=False)
        profiler=_PipelineProfiler(pipe,init_wall_seconds=init,output_path=out/'timings'/f'{sid}.json',implementation='seacache')
        profiler.install()
        with torch.no_grad():video=pipe.generate(row['prompt_en'],**gen)
        trace=pipe.model.seacache_controller.summary();dump(out/'traces'/f'{sid}.json',trace)
        t=json.loads((out/'timings'/f'{sid}.json').read_text())
        if t['status']!='success' or len(t['calls'])!=100:raise ValueError('incomplete timing')
        measurements.append(dict(sample_id=sid,generate_seconds=t['pipeline_generate_wall_seconds'],
            dit_tflops=flops_for_calls(t['calls'],profile,'seacache'),**extract_component_latency(t),**extract_component_tflops(profile)))
        cache_video(tensor=video[None],save_file=str(out/'videos'/f'{sid}.mp4'),fps=16,nrow=1,normalize=True,value_range=(-1,1))
        del video
        print(json.dumps(dict(completed=len(measurements),prompts=len(rows),sample_id=sid)),flush=True)
    dump(out/'components.json',dict(rows=measurements))
    dump(out/'COMPLETE.json',dict(status='generation_complete',videos=len(rows)))


if __name__=='__main__':main()
