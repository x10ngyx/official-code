"""Fixed-protocol generation for the three newly integrated Wan21 methods."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from protocol import ROOT, PACKAGES, checkpoint, external, prepare_resident, source_lock

METHODS={'magcache':'MagCache4Wan21','dicache':'DiCache4Wan21','taylorseer':'TaylorSeer4Wan21'}
PROTOCOL=dict(model='Wan2.1-T2V-1.3B',width=832,height=480,frames=81,fps=16,steps=50,
              solver='unipc',shift=5,cfg=5,seed=42,dit_compute_dtype='bfloat16',
              offload_model=False,t5_cpu=False,batch_size=1)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--baseline',action='store_true')
    p.add_argument('--wan21-root',type=Path,required=True)
    p.add_argument('--checkpoint-dir',type=Path,default=ROOT/'models/Wan2.1-T2V-1.3B')
    p.add_argument('--output-dir',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prompt')
    g.add_argument('--prompts',type=Path,help='JSONL with sample_id and prompt_en (or prompt); use Vbench200/prompts.jsonl for the standard evaluation')
    p.add_argument('--threshold',type=float,default=.08)
    p.add_argument('--max-skip-steps',type=int,default=4)
    p.add_argument('--retention-ratio',type=float,default=.2)
    p.add_argument('--fresh-threshold',type=int,default=5)
    return p.parse_args()


def load_prompts(args):
    rows=[dict(sample_id='sample_001',prompt=args.prompt)] if args.prompt else [json.loads(s) for s in args.prompts.read_text().splitlines() if s.strip()]
    ids=[]
    for r in rows:
        if 'prompt' not in r:
            r['prompt']=r.get('prompt_en')
        ident=r['sample_id']
        if not isinstance(ident,str) or not ident or ident in {'.','..'} or Path(ident).name!=ident:
            raise ValueError('unsafe sample_id')
        if not isinstance(r['prompt'],str) or not r['prompt'].strip():
            raise ValueError('empty prompt')
        ids.append(ident)
    if not rows or len(set(ids))!=len(ids):
        raise ValueError('empty or duplicated prompts')
    return rows


def main():
    args=parse_args()
    if Path(sys.prefix).name!='wan2.2':
        raise ValueError('use the wan2.2 conda environment')
    output=external(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use a fresh output directory; incompatible/partial results are not reused')
    source_lock(args.wan21_root)
    ckpt=checkpoint(args.checkpoint_dir)
    rows=load_prompts(args)
    sys.path.insert(0,str(PACKAGES/METHODS[args.method]))
    if args.method=='magcache':
        from magcache import MagCacheConfig
        MagCacheConfig(args.threshold,args.max_skip_steps,args.retention_ratio)
        from wan21_integration import apply_magcache
        def apply(pipe,path):
            apply_magcache(pipe,task='t2v-1.3B',threshold=args.threshold,max_skip_steps=args.max_skip_steps,
                           retention_ratio=args.retention_ratio,trace_path=str(path))
    elif args.method=='dicache':
        from dicache import DiCacheConfig
        DiCacheConfig(args.threshold,retention_ratio=args.retention_ratio)
        from wan21_integration import apply_dicache
        def apply(pipe,path):
            apply_dicache(pipe,task='t2v-1.3B',threshold=args.threshold,probe_depth=1,
                          retention_ratio=args.retention_ratio,dcta=True,trace_path=str(path))
    else:
        from taylorseer import apply_taylorseer
        if args.fresh_threshold<1:
            raise ValueError('fresh-threshold must be positive')
        def apply(pipe,path):
            apply_taylorseer(pipe,fresh_threshold=args.fresh_threshold,sample_steps=50)
    from inference_timing import _PipelineProfiler
    sys.path.insert(0,str(args.wan21_root))
    import torch
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import cache_video
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('select exactly one 48GB GPU with CUDA_VISIBLE_DEVICES')
    if torch.cuda.get_device_properties(0).total_memory < 44*1024**3:
        raise ValueError('fixed resident protocol requires a 48GB-class GPU')
    torch.cuda.set_device(0)
    started=time.perf_counter()
    pipe=wan.WanT2V(config=WAN_CONFIGS['t2v-1.3B'],checkpoint_dir=str(ckpt),device_id=0,
                   rank=0,t5_fsdp=False,dit_fsdp=False,use_usp=False,t5_cpu=False)
    prepare_resident(pipe)
    torch.cuda.synchronize()
    init_seconds=time.perf_counter()-started
    generation=dict(size=(832,480),frame_num=81,shift=5,sample_solver='unipc',sampling_steps=50,
                    guide_scale=5,seed=42,offload_model=False)
    # One native full-compute warmup, excluded from all reported measurements.
    with torch.no_grad():
        warm=pipe.generate(rows[0]['prompt'],**generation)
    del warm
    torch.cuda.synchronize()
    for folder in ['videos','timings','traces']:
        (output/folder).mkdir(parents=True,exist_ok=True)
    manifest=dict(protocol=PROTOCOL,method='baseline' if args.baseline else args.method,
                  method_parameters=dict(threshold=args.threshold,max_skip_steps=args.max_skip_steps,
                                         retention_ratio=args.retention_ratio,fresh_threshold=args.fresh_threshold),
                  prompts=rows,checkpoint_dir=str(ckpt),gpu=torch.cuda.get_device_name(0),
                  warmup='one native full generation',pipeline_init_seconds=init_seconds,
                  source_hashes={str(p.relative_to(PACKAGES)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for directory in [PACKAGES/METHODS[args.method],PACKAGES/'Wan21Benchmark']
                                 for p in directory.rglob('*.py') if 'experiment_results' not in p.parts})
    (output/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'README.md').write_text('# Fixed Wan21 inference\n\nrun.json locks protocol and source; videos/, timings/, traces/ contain per-sample artifacts. COMPLETE.json is written only after all generation succeeds.\n')
    try:
        for row in rows:
            sid=row['sample_id']
            trace=output/'traces'/f'{sid}.json'
            if not args.baseline:
                apply(pipe,trace)
            profiler=_PipelineProfiler(pipe,init_wall_seconds=init_seconds,
                                       output_path=output/'timings'/f'{sid}.json',
                                       implementation='wan21' if args.baseline else args.method)
            profiler.install()
            with torch.no_grad():
                video=pipe.generate(row['prompt'],**generation)
            if args.method=='taylorseer' and not args.baseline:
                timing=json.loads((output/'timings'/f'{sid}.json').read_text())
                trace.write_text(json.dumps(dict(schema='taylorseer_wan21_trace_v1',
                    fresh_threshold=args.fresh_threshold,max_order=1,first_enhance=1,
                    calls=timing['calls']),indent=2)+'\n')
            cache_video(tensor=video[None],save_file=str(output/'videos'/f'{sid}.mp4'),fps=16,
                        nrow=1,normalize=True,value_range=(-1,1))
            del video
        (output/'COMPLETE.json').write_text(json.dumps(dict(status='generation_complete',videos=len(rows)))+'\n')
    except BaseException as exc:
        (output/'FAILED.json').write_text(json.dumps(dict(error=repr(exc)))+'\n')
        raise


if __name__=='__main__':
    main()
