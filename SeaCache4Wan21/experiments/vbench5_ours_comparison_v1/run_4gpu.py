"""Four resident workers, 15 prompt/threshold jobs, same-GPU baseline pairing."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import run_comparison as common

ROOT = common.EXP_ROOT / 'seacache_wan21_vbench5_ours_comparison_4gpu_v1'
read, dump, sha256 = common.read, common.dump, common.sha256


def build_jobs(prompts, mapping):
    return [dict(index=t*5+i, gpu=(t*5+i)%4, target_index=t, target=target,
                 threshold=common.threshold_for(mapping,target), **prompt)
            for t,target in enumerate(common.TARGETS) for i,prompt in enumerate(prompts)]


def worker(gpu, config):
    import torch
    from ours4wan21.shared import benchmark
    benchmark()
    from protocol import checkpoint, prepare_resident, source_lock
    from metrics import flops_for_calls
    from reporting import extract_component_latency, extract_component_tflops
    from ours4wan21.inference import physical_gpu_uuid
    sys.path.insert(0,str(common.PROJECT))
    from inference_timing import _PipelineProfiler
    from wan21_integration import apply_seacache
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('exactly one visible GPU required')
    torch.cuda.set_device(0);torch.set_num_threads(1)
    uuid=physical_gpu_uuid(torch)
    if uuid!=config['gpu_uuids'][str(gpu)] or torch.cuda.get_device_properties(0).total_memory<44*1024**3:
        raise ValueError('physical GPU identity/memory mismatch')
    source=common.WORKSPACE/'data/source/Wan2.1-65386b2'
    source_lock(source)
    model=checkpoint(common.WORKSPACE/'models/Wan2.1-T2V-1.3B')
    sys.path.insert(0,str(source))
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import cache_video
    jobs=[j for j in config['jobs'] if j['gpu']==gpu]
    for j in jobs:
        if (ROOT/f"target_{j['target_index']}"/'videos'/f"{j['sample_id']}.mp4").exists():
            raise FileExistsError('partial shard must be inspected before reuse')
    started=time.perf_counter()
    pipe=wan.WanT2V(config=WAN_CONFIGS['t2v-1.3B'],checkpoint_dir=str(model),device_id=0,
                    rank=0,t5_fsdp=False,dit_fsdp=False,use_usp=False,t5_cpu=False)
    prepare_resident(pipe)
    if pipe.param_dtype!=torch.bfloat16 or pipe.t5_cpu:raise ValueError('wrong residency/dtype')
    torch.cuda.synchronize();init=time.perf_counter()-started
    gen=dict(size=(832,480),frame_num=81,shift=5.,sample_solver='unipc',sampling_steps=50,
             guide_scale=5.,seed=42,offload_model=False)
    with torch.no_grad():warmup=pipe.generate(config['prompts'][0]['prompt'],**gen)
    del warmup
    torch.cuda.synchronize()
    dump(ROOT/f'run_gpu{gpu}.json',dict(gpu_uuid=uuid,jobs=jobs,protocol=common.PROTOCOL,
         warmup='one full native generation, excluded',config_sha256=sha256(ROOT/'config.json')))
    profile=read(common.PROFILE);rows=[]
    for j in jobs:
        out=ROOT/f"target_{j['target_index']}";sid=j['sample_id']
        apply_seacache(pipe,task='t2v-1.3B',threshold=j['threshold'],trace_path=None,use_ret_steps=False)
        profiler=_PipelineProfiler(pipe,init_wall_seconds=init,output_path=out/'timings'/f'{sid}.json',implementation='seacache')
        profiler.install()
        with torch.no_grad():video=pipe.generate(j['prompt'],**gen)
        trace=pipe.model.seacache_controller.summary()
        timing=read(out/'timings'/f'{sid}.json')
        if timing['status']!='success' or len(timing['calls'])!=100:raise ValueError('incomplete timing')
        dump(out/'traces'/f'{sid}.json',trace)
        row=dict(sample_id=sid,gpu=gpu,target_index=j['target_index'],threshold=j['threshold'],
                 generate_seconds=timing['pipeline_generate_wall_seconds'],
                 dit_tflops=flops_for_calls(timing['calls'],profile,'seacache'),
                 **extract_component_latency(timing),**extract_component_tflops(profile))
        cache_video(tensor=video[None],save_file=str(out/'videos'/f'{sid}.mp4'),fps=16,nrow=1,normalize=True,value_range=(-1,1))
        del video
        rows.append(row)
        dump(ROOT/f'components_gpu{gpu}.json',dict(rows=rows))
        print(json.dumps(dict(gpu=gpu,completed=len(rows),jobs=len(jobs),sample_id=sid,target=j['target'])),flush=True)
    dump(ROOT/f'COMPLETE_gpu{gpu}.json',dict(status='generation_complete',videos=len(rows)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',type=int,choices=range(4))
    args=parser.parse_args()
    if 'wan2.2' not in Path(sys.prefix).name.lower():raise ValueError('use Wan2.2 environment')
    prompts=read(common.PREVIOUS/'baselines/gpu0/run.json')['prompts']
    if [r['sample_id'] for r in prompts]!=[f'vbench200_{s}' for s in ('001','016','056','135','159')]:
        raise ValueError('wrong prompt subset')
    jobs=build_jobs(prompts,read(common.MAPPING));uuids={}
    paths=[Path(__file__),Path(common.__file__),common.MAPPING,common.PROFILE,
           common.PROJECT/'seacache.py',common.PROJECT/'wan21_integration.py',common.PROJECT/'inference_timing.py']
    for gpu in range(4):
        base=common.PREVIOUS/'baselines'/f'gpu{gpu}';m=read(base/'run.json')
        if m['prompts']!=prompts or m['protocol']!=common.PROTOCOL or read(base/'COMPLETE.json')['videos']!=5:
            raise ValueError('baseline mismatch')
        if m['flops_profile_sha256']!=sha256(common.PROFILE):raise ValueError('profile mismatch')
        uuids[str(gpu)]=m['gpu_uuid']
        paths.extend([base/'run.json',base/'components.json',*[base/'videos'/f"{r['sample_id']}.mp4" for r in prompts]])
    config=dict(protocol=common.PROTOCOL,prompts=prompts,jobs=jobs,gpu_uuids=uuids,
                source_sha256={str(p):sha256(p) for p in paths},vbench_enabled=False,adaptive_rescan=False,
                comparison='same nominal targets, report actual speeds; single-seed five-prompt diagnostic')
    if args.resume or args.worker is not None:
        if read(ROOT/'config.json')!=config:raise ValueError('frozen input mismatch')
    else:
        link=common.PROJECT/'experiment_results'/ROOT.name
        if ROOT.exists() or link.exists() or link.is_symlink():raise FileExistsError('use --resume')
        ROOT.mkdir();link.symlink_to(ROOT,target_is_directory=True)
        for name in ('logs','analysis','target_0','target_1','target_2'):
            (ROOT/name).mkdir();(ROOT/name/'README.md').write_text(f'# {name}\n\nFour-GPU five-prompt SeaCache comparison artifacts.\n')
        for t in range(3):
            for name in ('videos','references','timings','traces'):(ROOT/f'target_{t}'/name).mkdir()
        for j in jobs:
            (ROOT/f"target_{j['target_index']}"/'references'/f"{j['sample_id']}.mp4").symlink_to(
                common.PREVIOUS/'baselines'/f"gpu{j['gpu']}"/'videos'/f"{j['sample_id']}.mp4")
        (ROOT/'README.md').write_text('# SeaCache four-GPU comparison\n\nconfig.json freezes 15 jobs across GPUs0-3 (4/4/4/3). target_*/ stores videos, same-GPU reference symlinks, timings, traces and quality. analysis/ holds 39-condition merged comparison. Original three-GPU warmup was cancelled without completed measured videos.\n')
        dump(ROOT/'config.json',config);dump(ROOT/'status.json',dict(status='prepared',gpus=[0,1,2,3],jobs=15))
    if args.prepare_only:
        print(json.dumps(dict(root=str(ROOT),shards={g:sum(j['gpu']==g for j in jobs) for g in range(4)})));return
    if args.worker is not None:worker(args.worker,config);return
    with (ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                 TORCH_HOME=str(common.WORKSPACE/'models/torch-cache'),OPENBLAS_NUM_THREADS='1',
                 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        def generate(gpu):
            if (ROOT/f'COMPLETE_gpu{gpu}.json').exists():return
            with (ROOT/'logs'/f'gpu{gpu}.log').open('ab') as log:
                subprocess.run([sys.executable,'-u',str(Path(__file__)),'--worker',str(gpu)],env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
        def quality(t):
            out=ROOT/f'target_{t}'
            if not (out/'quality').exists():
                with (ROOT/'logs'/f'quality{t}.log').open('ab') as log:
                    subprocess.run([sys.executable,str(common.OFFICIAL/'VideoMetrics/evaluate.py'),'--reference-dir',str(out/'references'),
                        '--candidate-dir',str(out/'videos'),'--expected-frames','81','--device','cuda:0','--output-dir',str(out/'quality')],
                        env=dict(env,CUDA_VISIBLE_DEVICES=str(t)),stdout=log,stderr=subprocess.STDOUT,check=True)
            return common.quality_means(out/'quality',[r['sample_id'] for r in prompts])
        try:
            if (ROOT/'COMPLETE.json').exists():return
            dump(ROOT/'status.json',dict(status='waiting_for_gpus',gpus=[0,1,2,3]))
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():time.sleep(30)
            dump(ROOT/'status.json',dict(status='generating',gpus=[0,1,2,3],jobs=15))
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(generate,range(4)))
            allrows=[]
            for g in range(4):
                rows=read(ROOT/f'components_gpu{g}.json')['rows'];expected=[j for j in jobs if j['gpu']==g]
                if len(rows)!=len(expected) or read(ROOT/f'COMPLETE_gpu{g}.json')['videos']!=len(expected):raise ValueError('incomplete shard')
                if {(r['target_index'],r['sample_id']) for r in rows}!={(j['target_index'],j['sample_id']) for j in expected}:raise ValueError('shard identity mismatch')
                allrows+=rows
            dump(ROOT/'status.json',dict(status='quality',candidate_videos=15))
            with ThreadPoolExecutor(max_workers=3) as pool:qualities=list(pool.map(quality,range(3)))
            sea=[]
            for t,target in enumerate(common.TARGETS):
                cr=[r for r in allrows if r['target_index']==t];br=[]
                for r in cr:
                    baseline=read(common.PREVIOUS/'baselines'/f"gpu{r['gpu']}"/'components.json')['rows']
                    br.append(next(b for b in baseline if b['sample_id']==r['sample_id']))
                result=dict(label='SeaCache',epoch=None,k=None,target=target,threshold=common.threshold_for(read(common.MAPPING),target),
                    speedup=sum(r['generate_seconds'] for r in br)/sum(r['generate_seconds'] for r in cr),
                    seconds=statistics.fmean(r['generate_seconds'] for r in cr),**qualities[t])
                for key in ('dit_tflops','t5_cuda_seconds','dit_cuda_seconds','vae_decode_cuda_seconds','estimated_t5_tflops_per_video','estimated_vae_decode_tflops_per_video'):
                    result[key]=statistics.fmean(r[key] for r in cr)
                sea.append(result);dump(ROOT/f'target_{t}'/'RESULT.json',result)
            checked=json.loads(subprocess.check_output([sys.executable,str(common.OURS/'experiments/vbench5_speed_targets_v1/readout_completed.py')],env=env,text=True))
            result=checked['results']
            for r in result:r.update(target=dict(zip((23,29,35),common.TARGETS))[r['k']],threshold=None)
            result+=sea
            with (ROOT/'analysis/comparison.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in result for k in r)));writer.writeheader();writer.writerows(result)
            lines=['# 五提示词 SeaCache / Ours 对照','','同名义目标但不保证等速；单seed42，未计算VBench。SeaCache为仓库corrected filtered-boundary实现。',
                   '完整generate计时不含加载、warmup、视频保存或评测；每条候选与同物理卡baseline配对。','',
                   '|方法|名义目标|实测加速|秒/视频|PSNR dB|SSIM|LPIPS|','|---|---:|---:|---:|---:|---:|---:|']
            for r in sorted(result,key=lambda r:(r['target'],r['label'])):
                lines.append(f"|{r['label']}|{r['target']:.1f}|{r['speedup']:.3f}|{r['seconds']:.3f}|{r['psnr_rgb_db']:.3f}|{r['ssim_rgb']:.4f}|{r['lpips_alex_v0_1_spatial']:.4f}|")
            (ROOT/'analysis/COMPARISON.md').write_text('\n'.join(lines)+'\n')
            dump(ROOT/'analysis/VALIDATION.json',dict(status='pass',conditions=39,sea_video_pairs=15,sea_frame_pairs=1215,
                 config_sha256=sha256(ROOT/'config.json'),quality_sha256={str(ROOT/f'target_{t}/quality/summary.json'):sha256(ROOT/f'target_{t}/quality/summary.json') for t in range(3)}))
            dump(ROOT/'COMPLETE.json',dict(status='complete',comparison_sha256=sha256(ROOT/'analysis/comparison.csv')))
            dump(ROOT/'status.json',dict(status='complete',sea_results=sea))
        except BaseException as error:
            dump(ROOT/'FAILED.json',dict(error=repr(error)));dump(ROOT/'status.json',dict(status='failed',error=repr(error)));raise


if __name__=='__main__':main()
