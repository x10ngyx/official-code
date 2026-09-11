"""Paired SeaCache/Dynamics128 nominal3.6x, calibrated and fixed, previous50 references."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'experiments/dynamics128_vbench50_4gpu_v1'))
import run_suite as prior
EXP,OFFICIAL,WORKSPACE=prior.EXP_ROOT,prior.OFFICIAL,prior.WORKSPACE
NAME='wan21_vbench50_speed36_seacache_dynamics128_e391_v1'
ROOT=EXP/NAME
METHODS=('seacache','dynamics128')
read,dump,sha,now=prior.read_json,prior.dump,prior.sha256,prior.now


def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def interpolate_threshold(rows,target=3.6):
    rows=sorted(rows,key=lambda r:float(r['speedup']))
    for a,b in zip(rows,rows[1:]):
        lo,hi=float(a['speedup']),float(b['speedup'])
        if lo<=target<=hi:
            ta,tb=float(a['threshold']),float(b['threshold'])
            if tb<=ta:raise ValueError('nonmonotone calibration bracket')
            return ta+(tb-ta)*(target-lo)/(hi-lo),[a,b]
    raise ValueError('target outside measured bracket; no extrapolation')


def select_k(rows,target=3.6):
    if {int(r['k']) for r in rows}!={23,29,35}:raise ValueError('need prior complete K23/K29/K35')
    x=[float(r['k']) for r in rows];y=[float(r['candidate_generate_seconds_mean']) for r in rows]
    mx,my=statistics.fmean(x),statistics.fmean(y)
    slope=sum((a-mx)*(b-my) for a,b in zip(x,y))/sum((a-mx)**2 for a in x)
    intercept=my-slope*mx;baseline=float(rows[0]['baseline_generate_seconds_mean'])
    if slope>=0 or baseline<=0:raise ValueError('invalid latency fit')
    fractional=(baseline/target-intercept)/slope
    options=[math.floor(fractional),math.ceil(fractional)]
    if not all(0<=k<=47 and intercept+slope*k>0 for k in options):raise ValueError('unfeasible K')
    speeds={str(k):baseline/(intercept+slope*k) for k in options}
    k=min(options,key=lambda k:(abs(speeds[str(k)]-target),k))
    return k,dict(slope=slope,intercept=intercept,fractional_k=fractional,predicted_speedups=speeds,
        selection='closest predicted full latency speedup, smaller K on tie; validate once on four prompts')


def validate_speed(speed,target=3.6):
    if not math.isfinite(speed) or abs(speed/target-1)>.05:
        raise ValueError(f'calibration speed {speed} outside target +/-5%; stop, no adaptive rescan')


def prepare(resume):
    previous,pc=prior.freeze(resume=True)
    done=read(previous/'COMPLETE.json')
    if done['status']!='complete' or sha(previous/'analysis/results.csv')!=done['results_sha256']:
        raise ValueError('prior50 final results mismatch')
    mapping=EXP/'wan21_seacache_speedup_calibration_v1/analysis/speed_threshold_mapping.calibrated.json'
    cal=Path(read(mapping)['fit_source'])
    if sha(cal)!=read(mapping)['fit_source_sha256']:raise ValueError('calibration source hash mismatch')
    threshold,bracket=interpolate_threshold(read_csv(cal))
    k,fit=select_k(read_csv(previous/'analysis/results.csv'))
    rows=[json.loads(x) for x in (previous/'prompts/selected.jsonl').read_text().splitlines()]
    shards=prior.partition(rows)
    probes={g:[random.Random(360+g).choice(rs)] for g,rs in shards.items()}
    sources=[Path(__file__),HERE/'sea_worker.py',Path(prior.__file__),mapping,cal,
        previous/'config.json',previous/'COMPLETE.json',previous/'analysis/results.csv',previous/'prompts/selected.jsonl',
        Path(pc['checkpoint']),Path(pc['flops_profile']),OFFICIAL/'SeaCache4Wan21/seacache.py',
        OFFICIAL/'SeaCache4Wan21/wan21_integration.py',OFFICIAL/'SeaCache4Wan21/inference_timing.py',
        *sorted((PROJECT/'ours4wan21').glob('*.py')),*sorted((OFFICIAL/'VideoMetrics/video_metrics').glob('*.py'))]
    gpu_uuids={};baseline_hashes={}
    for g,rs in shards.items():
        base=previous/'shards'/f'gpu{g}'/'baseline'
        bm=prior.validate_generation(base,prompt_ids=[r['sample_id'] for r in rs],method='baseline')
        if bm['protocol']!=prior.PROTOCOL:raise ValueError('baseline protocol mismatch')
        gpu_uuids[str(g)]=bm['gpu_uuid']
        sources+=[base/'run.json',base/'components.json',base/'COMPLETE.json']
        for r in rs:
            sid=r['sample_id']
            for folder,suffix in [('videos','.mp4'),('timings','.json'),('traces','.json')]:
                f=base/folder/(sid+suffix);baseline_hashes[str(f)]=sha(f)
    cfg=dict(schema='paired50_speed36_v1',previous=str(previous),protocol=prior.PROTOCOL,prompt_count=50,
        mode=prior.MODE,checkpoint=pc['checkpoint'],checkpoint_sha256=pc['checkpoint_sha256'],selected_epoch=391,
        target=3.6,k=k,threshold=threshold,threshold_bracket=bracket,k_latency_fit=fit,
        calibration_policy='4 prompts per method, speed-only verification +/-5%; no adaptive K/threshold rescan',
        probe_ids={str(g):[r['sample_id'] for r in rs] for g,rs in probes.items()},
        shard_ids={str(g):[r['sample_id'] for r in rs] for g,rs in shards.items()},gpu_uuids=gpu_uuids,
        flops_profile=pc['flops_profile'],baseline_policy='reuse previous50 same physical GPU; no new reference generation',
        vbench_enabled=False,video_metrics=list(prior.METRICS),source_sha256={str(p):sha(p) for p in sources},baseline_sha256=baseline_hashes)
    if resume:
        if read(ROOT/'config.json')!=cfg:raise ValueError('frozen input changed')
    else:
        prior.create_result(ROOT,'# SeaCache / Dynamics128 e391, nominal3.6x paired50\n\nconfig.json freezes original50 references and calibration choices. prompts/, references/, calibration/, shards/, analysis/, logs/ retain provenance; no VBench scoring.')
        sea_link=OFFICIAL/'SeaCache4Wan21/experiment_results'/NAME
        if sea_link.exists() or sea_link.is_symlink():raise FileExistsError(sea_link)
        sea_link.symlink_to(ROOT,target_is_directory=True)
        dump(ROOT/'config.json',cfg)
        for name in ('prompts','references','calibration','shards','analysis','logs'):
            (ROOT/name).mkdir();(ROOT/name/'README.md').write_text(f'# {name}\n\nPaired50 nominal3.6x {name}; see ../README.md.\n')
        for g in prior.GPUS:
            for prefix,rs in [('formal',shards[g]),('probe',probes[g])]:
                (ROOT/'prompts'/f'{prefix}_gpu{g}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
            refs=ROOT/'references'/f'gpu{g}';refs.mkdir()
            (refs/'README.md').write_text('# Reference shard\n\nVideo symlinks to completed same-GPU native baseline.\n')
            vids=refs/'videos';vids.mkdir()
            for r in shards[g]:(vids/(r['sample_id']+'.mp4')).symlink_to(previous/'shards'/f'gpu{g}'/'baseline/videos'/(r['sample_id']+'.mp4'))
        dump(ROOT/'status.json',dict(status='prepared',updated_at=now(),vbench_status='skipped_by_user'))
    for g in prior.GPUS:
        for prefix,rs in [('formal',shards[g]),('probe',probes[g])]:
            if [json.loads(x) for x in (ROOT/'prompts'/f'{prefix}_gpu{g}.jsonl').read_text().splitlines()]!=rs:
                raise ValueError('frozen prompt shard changed')
    return cfg


def check_generation(d,g,method,ids,cfg):
    m=prior.validate_generation(d,prompt_ids=ids,method='ours' if method=='dynamics128' else 'seacache',
        mode=prior.MODE,skip_budget=cfg['k'] if method=='dynamics128' else None)
    if m['protocol']!=prior.PROTOCOL or m['gpu_uuid']!=cfg['gpu_uuids'][str(g)]:raise ValueError('GPU/protocol mismatch')
    if method=='dynamics128' and m['policy_sha256']!=cfg['checkpoint_sha256']:raise ValueError('policy mismatch')
    if method=='seacache' and m['threshold']!=cfg['threshold']:raise ValueError('threshold mismatch')
    components=read(d/'components.json')['rows']
    if len(components)!=len(ids) or {r['sample_id'] for r in components}!=set(ids):raise ValueError('component coverage mismatch')
    for r in components:
        sid=r['sample_id'];t=read(d/'timings'/f'{sid}.json');trace=read(d/'traces'/f'{sid}.json')
        if t['status']!='success' or len(t['calls'])!=100 or t['pipeline_generate_wall_seconds']!=r['generate_seconds']:
            raise ValueError('timing incomplete')
        if method=='dynamics128':
            if trace['step_reuse']!=cfg['k'] or trace['total_steps']!=50:raise ValueError('Exact-K mismatch')
        elif trace['threshold']!=cfg['threshold'] or trace['total_branch_calls']!=100:raise ValueError('Sea trace mismatch')
    return components


def execute_suite(cfg):
    previous=Path(cfg['previous']);state=dict(status='waiting_for_gpus',stages={},vbench_status='skipped_by_user');lock=threading.Lock()
    env=dict(os.environ,OURS4WAN21_WORKSPACE=str(WORKSPACE),OURS4WAN21_EXP_BASE=str(EXP),EXP_BASE=str(EXP),
        CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',TORCH_HOME=str(prior.MODEL_ROOT/'torch-cache'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    def update(**kw):
        with lock:state.update(kw,updated_at=now());dump(ROOT/'status.json',state)
    def execute(name,cmd,g):
        with lock:state['stages'][name]=dict(status='running',gpu=g,started_at=now());dump(ROOT/'status.json',state)
        with (ROOT/'logs'/f'{name}.log').open('ab') as log:
            subprocess.run(cmd,cwd=PROJECT,env=dict(env,CUDA_VISIBLE_DEVICES=cfg['gpu_uuids'][str(g)]),stdout=log,stderr=subprocess.STDOUT,check=True)
        with lock:state['stages'][name].update(status='complete',completed_at=now());dump(ROOT/'status.json',state)
    def parallel(fn):
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fn,prior.GPUS))
    def directory(phase,g,method):return ROOT/('calibration' if phase=='probe' else 'shards')/f'gpu{g}'/method
    def generate(phase,g):
        parent=directory(phase,g,'seacache').parent;parent.mkdir(exist_ok=True)
        (parent/'README.md').write_text('# GPU shard\n\nseacache/ and dynamics128/ paired conditions, old native reference unchanged.\n')
        prompts=ROOT/'prompts'/f'{phase}_gpu{g}.jsonl'
        ids=cfg['probe_ids' if phase=='probe' else 'shard_ids'][str(g)]
        for method in METHODS:
            d=directory(phase,g,method)
            if not d.exists():
                if method=='seacache':cmd=[sys.executable,str(HERE/'sea_worker.py'),'--config',str(ROOT/'config.json'),'--prompts',str(prompts),
                    '--output-dir',str(d),'--result-parent',str(ROOT),'--gpu',str(g)]
                else:cmd=[sys.executable,'generate.py','--wan21-root',str(WORKSPACE/'data/source/Wan2.1-65386b2'),
                    '--checkpoint-dir',str(prior.MODEL_ROOT/'Wan2.1-T2V-1.3B'),'--prompts',str(prompts),'--flops-profile',cfg['flops_profile'],
                    '--policy-checkpoint',cfg['checkpoint'],'--state-mode',prior.MODE,'--skip-budget',str(cfg['k']),
                    '--output-dir',str(d),'--result-parent',str(ROOT)]
                execute(f'{phase}_gpu{g}_{method}',cmd,g)
            check_generation(d,g,method,ids,cfg)
    def quality(g):
        for method in METHODS:
            d=directory('formal',g,method);q=d/'quality'
            if not q.exists():execute(f'quality_gpu{g}_{method}',[sys.executable,str(OFFICIAL/'VideoMetrics/evaluate.py'),
                '--reference-dir',str(ROOT/'references'/f'gpu{g}'/'videos'),'--candidate-dir',str(d/'videos'),
                '--expected-frames','81','--device','cuda:0','--output-dir',str(q)],g)
            prior.quality_rows(q,cfg['shard_ids'][str(g)])
    try:
        update(status='waiting_for_gpus')
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():time.sleep(30)
        update(status='calibrating');parallel(lambda g:generate('probe',g))
        probes=[]
        for method in METHODS:
            base=[];cand=[]
            for g in prior.GPUS:
                ids=cfg['probe_ids'][str(g)]
                base += [r for r in read(previous/'shards'/f'gpu{g}'/'baseline/components.json')['rows'] if r['sample_id'] in ids]
                cand += check_generation(directory('probe',g,method),g,method,ids,cfg)
            if len(base)!=4 or len(cand)!=4:raise ValueError('calibration coverage mismatch')
            speed=sum(r['generate_seconds'] for r in base)/sum(r['generate_seconds'] for r in cand)
            probes.append(dict(method=method,prompts=4,measured_speedup=speed,mean_seconds=statistics.fmean(r['generate_seconds'] for r in cand)))
        dump(ROOT/'calibration/PROBE_RESULTS.json',dict(results=probes,target=3.6,k=cfg['k'],threshold=cfg['threshold']))
        for r in probes:validate_speed(r['measured_speedup'])
        dump(ROOT/'calibration/CALIBRATED.json',dict(status='calibrated',target=3.6,k=cfg['k'],threshold=cfg['threshold'],
            results=probes,config_sha256=sha(ROOT/'config.json'),probe_results_sha256=sha(ROOT/'calibration/PROBE_RESULTS.json'),
            fixed_for_all_formal_prompts=True,quality_used_for_selection=False))
        update(status='generating');parallel(lambda g:generate('formal',g))
        update(status='video_quality');parallel(quality)
        update(status='analyzing');finalize(cfg)
        update(status='complete')
    except BaseException as e:
        update(status='failed',error=repr(e));dump(ROOT/'FAILED.json',dict(error=repr(e),failed_at=now()));raise


def finalize(cfg):
    results=[];details=[];evidence={}
    for method in METHODS:
        base=[];cand=[];qs=[]
        for g in prior.GPUS:
            ids=cfg['shard_ids'][str(g)];d=ROOT/'shards'/f'gpu{g}'/method
            b=read(Path(cfg['previous'])/'shards'/f'gpu{g}'/'baseline/components.json')['rows']
            c=check_generation(d,g,method,ids,cfg);quality=prior.quality_rows(d/'quality',ids)
            base+=b;cand+=[dict(r,**{k:0.0 for k in prior.FIELDS[7:] if k not in r}) for r in c];qs+=quality
            byb={r['sample_id']:r for r in b};byc={r['sample_id']:r for r in c}
            for q in quality:
                sid=q['video_id'];details.append(dict(method=method,sample_id=sid,gpu=g,
                    baseline_seconds=byb[sid]['generate_seconds'],candidate_seconds=byc[sid]['generate_seconds'],
                    **{m:float(q[m+'_mean']) for m in prior.METRICS}))
            for name in ('components.json','run.json','COMPLETE.json','quality/summary.json','quality/per_video.csv'):
                evidence[str((d/name).relative_to(ROOT))]=sha(d/name)
        result=dict(method=method,target=3.6,k=cfg['k'] if method=='dynamics128' else '',
            threshold=cfg['threshold'] if method=='seacache' else '',checkpoint_epoch=391 if method=='dynamics128' else '',
            **prior.summarize(base,cand))
        for m in prior.METRICS:
            values=[float(q[m+'_mean']) for q in qs];result[m]=statistics.fmean(values);result[m+'_std_across_prompts']=statistics.stdev(values)
        results.append(result)
    if len(details)!=100 or len({(r['method'],r['sample_id']) for r in details})!=100:raise ValueError('final coverage mismatch')
    for path,digest in cfg['baseline_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('original baseline changed')
    out=ROOT/'analysis';prior.write_csv(out/'results.csv',results);prior.write_csv(out/'per_video.csv',details)
    dump(out/'VALIDATION.json',dict(status='pass',formal_candidates=100,baseline_reused=50,quality_frames=8100,calibration_candidates=8,
        config_sha256=sha(ROOT/'config.json'),calibration_sha256=sha(ROOT/'calibration/CALIBRATED.json'),evidence_sha256=evidence,vbench_status='skipped_by_user'))
    report=['# 同50prompt名义3.6×：SeaCache / Dynamics128 e391','',
        '复用既有同卡native baseline；50条prompt、seed42。只评测PSNR/SSIM/LPIPS，无VBench分数。',
        '标定4条/方法仅用速度，正式固定阈值/K，无自适应补测。3.6×为名义目标，实际速度单列。',
        '推理计时排除加载/warmup/保存/评测；质量逐视频81帧均值再对50prompt等权平均。','',
        '|方法|K或阈值|实测加速|秒/视频|DiT TFLOPs|PSNR|SSIM|LPIPS|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        setting=str(r['k']) if r['k']!='' else f"{r['threshold']:.7f}"
        report.append(f"|{r['method']}|{setting}|{r['latency_speedup']:.4f}|{r['candidate_generate_seconds_mean']:.3f}|{r['candidate_dit_tflops_mean']:.3f}|{r['psnr_rgb_db']:.4f}|{r['ssim_rgb']:.5f}|{r['lpips_alex_v0_1_spatial']:.5f}|")
    (out/'RESULTS.md').write_text('\n'.join(report)+'\n')
    dump(ROOT/'COMPLETE.json',dict(status='complete',completed_at=now(),candidates=100,baseline_reused=50,
        vbench_status='skipped_by_user',results_sha256=sha(out/'results.csv'),validation_sha256=sha(out/'VALIDATION.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if 'wan2.2' not in Path(sys.prefix).name.lower():raise ValueError('use Wan2.2 environment')
    cfg=prepare(a.resume)
    if a.prepare_only:
        print(json.dumps(dict(status='prepared',root=str(ROOT),k=cfg['k'],threshold=cfg['threshold'],probe_ids=cfg['probe_ids'])));return
    with (ROOT/'queue.lock').open('a') as local,(EXP/'wan21_benchmark_4gpu.lock').open('a') as gpu:
        fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'COMPLETE.json').exists():return
        fcntl.flock(gpu,fcntl.LOCK_EX);execute_suite(cfg)


if __name__=='__main__':main()
