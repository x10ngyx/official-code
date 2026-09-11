"""Freeze 5x4x2 paired trajectories, match using measured calibration, validate and report."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
SEA=HERE.parents[1];OFFICIAL=SEA.parent
sys.path.insert(0,str(OFFICIAL/'Ours4Wan21/experiments/vbench50_speed36_pair_v1'))
import run_pair as pair
from schedule import ENDPOINTS,linear_path
ROOT=pair.EXP/'seacache_wan21_linear_increase_matched_5x4_v1'
read,dump,sha=pair.read,pair.dump,pair.sha
FIELDS=pair.prior.FIELDS[:7]
METRICS=pair.prior.METRICS

def mkdir(path,description):
    path.mkdir(parents=True,exist_ok=True);(path/'README.md').write_text(description+'\n')

def prepare():
    if (ROOT/'config.json').exists():
        cfg=read(ROOT/'config.json');validate_sources(cfg);return cfg
    old=read(pair.ROOT/'config.json');previous=Path(old['previous'])
    rows=[json.loads(s) for s in (previous/'prompts/selected.jsonl').read_text().splitlines()]
    # Reuse previously frozen speed-probe IDs, chosen without quality selection.
    prompts={g:next(r for r in rows if r['sample_id']==old['probe_ids'][g][0]) for g in map(str,range(4))}
    mapping=pair.EXP/'wan21_seacache_speedup_calibration_v1/analysis/speed_threshold_mapping.calibrated.json'
    cal=Path(read(mapping)['fit_source'])
    if sha(cal)!=read(mapping)['fit_source_sha256']:raise ValueError('calibration source changed')
    sources=[*HERE.glob('*.py'),SEA/'seacache.py',SEA/'wan21_integration.py',SEA/'inference_timing.py',mapping,cal,Path(old['flops_profile']),pair.ROOT/'config.json']
    baselines={}
    for g,r in prompts.items():
        base=previous/'shards'/f'gpu{g}'/'baseline';sid=r['sample_id']
        run=read(base/'run.json')
        if run['protocol']!=old['protocol'] or run['gpu_uuid']!=old['gpu_uuids'][g]:raise ValueError('reference protocol/GPU mismatch')
        baselines[g]=next(x for x in read(base/'components.json')['rows'] if x['sample_id']==sid)
        sources += [base/'components.json',base/'run.json',base/'videos'/f'{sid}.mp4',base/'timings'/f'{sid}.json']
    cfg=dict(schema='seacache_linear_matched_v1',protocol=old['protocol'],previous=str(previous),prompts=prompts,baselines=baselines,
        gpu_uuids=old['gpu_uuids'],flops_profile=old['flops_profile'],calibration_csv=str(cal),
        strategies=[dict(start=a,end=b,path=linear_path(a,b)) for a,b in ENDPOINTS],formal_trajectories=40,
        selection='same four frozen prior speed-probe prompts, no quality selection',
        matching='increasing measured sum baseline / sum candidate latency; interpolate adjacent measured calibration points; fixed once for four prompts; no extra inference',
        speed_match_tolerance=.05,vbench_enabled=True,vbench_scope='custom-input ten-dimension raw mean, not official VBench leaderboard score',
        source_sha256={str(p):sha(p) for p in sources})
    mkdir(ROOT,'# Linear increasing versus calibrated fixed SeaCache\n\n40 formal trajectories, 4 reused same-GPU baselines; config.json, matching.json, conditions/, references/, analysis/, logs/. Four native warmups excluded from measurements.')
    for p in (SEA,OFFICIAL/'Ours4Wan21'):
        link=p/'experiment_results'/ROOT.name
        if not link.is_symlink():link.symlink_to(ROOT,target_is_directory=True)
    for name in ('conditions','references','analysis','logs'):mkdir(ROOT/name,f'# {name}\n\nSee ../README.md.')
    for g,r in prompts.items():
        ref=ROOT/'references'/f'gpu{g}';mkdir(ref,'# Same-GPU native reference\n\nReused baseline video symlink.');(ref/'videos').mkdir()
        (ref/'videos'/f"{r['sample_id']}.mp4").symlink_to(previous/'shards'/f'gpu{g}'/'baseline/videos'/f"{r['sample_id']}.mp4")
    dump(ROOT/'config.json',cfg);validate_sources(cfg);return cfg

def validate_sources(cfg):
    for p,h in cfg['source_sha256'].items():
        if sha(Path(p))!=h:raise ValueError(f'frozen input changed: {p}')

def condition(cfg,phase,i,g):
    d=ROOT/'conditions'/f'{phase}_{i+1}'/f'gpu{g}';sid=cfg['prompts'][str(g)]['sample_id']
    done=read(d/'COMPLETE.json');run=read(d/'run.json');t=read(d/'timings'/f'{sid}.json');trace=read(d/'traces'/f'{sid}.json');r=read(d/'components.json')['rows'][0]
    path=cfg['strategies'][i]['path'] if phase=='increase' else [read(ROOT/'matching.json')['pairs'][i]['threshold']]*50
    if run['protocol']!=cfg['protocol'] or run['gpu_uuid']!=cfg['gpu_uuids'][str(g)] or run['prompt']!=cfg['prompts'][str(g)] or run['threshold_path']!=path or trace['threshold_path']!=path:raise ValueError('condition identity mismatch')
    if done['video_sha256']!=sha(d/'videos'/f'{sid}.mp4'):raise ValueError('video SHA mismatch')
    if t['status']!='success' or len(t['calls'])!=100 or len(trace['decisions'])!=100 or r['sample_id']!=sid or t['pipeline_generate_wall_seconds']!=r['generate_seconds']:raise ValueError('incomplete trace/timing')
    for j,(decision,call) in enumerate(zip(trace['decisions'],t['calls'])):
        if decision['step_index']!=j//2 or decision['branch']!=('cond' if j%2==0 else 'uncond') or decision['requested_threshold']!=path[j//2] or decision['execution']!=decision['action']:raise ValueError('decision mismatch')
        if call['blocks_executed']!=(0 if decision['action']=='reuse' else 30):raise ValueError('trace versus executed blocks mismatch')
    if any(not math.isfinite(r[k]) or r[k]<=0 for k in FIELDS):raise ValueError('invalid components')
    return r,trace

def match(cfg):
    rows=[]
    for i in range(5):
        candidates=[condition(cfg,'increase',i,g)[0] for g in range(4)]
        speed=sum(r['generate_seconds'] for r in cfg['baselines'].values())/sum(r['generate_seconds'] for r in candidates)
        threshold,bracket=pair.interpolate_threshold(pair.read_csv(Path(cfg['calibration_csv'])),speed)
        rows.append(dict(strategy=i+1,increase_speedup=speed,threshold=threshold,calibration_bracket=bracket))
    dump(ROOT/'matching.json',dict(pairs=rows,quality_used=False,config_sha256=sha(ROOT/'config.json')))

def finalize(cfg):
    results=[];details=[];evidence={}
    for i in range(5):
        for phase in ('increase','fixed'):
            label=f'{phase}_{i+1}';candidates=[];qualities=[];traces=[]
            for g in range(4):
                r,tr=condition(cfg,phase,i,g);candidates.append(r);traces.append(tr)
                d=ROOT/'conditions'/label/f'gpu{g}';sid=r['sample_id']
                q=pair.prior.quality_rows(d/'quality',[sid])[0];qualities.append(q)
                details.append(dict(strategy=i+1,phase=phase,gpu=g,**r,baseline_seconds=cfg['baselines'][str(g)]['generate_seconds'],reuse_steps=tr['reuse']/2,**{m:float(q[m+'_mean']) for m in METRICS}))
                for f in ('components.json','run.json','COMPLETE.json','quality/summary.json','quality/per_video.csv',f'timings/{sid}.json',f'traces/{sid}.json'):
                    evidence[str((d/f).relative_to(ROOT))]=sha(d/f)
            vpath=ROOT/'conditions'/label/'vbench/vbench_custom_aggregate_scores.json';v=read(vpath)
            if v['official_full_vbench_score'] is not False or len(v['raw_dimension_scores'])!=10 or not math.isclose(st.fmean(v['raw_dimension_scores'].values()),v['vbench_score']):raise ValueError('VBench scope/mean mismatch')
            evidence[str(vpath.relative_to(ROOT))]=sha(vpath)
            b=list(cfg['baselines'].values())
            r=dict(strategy=i+1,phase=phase,threshold_start=cfg['strategies'][i]['start'] if phase=='increase' else read(ROOT/'matching.json')['pairs'][i]['threshold'],threshold_end=cfg['strategies'][i]['end'] if phase=='increase' else read(ROOT/'matching.json')['pairs'][i]['threshold'],
                latency_speedup=sum(x['generate_seconds'] for x in b)/sum(x['generate_seconds'] for x in candidates),
                dit_compute_ratio=sum(x['dit_tflops'] for x in b)/sum(x['dit_tflops'] for x in candidates),
                **{k:st.fmean(x[k] for x in candidates) for k in FIELDS},reuse_steps=st.fmean(x['reuse']/2 for x in traces),
                **{m:st.fmean(float(x[m+'_mean']) for x in qualities) for m in METRICS},vbench_score=v['vbench_score'])
            results.append(r)
    comparisons=[]
    for inc,fix in zip(results[::2],results[1::2]):
        gap=fix['generate_seconds']/inc['generate_seconds']-1
        comparisons.append(dict(strategy=inc['strategy'],increase_speedup=inc['latency_speedup'],fixed_speedup=fix['latency_speedup'],fixed_latency_relative_gap=gap,speed_matched_within_5pct=abs(gap)<=cfg['speed_match_tolerance'],**{m+'_increase_minus_fixed':inc[m]-fix[m] for m in (*METRICS,'vbench_score')}))
    if len(details)!=40 or len({(r['strategy'],r['phase'],r['sample_id']) for r in details})!=40:raise ValueError('40-pair coverage mismatch')
    validate_sources(cfg)
    for name,rows in [('results',results),('per_video',details),('comparison',comparisons)]:pair.prior.write_csv(ROOT/'analysis'/f'{name}.csv',rows)
    dump(ROOT/'analysis/VALIDATION.json',dict(status='pass_with_small_sample_and_matching_caveats',formal_trajectories=40,reused_baselines=4,branch_calls=4000,quality_frame_pairs=3240,evidence_sha256=evidence,speed_match_tolerance=.05,matched_pairs=sum(r['speed_matched_within_5pct'] for r in comparisons),vbench_scope=cfg['vbench_scope']))
    lines=['# SeaCache 线性递增阈值与标定固定阈值对照','',
        '同4条prompt、seed42、同物理GPU baseline；5组×4×2=40条正式轨迹。',
        '固定阈值只由递增组实测总耗时加速和已有标定相邻点插值确定，未用质量调参。',
        '加速比=4条baseline总时间/候选总时间；质量为逐视频81帧均值后对prompt等权平均。',
        '每组仅4条prompt、单seed；实测耗时差超过5%的组不能视为等速质量结论。VBench score为custom十维原始均值，不是官方总分。','',
        '|策略|方法|threshold|实测加速|秒/视频|DiT TFLOPs|PSNR|SSIM|LPIPS|VBench custom|',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"|{r['strategy']}|{r['phase']}|{r['threshold_start']:.5f}→{r['threshold_end']:.5f}|{r['latency_speedup']:.4f}|{r['generate_seconds']:.3f}|{r['dit_tflops']:.2f}|{r['psnr_rgb_db']:.4f}|{r['ssim_rgb']:.5f}|{r['lpips_alex_v0_1_spatial']:.5f}|{r['vbench_score']:.5f}|")
    lines+=['','|策略|固定相对递增耗时差|PSNR差（递增−固定）|±5%匹配|','|---:|---:|---:|---|']
    for r in comparisons:lines.append(f"|{r['strategy']}|{r['fixed_latency_relative_gap']:+.2%}|{r['psnr_rgb_db_increase_minus_fixed']:+.4f}|{r['speed_matched_within_5pct']}|")
    (ROOT/'analysis/RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(ROOT/'COMPLETE.json',dict(status='complete',formal_trajectories=40,results_sha256=sha(ROOT/'analysis/results.csv'),validation_sha256=sha(ROOT/'analysis/VALIDATION.json')))

def execute(cfg):
    env=dict(os.environ,OURS4WAN21_WORKSPACE=str(pair.WORKSPACE),OURS4WAN21_EXP_BASE=str(pair.EXP),EXP_BASE=str(pair.EXP),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',TORCH_HOME=str(pair.prior.MODEL_ROOT/'torch-cache'),PYTHON_BIN=sys.executable,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    def status(s):dump(ROOT/'status.json',dict(status=s,updated_at=pair.now()))
    def command(cmd,g,log):
        with (ROOT/'logs'/f'{log}.log').open('ab') as f:subprocess.run(cmd,cwd=OFFICIAL,env=dict(env,CUDA_VISIBLE_DEVICES=cfg['gpu_uuids'][str(g)]),stdout=f,stderr=subprocess.STDOUT,check=True)
    processes=[];logs=[]
    try:
        status('waiting_for_gpus')
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():time.sleep(10)
        validate_sources(cfg);status('generating_increase')
        for g in range(4):
            f=(ROOT/'logs'/f'worker_gpu{g}.log').open('ab');logs.append(f)
            processes.append(subprocess.Popen([sys.executable,str(HERE/'worker.py'),'--root',str(ROOT),'--gpu',str(g)],cwd=OFFICIAL,env=dict(env,CUDA_VISIBLE_DEVICES=cfg['gpu_uuids'][str(g)]),stdout=f,stderr=subprocess.STDOUT))
        while not all((ROOT/'conditions'/f'increase_{i+1}'/f'gpu{g}'/'COMPLETE.json').exists() for i in range(5) for g in range(4)):
            if any(p.poll() is not None and p.returncode!=0 for p in processes):raise RuntimeError('generation worker failed')
            time.sleep(5)
        if not (ROOT/'matching.json').exists():match(cfg)
        status('generating_fixed')
        for p in processes:
            if p.wait()!=0:raise RuntimeError('fixed generation failed')
        status('video_quality')
        def quality(g):
            for i in range(5):
                for phase in ('increase','fixed'):
                    d=ROOT/'conditions'/f'{phase}_{i+1}'/f'gpu{g}'
                    if not (d/'quality/summary.json').exists():command([sys.executable,str(OFFICIAL/'VideoMetrics/evaluate.py'),'--reference-dir',str(ROOT/'references'/f'gpu{g}'/'videos'),'--candidate-dir',str(d/'videos'),'--expected-frames','81','--device','cuda:0','--output-dir',str(d/'quality')],g,f'quality_{phase}_{i+1}_gpu{g}')
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(quality,range(4)))
        status('vbench_custom')
        def vbench(job):
            phase,i=job;label=f'{phase}_{i+1}';d=ROOT/'conditions'/label
            mkdir(d,'# Condition\n\nFour GPU trajectories, videos/ symlink view and VBench custom metrics.')
            videos=d/'videos';videos.mkdir(exist_ok=True)
            for g in range(4):
                sid=cfg['prompts'][str(g)]['sample_id'];link=videos/f'{sid}.mp4'
                if not link.is_symlink():link.symlink_to(d/f'gpu{g}'/'videos'/f'{sid}.mp4')
            dump(d/'prompt_map.json',{f"{r['sample_id']}.mp4":r['prompt_en'] for r in cfg['prompts'].values()})
            if not (d/'vbench/vbench_custom_aggregate_scores.json').exists():command(['bash',str(OFFICIAL/'VbenchEvaluation/run_custom_vbench.sh'),str(videos),str(d/'vbench'),str(d/'prompt_map.json')],i%4,f'vbench_{label}')
        # Static queues ensure no two evaluators share a GPU.
        def vqueue(g):
            for i in range(5):
                if i%4==g:
                    for phase in ('increase','fixed'):vbench((phase,i))
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(vqueue,range(4)))
        status('analyzing');finalize(cfg);status('complete')
    except BaseException as exc:
        dump(ROOT/'FAILED.json',dict(error=repr(exc),at=pair.now()));status('failed')
        for p in processes:
            if p.poll() is None:p.terminate()
        raise
    finally:
        for f in logs:f.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--finalize-only',action='store_true');a=p.parse_args()
    cfg=prepare()
    if a.prepare_only:print(json.dumps(dict(root=str(ROOT),prompts={g:r['sample_id'] for g,r in cfg['prompts'].items()})));return
    if a.finalize_only:finalize(cfg);return
    with (ROOT/'queue.lock').open('a') as local,(pair.EXP/'wan21_benchmark_4gpu.lock').open('a') as shared:
        fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'COMPLETE.json').exists():return
        fcntl.flock(shared,fcntl.LOCK_EX);execute(cfg)
if __name__=='__main__':main()
