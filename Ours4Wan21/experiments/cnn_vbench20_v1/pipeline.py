import argparse,csv,fcntl,importlib.util,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from common import *

def directory(p,desc):
    p.mkdir(parents=True,exist_ok=True)
    if not (p/'README.md').exists():(p/'README.md').write_text(desc+'\n')
    return p

def prepare():
    reference=load_evaluation_bundle(REFERENCE);training=read(TRAIN/'config.json')
    assert training['epochs']==200 and training['selection_start']==170
    old=read(EXP_ROOT/'ours21_increase500_iql2_v1/config.json')
    files=[*sorted(HERE.glob('*.py')),FORMAL/'model.py',FORMAL/'features.py',TRAIN/'config.json',REFERENCE/'manifest.json',
           *sorted((PROJECT/'ours4wan21').glob('*.py')),HERE.parent/'iql_aggressiveness_2x4_v1/common.py',HERE.parent/'increase500_yuv_psnr_v1/run.py',Path(old['paths']['flops_profile'])]
    cfg=dict(schema='ours21_cnn_vbench20_v1',protocol=PROTOCOL,groups=['G1','G2','G3','G4'],prompts=reference['rows'],
             gpu_uuids=training['gpu_uuids'],targets=TARGETS,skip_budgets=BUDGETS,training_root=str(TRAIN),
             vbench_enabled=False,reference=str(REFERENCE),paths=old['paths'],inputs=old['inputs'],source_hashes={str(p):sha256(p) for p in files})
    directory(ROOT,(HERE/'README.md').read_text())
    for name in ('logs','jobs','evaluation','evaluation/candidates','evaluation/quality_inputs','evaluation/quality','yuv','yuv/frames'):directory(ROOT/name,'# '+name+'\n\nSee root README.')
    if (ROOT/'config.json').exists():assert read(ROOT/'config.json')==cfg
    else:dump(ROOT/'config.json',cfg)
    link=PROJECT/'experiment_results'/ROOT.name
    if not link.exists():link.symlink_to(ROOT)
    dump(ROOT/'STATUS.json',dict(stage='waiting_for_cnn_selection',training_root=str(TRAIN)))

def selected():
    result={}
    for g in ['G1','G2','G3','G4']:
        done=TRAIN/'training'/g/'TRAINING_COMPLETE.json';pick=TRAIN/'analysis'/g/'COMPLETE.json'
        if not done.exists() or not pick.exists():return None
        d=read(done);p=read(pick)
        assert d['epochs']==200 and d['status']==p['status']=='complete' and 170<p['selected_epoch']<200
        assert sha256(p['checkpoint'])==p['sha256']
        result[g]=p
    return result

def make_jobs(cfg,picks):
    queues={i:[] for i in range(4)}
    for g in cfg['groups']:
        p=picks[g]
        for k in BUDGETS:
            for row in cfg['prompts']:
                gpu=cfg['gpu_uuids'].index(row['baseline_gpu_uuid']);out=ROOT/'evaluation/candidates'/g/f'K{k}'/row['sample_id']
                directory(out.parent,'# Candidates at fixed K; each video has a sealed restart boundary.')
                queues[gpu].append(dict(kind='candidate',group=g,state_mode='cnn_'+g,sample_id=row['sample_id'],prompt=row['prompt'],skip_budget=k,expected_gpu_uuid=row['baseline_gpu_uuid'],checkpoint=dict(path=p['checkpoint'],sha256=p['sha256']),output=str(out)))
    assert sum(map(len,queues.values()))==240
    for gpu,jobs in queues.items():dump(ROOT/'jobs'/f'gpu{gpu}.json',jobs)
    dump(ROOT/'SELECTED.json',picks);return queues

def execute(cfg,label,args,gpu):
    verify_sources(cfg)
    with (ROOT/'logs'/f'{label}.log').open('a') as f:subprocess.run(args,cwd=PROJECT,env=environment(cfg['gpu_uuids'][gpu]),stdout=f,stderr=subprocess.STDOUT,check=True)

def quality(cfg,gpu,jobs):
    from ours4wan21.online_common import verified
    from generate_worker import job_identity
    d=directory(ROOT/'evaluation/quality_inputs'/f'gpu{gpu}','# Native/candidate video pairs')
    for side in ('reference','candidate'):directory(d/side,'# Source video links')
    for j in jobs:
        assert verified(j['output'],job_identity(j,cfg));name=f"{j['group']}_K{j['skip_budget']}_{j['sample_id']}"
        for side,target in [('reference',REFERENCE/'baselines'/j['sample_id']/'video.mp4'),('candidate',Path(j['output'])/'video.mp4')]:
            link=d/side/(name+'.mp4')
            if not link.exists():link.symlink_to(target)
            assert link.resolve()==target.resolve()
    out=ROOT/'evaluation/quality'/f'gpu{gpu}'
    if (out/'summary.json').exists():assert read(out/'summary.json')['video_count']==len(jobs);return
    execute(cfg,f'quality_{gpu}',[sys.executable,str(OFFICIAL/'VideoMetrics/evaluate.py'),'--reference-dir',str(d/'reference'),'--candidate-dir',str(d/'candidate'),'--expected-frames','81','--device','cuda:0','--model-cache',str(MODEL_ROOT/'torch-cache'),'--output-dir',str(out)],gpu)

def yuv_quality(gpu):
    # Use the exact established YUV611 implementation and verify RGB reproduction.
    p=HERE.parent/'increase500_yuv_psnr_v1/run.py';spec=importlib.util.spec_from_file_location('cnn_yuv',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=ROOT/'yuv';m.checks();jobs=read(ROOT/'jobs'/f'gpu{gpu}.json');out=ROOT/'yuv'/f'gpu{gpu}.json'
    if out.exists():assert len(read(out))==len(jobs);return
    qs={r['video_id']:r for r in csv.DictReader((ROOT/'evaluation/quality'/f'gpu{gpu}/per_video.csv').open())};by_prompt={}
    for j in jobs:
        q=qs[f"{j['group']}_K{j['skip_budget']}_{j['sample_id']}"]
        row=dict(method=j['group'],setting=f"K{j['skip_budget']}",sample_id=j['sample_id'],reference=q['reference'],candidate=q['candidate'],original_rgb_psnr=q['psnr_rgb_db_mean'])
        by_prompt.setdefault(j['sample_id'],[]).append(row)
    results=[]
    for rows in by_prompt.values():results.extend(m.worker(rows))
    dump(out,results)

def report(cfg,queues):
    from ours4wan21.online_common import verified
    from generate_worker import job_identity
    assert load_evaluation_bundle(REFERENCE)['rows']==cfg['prompts'];picks=read(ROOT/'SELECTED.json');assert picks==selected();detail=[]
    for gpu,jobs in queues.items():
        folder=ROOT/'evaluation/quality'/f'gpu{gpu}';s=read(folder/'summary.json');assert s['video_count']==len(jobs) and s['frame_count_total']==81*len(jobs)
        qs={r['video_id']:r for r in csv.DictReader((folder/'per_video.csv').open())};ys={(r['method'],r['setting'],r['sample_id']):r for r in read(ROOT/'yuv'/f'gpu{gpu}.json')}
        for j in jobs:
            out=Path(j['output']);assert verified(out,job_identity(j,cfg));gen=read(out/'generation.json');c=read(out/'measurement.json');t=read(out/'trace.json')
            assert gen['gpu_uuid']==j['expected_gpu_uuid']==cfg['gpu_uuids'][gpu] and gen['protocol']==PROTOCOL
            assert t['cnn_group']==j['group'] and t['policy_family']=='three_branch_cnn'
            assert j['checkpoint']==dict(path=picks[j['group']]['checkpoint'],sha256=picks[j['group']]['sha256'])
            q=qs[f"{j['group']}_K{j['skip_budget']}_{j['sample_id']}"]
            assert (int(q['frames']),int(q['width']),int(q['height']))==(81,832,480)
            for side in ('reference','candidate'):assert sha256(q[side])==q[side+'_sha256']
            ref=REFERENCE/'baselines'/j['sample_id'];assert Path(q['reference']).resolve()==(ref/'video.mp4').resolve() and Path(q['candidate']).resolve()==(out/'video.mp4').resolve()
            trace=audit_trace(t,read(out/'timing.json'),j['skip_budget']);assert all(math.isfinite(c[k]) and c[k]>=0 for k in FIELDS)
            assert all(math.isfinite(float(q[k+'_mean'])) for k in METRICS)
            y=ys[j['group'],f"K{j['skip_budget']}",j['sample_id']];assert abs(y['rgb_psnr']-float(q['psnr_rgb_db_mean']))<1e-8
            detail.append(dict(group=j['group'],k=j['skip_budget'],sample_id=j['sample_id'],checkpoint_epoch=picks[j['group']]['selected_epoch'],baseline_seconds=read(ref/'measurement.json')['generate_seconds'],**{k:c[k] for k in FIELDS},**{k:float(q[k+'_mean']) for k in METRICS},yuv_611_db=y['yuv_611_db'],**trace))
    assert len(detail)==240;results=[]
    for g in cfg['groups']:
        for k in BUDGETS:
            rr=[r for r in detail if r['group']==g and r['k']==k];assert len(rr)==20
            results.append(dict(group=g,k=k,checkpoint_epoch=rr[0]['checkpoint_epoch'],speedup=sum(r['baseline_seconds'] for r in rr)/sum(r['generate_seconds'] for r in rr),**{key:sum(r[key] for r in rr)/20 for key in (*FIELDS,*METRICS,'yuv_611_db')}))
    writecsv(ROOT/'per_video.csv',detail);writecsv(ROOT/'results.csv',results)
    lines=['# Four CNN groups: same VBench20 prompts','','Selected after 200 training epochs; K23/K29/K35, fixed seed42. 240 candidates, 20 reused same-GPU baselines. No VBench score.','', '| Group | Epoch | K | Actual speedup | RGB PSNR | YUV611 | SSIM | LPIPS |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['group']} | {r['checkpoint_epoch']} | {r['k']} | {r['speedup']:.4f} | {r[METRICS[0]]:.4f} | {r['yuv_611_db']:.4f} | {r[METRICS[1]]:.5f} | {r[METRICS[2]]:.5f} |")
    lines+=['','Full generate latency includes selected feature construction and CNN decisions. Component times and TFLOPs are in per_video.csv/results.csv. CNN operation counts are Conv/Linear estimates only; FFT/statistics FLOPs are not counted.','', 'Same fixed20 set has been used previously; single-seed descriptive comparison, not a fresh held-out generalization claim. YUV611 uses BT.709 full-range channel-dB weighting (6Y+U+V)/8, per-frame then per-video.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(ROOT/'COMPLETE.json',dict(status='complete',groups=4,candidates=240,paired_frames=19440,baseline_reused=20,vbench_status='skipped_per_established_scope',config_sha256=sha256(ROOT/'config.json'),results_sha256=sha256(ROOT/'results.csv'),detail_sha256=sha256(ROOT/'per_video.csv')))

def run():
    cfg=read(ROOT/'config.json');verify_sources(cfg)
    with (ROOT/'pipeline.lock').open('a') as local:
        fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'COMPLETE.json').exists():return
        try:
            dump(ROOT/'STATUS.json',dict(stage='waiting_for_cnn_selection',pid=os.getpid(),training_root=str(TRAIN)))
            while selected() is None:
                if (TRAIN/'FAILED.json').exists():raise RuntimeError('training pipeline failed')
                time.sleep(15)
            with (EXP_ROOT/'wan21_benchmark_4gpu.lock').open('a') as shared:
                dump(ROOT/'STATUS.json',dict(stage='waiting_for_gpu_lock',pid=os.getpid()));fcntl.flock(shared,fcntl.LOCK_EX)
                verify_sources(cfg);queues=make_jobs(cfg,selected())
                dump(ROOT/'STATUS.json',dict(stage='generation',candidates=240,pid=os.getpid()))
                with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda i:execute(cfg,f'generate_{i}',[sys.executable,str(HERE/'generate_worker.py'),'--jobs',str(ROOT/'jobs'/f'gpu{i}.json')],i),range(4)))
                dump(ROOT/'STATUS.json',dict(stage='quality',pid=os.getpid()))
                with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda i:quality(cfg,i,queues[i]),range(4)))
            dump(ROOT/'STATUS.json',dict(stage='yuv_and_report',pid=os.getpid()))
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda i:execute(cfg,f'yuv_{i}',[sys.executable,str(HERE/'pipeline.py'),'yuv','--gpu',str(i)],i),range(4)))
            report(cfg,queues);dump(ROOT/'STATUS.json',dict(stage='complete'))
        except BaseException as e:
            dump(ROOT/'FAILED.json',dict(error=repr(e),traceback=traceback.format_exc()));dump(ROOT/'STATUS.json',dict(stage='failed',error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','yuv']);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='run':run()
    else:yuv_quality(a.gpu)
