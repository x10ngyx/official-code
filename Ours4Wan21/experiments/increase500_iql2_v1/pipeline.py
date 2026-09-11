"""Wait for increase500; mixed3500 features -> two IQL runs -> select -> same20 evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from common import *


def directory(p, description):
    p.mkdir(parents=True,exist_ok=True)
    if not (p/'README.md').exists():(p/'README.md').write_text(description+'\n')
    return p


def groups():
    return [dict(name=name,feature='dynamics128',level=name,mode=MODE,profile=profile,
        training_config=asdict(training_config(profile)),cache=str(ROOT/(ROOT.name+'_cache')),
        training=str(ROOT/name/(ROOT.name+'_'+name+'_train')),analysis=str(ROOT/name/(ROOT.name+'_'+name+'_analysis')),
        weights=str(MODEL_ROOT/(ROOT.name+'_'+name)))
        for name,profile in [('conservative','baseline'),('aggressive','aggressive_v1')]]


def prepare():
    reference=load_evaluation_bundle(REFERENCE)
    uuids=gpu_uuids(['0','1','2','3'])
    assert all(r['baseline_gpu_uuid'] in uuids for r in reference['rows'])
    assert len(reference['rows'])==20
    profile=EXP_ROOT/'wan21_seacache_threshold_collection_v1/calflops_profile.json'
    files=[*sorted(HERE.glob('*.py')), *sorted((PROJECT/'ours4wan21').glob('*.py')),
           PROJECT/'experiments/iql_aggressiveness_2x4_v1/common.py',
           COLLECTION/'manifests/plan.jsonl',COLLECTION/'manifests/increase_runnable.jsonl',
           FEATURE_SOURCE/'index.json',FEATURE_SOURCE/'COMPLETE.json',REFERENCE/'manifest.json',profile]
    cfg=dict(schema='ours21_increase500_iql2_v1',protocol=PROTOCOL,groups=groups(),
        prompts=reference['rows'],gpu_uuids=uuids,targets=TARGETS,skip_budgets=BUDGETS,
        vbench_enabled=False,reference=str(REFERENCE),collection=str(COLLECTION),
        selection_rule='existing two-sided e301-399 validation actor agreement .96/.95/max; minimize Q drift/IQR',
        source_hashes={str(p):sha256(p) for p in files},
        inputs={'flops_profile':dict(path=str(profile),sha256=sha256(profile))},
        paths=dict(wan21_root=str(WORKSPACE/'data/source/Wan2.1-65386b2'),
                   wan_checkpoint=str(MODEL_ROOT/'Wan2.1-T2V-1.3B'),flops_profile=str(profile)))
    directory(ROOT,(HERE/'README.md').read_text())
    if (ROOT/'config.json').exists():assert read(ROOT/'config.json')==cfg
    else:dump(ROOT/'config.json',cfg)
    for name in ('logs','jobs','evaluation','conservative','aggressive'):
        directory(ROOT/name,'# '+name+'\n\nSee suite README and frozen config.')
    link=PROJECT/'experiment_results'/ROOT.name
    if not link.exists():link.symlink_to(ROOT)
    assert link.resolve()==ROOT
    print('Prepared two groups, same frozen20, 120 candidate videos; no VBench score.',flush=True)


def execute(cfg,label,cmd,gpu=0):
    verify_sources(cfg)
    dump(ROOT/'jobs'/f'{label}_command.json',dict(command=cmd,gpu_uuid=cfg['gpu_uuids'][gpu]))
    with (ROOT/'logs'/f'{label}.log').open('a') as f:
        subprocess.run(cmd,cwd=PROJECT,env=environment(cfg['gpu_uuids'][gpu]),stdout=f,stderr=subprocess.STDOUT,check=True)


def dataset_ready():
    if not (COLLECTION/'COMPLETE.json').exists():return False
    done=read(COLLECTION/'DATA_COMPLETE.json')
    assert done['candidates']==500 and done['mixed_trajectories']==3500
    assert done['vbench_status']=='skipped_by_user'
    assert sha256(COLLECTION/'mixed/selection.json')==done['mixed_selection_sha256']
    from ours4wan21.selection import selected_paths
    paths=selected_paths(COLLECTION/'mixed/selection.json',COLLECTION/'mixed')
    assert len(paths)==3500
    return True


def features(cfg):
    out=ROOT/(ROOT.name+'_features');selection=COLLECTION/'mixed/selection.json'
    from ours4wan21.feature_cache import load_feature_index
    from ours4wan21.selection import selected_paths
    paths=selected_paths(selection,COLLECTION/'mixed')
    if (out/'COMPLETE.json').exists():
        idx=load_feature_index(out);assert idx['selection_sha256']==sha256(selection) and len(idx['rows'])==3500
        return
    base=[sys.executable,'prepare_features.py','--collection-root',str(COLLECTION/'mixed'),
          '--selection',str(selection),'--output-dir',str(out),'--num-workers','4']
    if not out.exists():execute(cfg,'features_initialize',base+['--initialize-only'])
    old=load_feature_index(FEATURE_SOURCE)
    reused=0
    # Copy only small feature tensors. Rebind order through identity, never row number.
    for i,path in enumerate(paths):
        row=read(path)['trajectory_row'];tid=row['trajectory_id']
        if tid not in old['rows']:continue
        entry=old['rows'][tid];source=FEATURE_SOURCE/entry['file'];dest=out/'rows'/f'{i:04d}.pt'
        assert sha256(source)==entry['sha256']
        if dest.exists():assert sha256(dest)==entry['sha256']
        else:shutil.copy2(source,dest)
        reused+=1
    assert reused==3000
    dump(out/'REUSE.json',dict(reused_trajectories=3000,new_trajectories=500,
        source_index_sha256=sha256(FEATURE_SOURCE/'index.json'),selection_sha256=sha256(selection)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda gpu:execute(cfg,f'features_gpu{gpu}',base+
            ['--resume','--worker-index',str(gpu),'--device','cuda:0'],gpu),range(4)))
    execute(cfg,'features_finalize',base+['--resume','--finalize-only'])
    index=load_feature_index(out)
    assert len(index['rows'])==3500 and index['selection_sha256']==sha256(selection)


def cache(cfg):
    out=ROOT/(ROOT.name+'_cache')
    if not (out/'COMPLETE.json').exists():
        if out.exists():raise RuntimeError('partial cache requires recovery; refusing overwrite')
        execute(cfg,'cache',[sys.executable,'prepare_data.py','--collection-root',str(COLLECTION/'mixed'),
            '--selection',str(COLLECTION/'mixed/selection.json'),'--feature-cache',str(ROOT/(ROOT.name+'_features')),
            '--state-mode',MODE,'--output-dir',str(out)])
    marker=read(out/'COMPLETE.json');m=read(out/'manifest.json')
    assert marker['status']=='complete' and marker['sha256']==sha256(out/'transitions.pt')
    assert m['trajectories']==3500 and m['transitions']==175000
    assert m['selection']['split_counts']==dict(train=2800,val=350,test=350)
    assert m['selection_sha256']==sha256(COLLECTION/'mixed/selection.json')


def train_group(cfg,pair):
    gpu,g=pair
    if not (Path(g['training'])/'TRAINING_COMPLETE.json').exists():
        if Path(g['training']).exists():raise RuntimeError('partial training requires recovery; refusing overwrite')
        execute(cfg,g['name']+'_train',[sys.executable,'train.py','--dataset',g['cache'],
            '--state-mode',g['mode'],'--output-dir',g['training'],'--checkpoint-dir',g['weights'],
            '--device','cuda','--training-seed','42','--iql-profile',g['profile']],gpu)
    verify_training(g)
    if not (Path(g['analysis'])/'checkpoint_selection.json').exists():
        execute(cfg,g['name']+'_select',[sys.executable,'analyze_training.py','--training-result',g['training'],
            '--dataset',g['cache'],'--checkpoint-dir',g['weights'],'--output-dir',g['analysis'],'--device','cuda'],gpu)
    selected=read(Path(g['analysis'])/'checkpoint_selection.json')
    assert 300<selected['checkpoint_epoch']<400 and sha256(selected['checkpoint'])==selected['checkpoint_sha256']
    dump(ROOT/g['name']/'SELECTED.json',selected)


def evaluation_jobs(cfg):
    queues={i:[] for i in range(4)}
    for g in cfg['groups']:
        selected=read(Path(g['analysis'])/'checkpoint_selection.json')
        for k in BUDGETS:
            for row in cfg['prompts']:
                gpu=cfg['gpu_uuids'].index(row['baseline_gpu_uuid'])
                output=ROOT/'evaluation/candidates'/g['name']/f'K{k}'/row['sample_id']
                queues[gpu].append(dict(kind='candidate',group=g['name'],state_mode=g['mode'],
                    sample_id=row['sample_id'],prompt=row['prompt'],skip_budget=k,
                    expected_gpu_uuid=row['baseline_gpu_uuid'],
                    checkpoint=dict(path=selected['checkpoint'],sha256=selected['checkpoint_sha256']),output=str(output)))
    assert sum(map(len,queues.values()))==120
    for gpu,jobs in queues.items():dump(ROOT/'jobs'/f'gpu{gpu}.json',jobs)
    return queues


def quality(cfg,gpu,jobs):
    from ours4wan21.online_common import verified
    from generate_worker import job_identity
    d=directory(ROOT/'evaluation/quality_inputs'/f'gpu{gpu}','# Same-GPU quality pairs')
    for side in ('reference','candidate'):directory(d/side,'# Video symlinks')
    for j in jobs:
        assert verified(j['output'],job_identity(j,cfg))
        key=f"{j['group']}_K{j['skip_budget']}_{j['sample_id']}"
        for side,target in [('reference',REFERENCE/'baselines'/j['sample_id']/'video.mp4'),('candidate',Path(j['output'])/'video.mp4')]:
            link=d/side/(key+'.mp4')
            if not link.exists():link.symlink_to(target)
            assert link.resolve()==target.resolve()
    out=ROOT/'evaluation/quality'/f'gpu{gpu}'
    if (out/'summary.json').exists():
        assert read(out/'summary.json')['video_count']==len(jobs)
        return
    execute(cfg,f'quality_gpu{gpu}',[sys.executable,str(OFFICIAL/'VideoMetrics/evaluate.py'),
        '--reference-dir',str(d/'reference'),'--candidate-dir',str(d/'candidate'),'--expected-frames','81',
        '--device','cuda:0','--model-cache',str(MODEL_ROOT/'torch-cache'),'--output-dir',str(out)],gpu)


def report(cfg,queues):
    import csv
    from ours4wan21.online_common import verified
    from generate_worker import job_identity
    detail=[]
    assert cfg['prompts']==load_evaluation_bundle(REFERENCE)['rows']
    for g in cfg['groups']:verify_training(g)
    for gpu,jobs in queues.items():
        folder=ROOT/'evaluation/quality'/f'gpu{gpu}'
        summary=read(folder/'summary.json')
        assert summary['video_count']==len(jobs) and summary['frame_count_total']==81*len(jobs)
        qs={r['video_id']:r for r in csv.DictReader((folder/'per_video.csv').open())}
        assert set(qs)=={f"{j['group']}_K{j['skip_budget']}_{j['sample_id']}" for j in jobs}
        for j in jobs:
            out=Path(j['output']);assert verified(out,job_identity(j,cfg))
            gen=read(out/'generation.json');c=read(out/'measurement.json')
            assert gen['gpu_uuid']==j['expected_gpu_uuid']==cfg['gpu_uuids'][gpu] and gen['protocol']==PROTOCOL
            selected=read(ROOT/j['group']/'SELECTED.json')
            assert j['checkpoint']==dict(path=selected['checkpoint'],sha256=selected['checkpoint_sha256'])
            q=qs[f"{j['group']}_K{j['skip_budget']}_{j['sample_id']}"]
            assert (int(q['frames']),int(q['width']),int(q['height']))==(81,832,480)
            for side in ('reference','candidate'):assert sha256(q[side])==q[side+'_sha256']
            ref=REFERENCE/'baselines'/j['sample_id']
            assert Path(q['reference']).resolve()==(ref/'video.mp4').resolve()
            assert Path(q['candidate']).resolve()==(out/'video.mp4').resolve()
            trace=audit_trace(read(out/'trace.json'),read(out/'timing.json'),j['skip_budget'])
            assert all(math.isfinite(c[k]) and c[k]>=0 for k in FIELDS)
            assert all(math.isfinite(float(q[k+'_mean'])) for k in METRICS)
            baseline=read(ref/'measurement.json')
            detail.append(dict(group=j['group'],k=j['skip_budget'],sample_id=j['sample_id'],
                checkpoint_epoch=selected['checkpoint_epoch'],baseline_seconds=baseline['generate_seconds'],
                **{k:c[k] for k in FIELDS},**{k:float(q[k+'_mean']) for k in METRICS},**trace))
    assert len(detail)==120
    results=[]
    for g in cfg['groups']:
        for k in BUDGETS:
            rr=[r for r in detail if r['group']==g['name'] and r['k']==k];assert len(rr)==20
            results.append(dict(group=g['name'],k=k,checkpoint_epoch=rr[0]['checkpoint_epoch'],
                speedup=sum(r['baseline_seconds'] for r in rr)/sum(r['generate_seconds'] for r in rr),
                **{key:sum(r[key] for r in rr)/20 for key in (*FIELDS,*METRICS)}))
    writecsv(ROOT/'per_video.csv',detail);writecsv(ROOT/'results.csv',results)
    text=['# Mixed3500 Dynamics128 IQL comparison','',
          'Same frozen20 prompts, seed42, K23/29/35. Validation-only checkpoint selection. No VBench score.',
          'Single-seed descriptive results; conservative/aggressive differ jointly in tau, beta and weight cap.','',
          '| Group | Epoch | K | Speedup | PSNR | SSIM | LPIPS |','|---|---:|---:|---:|---:|---:|---:|']
    for r in results:text.append(f"| {r['group']} | {r['checkpoint_epoch']} | {r['k']} | {r['speedup']:.4f} | {r[METRICS[0]]:.4f} | {r[METRICS[1]]:.5f} | {r[METRICS[2]]:.5f} |")
    (ROOT/'REPORT.md').write_text('\n'.join(text)+'\n')
    dump(ROOT/'COMPLETE.json',dict(status='complete',groups=2,candidates=120,paired_frames=9720,
        vbench_status='skipped_by_user',results_sha256=sha256(ROOT/'results.csv'),
        detail_sha256=sha256(ROOT/'per_video.csv'),config_sha256=sha256(ROOT/'config.json')))


def run_pipeline():
    cfg=read(ROOT/'config.json');verify_sources(cfg)
    with (ROOT/'pipeline.lock').open('a') as local:
        fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'COMPLETE.json').exists():return
        dump(ROOT/'STATUS.json',dict(stage='waiting_for_increase500',pid=os.getpid()))
        while not dataset_ready():
            if (COLLECTION/'LAST_ERROR.json').exists():raise RuntimeError('collection reported an error; refusing partial dataset')
            time.sleep(20)
        with (EXP_ROOT/'wan21_benchmark_4gpu.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX)
            verify_sources(cfg)
            for stage,operation in [('features',lambda:features(cfg)),('cache',lambda:cache(cfg))]:
                dump(ROOT/'STATUS.json',dict(stage=stage));operation()
            dump(ROOT/'STATUS.json',dict(stage='training_and_selection'))
            with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda pair:train_group(cfg,pair),enumerate(cfg['groups'])))
            queues=evaluation_jobs(cfg)
            dump(ROOT/'STATUS.json',dict(stage='generation',candidates=120))
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda gpu:execute(cfg,f'generate_gpu{gpu}',[sys.executable,str(HERE/'generate_worker.py'),
                    '--jobs',str(ROOT/'jobs'/f'gpu{gpu}.json')],gpu),range(4)))
            dump(ROOT/'STATUS.json',dict(stage='quality'))
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda gpu:quality(cfg,gpu,queues[gpu]),range(4)))
            dump(ROOT/'STATUS.json',dict(stage='report'));report(cfg,queues)
            dump(ROOT/'STATUS.json',dict(stage='complete'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run']);a=p.parse_args()
    try:
        if a.phase=='prepare':prepare()
        else:run_pipeline()
    except Exception as exc:
        if ROOT.exists():dump(ROOT/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
