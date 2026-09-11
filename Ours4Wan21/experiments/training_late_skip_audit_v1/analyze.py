"""Audit measured late-skip coverage in the frozen dataset used by Dynamics128 e391."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

EXP = Path('/mnt/hdd/xiongyuxiang/tmp/exp')
PREFIX = 'ours21_random3000_12groups_v1_sea7_dynamics_raw_sea128'
CACHE = EXP / (PREFIX + '_cache')
TRAIN = EXP / (PREFIX + '_train')
INC = EXP / 'wan21_increase_e391_trace_comparison_v1'
OUT = EXP / 'ours21_training_late_skip_audit_v1'
SOURCES = {}


def read(path, expected=None):
    path = Path(path)
    value = path.read_bytes()
    digest = hashlib.sha256(value).hexdigest()
    if expected is not None:
        assert digest == expected, str(path)
    SOURCES[str(path)] = digest
    return value


def js(path, expected=None):
    return json.loads(read(path, expected))


def writecsv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def longest(values):
    best = run = 0
    for v in values:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def measures(a):
    k = int(sum(a))
    return dict(k=k, early25=int(sum(a[:25])), late25=int(sum(a[25:])),
        late20=int(sum(a[30:])), late15=int(sum(a[35:])), run25=longest(a[25:]),
        run20=longest(a[30:]), run15=longest(a[35:]),
        centroid=float(np.dot(np.arange(1,51), a)/k),
        late_share=float(sum(a[25:])/k), skip_path=''.join(map(str,map(int,a))))


def main():
    torch.set_num_threads(1)
    manifest = js(CACHE/'manifest.json')
    assert js(TRAIN/'dataset_manifest.json') == manifest
    assert manifest['trajectories'] == 3000 and manifest['transitions'] == 150000
    assert manifest['selection_sha256'] == 'f98653e3db6a8ac9b03567744bcd679810705628949c899bc4655653a2862208'
    assert manifest['selection']['strategy'] == 'all-completed'
    complete = js(CACHE/'COMPLETE.json')
    read(CACHE/'transitions.pt', complete['sha256'])
    bundle = torch.load(CACHE/'transitions.pt', map_location='cpu', weights_only=False)
    assert bundle['manifest'] == manifest
    a = bundle['tensors']['action'].reshape(3000,50).numpy().astype(int)
    mask = bundle['tensors']['actor_mask'].reshape(3000,50).numpy().astype(int)
    rewards = bundle['tensors']['reward'].reshape(3000,50).numpy()
    assert np.isin(a, [0,1]).all() and np.isin(mask,[0,1]).all()
    assert not a[:,[0,49]].any() and not mask[:,[0,49]].any()
    assert not rewards[:,:49].any()
    indices = {}
    for split, field in (('train','train_indices'),('evaluation','val_indices'),('test','test_indices')):
        idx=bundle[field].numpy();indices[split]=set(idx.tolist())
        assert len(idx)==(120000 if split=='train' else 15000)
    assert len(set.union(*indices.values())) == 150000
    references=[]
    old_valid=js(INC/'VALIDATION.json')
    old_rows=list(csv.DictReader(read(INC/'trace_summary_sorted.csv').decode().splitlines()))
    for r in old_rows:
        if r['method'] != 'increase': continue
        t=js(r['source_trace'],old_valid['source_sha256'][r['source_trace']])
        paths=[[int(d['action']=='reuse') for d in t['decisions'] if d['branch']==b] for b in ('cond','uncond')]
        assert paths[0]==paths[1] and len(paths[0])==50
        m=measures(paths[0]);assert m['skip_path']==r['skip_path']
        references.append(dict(level=int(r['level']),sample_id=r['sample_id'],**m))
    assert len(references)==20
    rows=[]
    for i,s in enumerate(manifest['sources']):
        assert all(j in indices[s['split']] for j in range(i*50,(i+1)*50))
        source={p:js(p,h) for p,h in s['files'].items()}
        tr=next(v for p,v in source.items() if p.endswith('/trace.json'))
        completed=next(v for p,v in source.items() if '/completed/' in p)
        cr=completed['trajectory_row']; mr=tr['manifest_record']
        assert tr['trajectory_id']==s['trajectory_id']==cr['trajectory_id']
        assert tr['policy_family']=='random_continuous_seacache_threshold'
        assert tr['total_steps']==50 and len(tr['decisions'])==100
        branches=[]
        for b in ('cond','uncond'):
            ds=[d for d in tr['decisions'] if d['branch']==b]
            assert [d['step_index'] for d in ds]==list(range(50))
            assert all(d['action']==d['execution'] and d['action'] in ('reuse','recompute') for d in ds)
            branches.append([int(d['execution']=='reuse') for d in ds])
        assert branches[0]==branches[1]==a[i].tolist()
        assert sum(a[i])*2==tr['reuse'] and sum(a[i])==cr['actual_both_reuse_steps']
        assert np.isclose(float(rewards[i,-1]),cr['mean_psnr'],atol=1e-5)
        threshold=np.array(tr['threshold_path'])
        assert threshold.shape==(50,) and np.isfinite(threshold).all()
        assert threshold.tolist()==mr['threshold_path']==cr['threshold_path']
        assert min(threshold)>=mr['threshold_min']-1e-12 and max(threshold)<=mr['threshold_max']+1e-12
        assert mr['threshold_min']==.04 and mr['threshold_max']==.7
        m=measures(a[i])
        actor_late=mask[i,25:]
        rows.append(dict(trajectory_id=s['trajectory_id'],sample_id=s['sample_id'],split=s['split'],**m,
            psnr=float(rewards[i,-1]),actor_late_rows=int(sum(actor_late)),
            actor_late_skips=int(np.dot(actor_late,a[i,25:])),
            forced_late_skips=int(np.dot(1-actor_late,a[i,25:])),
            threshold_first25=float(threshold[:25].mean()),threshold_last25=float(threshold[25:].mean()),
            threshold_end=float(threshold[-1]),threshold_peak=float(threshold.max()),
            threshold_monotone_increasing=bool(np.all(np.diff(threshold)>=-1e-12)),
            target_speedup=mr['target_speedup'],q=mr['q'],prompt=cr['prompt']))
        if (i+1)%500==0: print(f'Validated {i+1}/3000 raw traces against training tensors',flush=True)
    assert len({r['trajectory_id'] for r in rows})==3000
    assert Counter(r['split'] for r in rows)==dict(train=2400,evaluation=300,test=300)
    prompt_splits={}
    for r in rows:
        assert prompt_splits.setdefault(r['sample_id'],r['split'])==r['split']
    assert set(Counter(r['sample_id'] for r in rows).values())=={3}
    matched=[]
    for split in ('train','evaluation','test','all'):
        scope=[r for r in rows if split=='all' or r['split']==split]
        for k in sorted({r['k'] for r in references}):
            ref=[r for r in references if r['k']==k];group=[r for r in scope if r['k']==k]
            base=dict(split=split,k=k,reference_n=len(ref),n=len(group),prompts=len({r['sample_id'] for r in group}))
            for window in (25,20,15):
                threshold=min(r[f'late{window}'] for r in ref)
                run=min(r[f'run{window}'] for r in ref)
                base.update({f'ref_late{window}':threshold,f'ref_run{window}':run,
                    f'match_late{window}':sum(r[f'late{window}']>=threshold for r in group),
                    f'match_both{window}':sum(r[f'late{window}']>=threshold and r[f'run{window}']>=run for r in group)})
            base.update(ref_centroid=min(r['centroid'] for r in ref),
                match_centroid=sum(r['centroid']>=min(t['centroid'] for t in ref)-1e-10 for r in group),
                late25_rate=base['match_late25']/len(group) if group else None,
                both25_rate=base['match_both25']/len(group) if group else None,
                late25_median=float(np.median([r['late25'] for r in group])) if group else None,
                late25_max=max((r['late25'] for r in group),default=None),
                match_prompts=len({r['sample_id'] for r in group if r['late25']>=base['ref_late25']}))
            matched.append(base)
    criteria=[]
    for split in ('train','evaluation','test','all'):
        group=[r for r in rows if split=='all' or r['split']==split]
        tests={'late25_gt_early25':lambda r:r['late25']>r['early25'],
            'late25_ge20':lambda r:r['late25']>=20,'late25_ge21':lambda r:r['late25']>=21,
            'late25_ge22':lambda r:r['late25']>=22,'late25_ge23':lambda r:r['late25']>=23,
            'late25_ge22_and_run25_ge14':lambda r:r['late25']>=22 and r['run25']>=14,
            'run25_ge14':lambda r:r['run25']>=14,'run25_ge15':lambda r:r['run25']>=15,
            'monotone_increasing_threshold':lambda r:r['threshold_monotone_increasing'],
            'threshold_last25_gt_first25':lambda r:r['threshold_last25']>r['threshold_first25']}
        for name,fn in tests.items():
            good=[r for r in group if fn(r)]
            criteria.append(dict(split=split,criterion=name,count=len(good),n=len(group),rate=len(good)/len(group),
                prompts=len({r['sample_id'] for r in good}),actor_late_skips=sum(r['actor_late_skips'] for r in good)))
    kprofile=[]
    for k in sorted({r['k'] for r in rows}):
        group=[r for r in rows if r['split']=='train' and r['k']==k]
        if not group:continue
        kprofile.append(dict(k=k,n=len(group),late25_mean=float(np.mean([r['late25'] for r in group])),
            late25_median=float(np.median([r['late25'] for r in group])),
            late25_p90=float(np.percentile([r['late25'] for r in group],90)),
            late25_min=min(r['late25'] for r in group),late25_max=max(r['late25'] for r in group),
            run25_max=max(r['run25'] for r in group)))
    OUT.mkdir(parents=True,exist_ok=True)
    writecsv('training_trajectories.csv',rows);writecsv('increase_references.csv',references)
    writecsv('matched_k_coverage.csv',matched);writecsv('criteria_sensitivity.csv',criteria);writecsv('train_k_profile.csv',kprofile)
    hist=[dict(late25=x,count=sum(r['split']=='train' and r['late25']==x for r in rows)) for x in range(26)]
    writecsv('late25_histogram.csv',hist)
    summary=dict(train_trajectories=2400,all_trajectories=3000,train_prompts=800,
        train_max_late25=max(r['late25'] for r in rows if r['split']=='train'),
        all_max_late25=max(r['late25'] for r in rows),
        train_threshold_peak=max(r['threshold_peak'] for r in rows if r['split']=='train'),
        train_threshold_monotone=sum(r['split']=='train' and r['threshold_monotone_increasing'] for r in rows),
        matched=[r for r in matched if r['split']=='train'],criteria=[r for r in criteria if r['split']=='train'])
    dump('SUMMARY.json',summary)
    SOURCES[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump('VALIDATION.json',dict(status='pass',train_manifest_matches_cache=True,
        train_tensor_sha256_matches_complete=True,raw_traces=3000,raw_cfg_decisions=300000,
        actual_training_actions=150000,all_cfg_actions_match_training_tensor=True,
        split_trajectories=dict(train=2400,evaluation=300,test=300),no_prompt_split_leakage=True,
        selection_sha256=manifest['selection_sha256'],source_sha256=SOURCES))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
