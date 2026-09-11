"""Read-only diagnostics of four completed online collection rounds."""
from pathlib import Path
import csv, json, importlib.util, statistics, itertools
from collections import Counter
import numpy as np

HERE=Path(__file__).resolve().parent
HELPER=HERE.parent/'iql_increase_traces_v1/plot.py'
spec=importlib.util.spec_from_file_location('audit_helper',HELPER)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
RUN=h.BASE/'ours21_dynamics128_e391_online_offline800_8rounds_a25_v1'
OUT=RUN/'analysis/r1_r4_policy_diagnostics';h.OUT=OUT

def mean(xs): return statistics.fmean(xs)
def lengths(path): return [len(x) for x in path.split('0') if x]
def features(path):
    runs=lengths(path);a=lengths(path[:25]);b=lengths(path[25:])
    return dict(early=path[:25].count('1'),late=path[25:].count('1'),
        longest=max(runs,default=0),early_longest=max(a,default=0),late_longest=max(b,default=0),
        runs=len(runs),mean_run=mean(runs),first_skip=path.index('1')+1,
        centroid=mean(i+1 for i,v in enumerate(path) if v=='1'),
        late20=path[30:].count('1'),late15=path[35:].count('1'))

def main():
    OUT.mkdir(exist_ok=True);manifest=h.js(RUN/'manifest.json');data=[];parents=[]
    for n in range(1,5):
        root=RUN/'rounds'/f'round_{n:03d}';plan=h.js(root/'plan.json');assert len(plan['rows'])==100
        qseal=h.js(root/'quality/COMPLETE.json')
        for name,digest in qseal['files'].items():h.read(root/'quality'/name,digest)
        quality={r['video_id']:r for r in h.rows(root/'quality/metrics/per_video.csv')}
        assert set(quality)=={r['trajectory_id'] for r in plan['rows']}
        train=h.js(root/'training/COMPLETE.json')
        for name,digest in train['files'].items():h.read(root/'training'/name,digest)
        selected=h.js(root/'training/selected.json');assert h.sha(selected['path'])==selected['sha256']
        expected=({'path':manifest['paths']['start_checkpoint'],'sha256':manifest['inputs']['start_checkpoint']['sha256']} if n==1 else {k:parents[-1]['selected'][k] for k in ('path','sha256')})
        assert expected==plan['parent'] and h.sha(expected['path'])==expected['sha256']
        parents.append(dict(round=n,parent=expected,selected=selected))
        for row in plan['rows']:
            uid=row['trajectory_id'];p=root/'collection'/uid;seal=h.js(p/'COMPLETE.json');job=seal['identity']['job']
            assert job['checkpoint']==expected and job['sampling_seed']==row['sampling_seed'] and job['sample_id']==row['sample_id']
            for name in ('trace.json','timing.json','measurement.json','generation.json'):h.read(p/name,seal['files'][name])
            trace=h.js(p/'trace.json');path=''.join('1' if i in trace['per_branch']['cond']['reuse_path'] else '0' for i in range(50))
            audited=h.audit(p/'trace.json',p/'timing.json',uid,path)
            assert audited['skip_steps']==row['skip_budget'] and path[0]==path[-1]=='0'
            q=quality[uid];base=Path(manifest['paths']['training_bundle'])/'baselines'/row['sample_id'];bs=h.js(base/'COMPLETE.json')
            assert q['candidate_sha256']==seal['files']['video.mp4'] and q['reference_sha256']==bs['files']['video.mp4']
            h.read(base/'measurement.json',bs['files']['measurement.json'])
            bt=h.js(base/'measurement.json')['generate_seconds'];ct=h.js(p/'measurement.json')['generate_seconds']
            r=dict(round=n,trace_id=uid,prompt_id=row['sample_id'],sampling_seed=row['sampling_seed'],k=row['skip_budget'],
                psnr=float(q['psnr_rgb_db_mean']),speedup=bt/ct,baseline_seconds=bt,generate_seconds=ct,skip_path=path,**features(path))
            assert r['early']+r['late']==r['k'] and np.isfinite(r['psnr']) and np.isfinite(r['speedup'])
            data.append(r)
    assert len(data)==len({r['trace_id'] for r in data})==400 and len(h.RAW)==40000
    metrics=['speedup','psnr','k','early','late','longest','early_longest','late_longest','runs','mean_run','first_skip','centroid','late20','late15']
    def summarize(rows):
        d=dict(n=len(rows),**{m:mean(r[m] for r in rows) for m in metrics})
        d.update(longest_median=float(np.median([r['longest'] for r in rows])),longest_p90=float(np.quantile([r['longest'] for r in rows],.9)),longest_max=max(r['longest'] for r in rows),
            late_longest_p90=float(np.quantile([r['late_longest'] for r in rows],.9)),late_longest_max=max(r['late_longest'] for r in rows),
            ge10=sum(r['longest']>=10 for r in rows),ge14=sum(r['longest']>=14 for r in rows),late_ge14=sum(r['late_longest']>=14 for r in rows),
            late_ge22=sum(r['late']>=22 for r in rows),late_more=sum(r['late']>r['early'] for r in rows),
            late_share=sum(r['late'] for r in rows)/sum(r['k'] for r in rows),ratio_total_seconds=sum(r['baseline_seconds'] for r in rows)/sum(r['generate_seconds'] for r in rows))
        return d
    summary=[dict(round=n,**summarize([r for r in data if r['round']==n])) for n in range(1,5)]
    byk=[dict(round=n,budget=k,**summarize(rr)) for n in range(1,5) for k in sorted({r['k'] for r in data}) if (rr:=[r for r in data if r['round']==n and r['k']==k])]
    common=set.intersection(*[{r['k'] for r in data if r['round']==n} for n in range(1,5)])
    counts=Counter(r['k'] for r in data if r['k'] in common);weights={k:v/sum(counts.values()) for k,v in counts.items()}
    standardized=[dict(round=n,**{m:sum(weights[k]*next(x[m] for x in byk if x['round']==n and x['budget']==k) for k in common) for m in metrics}) for n in range(1,5)]
    bands=[dict(round=n,band=f'{lo}–{hi}',**summarize([r for r in data if r['round']==n and lo<=r['k']<=hi])) for n in range(1,5) for lo,hi in [(17,23),(24,30),(31,37)]]
    profiles=[dict(round=n,step=i+1,skips=sum(r['skip_path'][i]=='1' for r in data if r['round']==n),n=100,rate=mean(int(r['skip_path'][i]) for r in data if r['round']==n)) for n in range(1,5) for i in range(50)]
    ten=[dict(round=n,window=f'{i+1:02d}–{i+10:02d}',mean_skip=mean(r['skip_path'][i:i+10].count('1') for r in data if r['round']==n)) for n in range(1,5) for i in range(0,50,10)]
    # Trajectory counts, not run-weighted histograms.
    hist=[dict(round=n,bin=f'{lo}–{hi}',count=sum(lo<=r['longest']<=hi for r in data if r['round']==n),n=100) for n in range(1,5) for lo,hi in [(1,4),(5,7),(8,10),(11,13),(14,16),(17,50)]]
    h.writecsv('trajectories.csv',data);h.writecsv('round_summary.csv',summary);h.writecsv('by_k.csv',byk);h.writecsv('standardized.csv',standardized);h.writecsv('budget_bands.csv',bands);h.writecsv('step_profile.csv',profiles);h.writecsv('ten_step_profile.csv',ten);h.writecsv('run_histogram.csv',hist)
    # Prior R1–R3 numeric agreement is an independent source reconciliation.
    prior=h.js(RUN/'analysis/r1_r2_increase_long/round123_means.json')
    for r,old in zip(summary,prior['results']):
        assert abs(r['speedup']-old['mean_speedup'])<1e-12 and abs(r['psnr']-old['mean_psnr_db'])<1e-12
    source=HERE/'analyze.py';h.SOURCES[str(source)]=h.sha(source)
    validation=dict(status='pass',traces=400,cfg_calls=40000,cells=20000,source_sha256=h.SOURCES,parents=parents,
        common_k=sorted(common),pooled_k_weights=weights,standardization_coverage=sum(counts.values())/400,
        output_sha256={p.name:h.sha(p) for p in OUT.glob('*.csv')})
    (OUT/'VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    (OUT/'SUMMARY.json').write_text(json.dumps(dict(rounds=summary,standardized=standardized,bands=bands,ten_step=ten),ensure_ascii=False,indent=2)+'\n')
    (OUT/'README.md').write_text('# R1–R4 collection policy diagnostics\n\nreport.html: portable technical report; artifact.json: canonical report data; analysis.ipynb: executable companion; SUMMARY.json: metrics; trajectories.csv: all 400 paths; round_summary/by_k/standardized/budget_bands/step_profile/ten_step_profile/run_histogram.csv: reviewed aggregates; VALIDATION.json: sealed source hashes and checkpoint provenance. Code: experiments/online_r1_r4_policy_diagnostics_v1/.\n')
    print(json.dumps(dict(rounds=summary,standardized=standardized,bands=bands),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
