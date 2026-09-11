"""Audit existing 50-prompt results and preserve a reproducible paired comparison."""
import csv
import json
import math
from pathlib import Path
import statistics as st
import audit30

s = audit30.s
read, sha = s.read, s.sha

def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def main():
    sea_audit = audit30.audit()
    cfg = read(s.ROOT/'config.json')
    ours = Path(cfg['previous'])
    pc = read(ours/'config.json')
    done = read(ours/'COMPLETE.json')
    val = read(ours/'analysis/VALIDATION.json')
    assert done['status'] == 'complete' and val['status'] == 'pass'
    for file,key in [('results.csv','results_sha256'),('VALIDATION.json','validation_sha256')]:
        assert sha(ours/'analysis'/file) == done[key]
    assert sha(ours/'config.json') == val['config_sha256']
    for path,digest in val['evidence_sha256'].items():
        assert sha(ours/path) == digest
    assert sha(Path(pc['checkpoint'])) == pc['checkpoint_sha256']
    o = next(r for r in rows(ours/'analysis/results.csv') if int(r['k']) == 35)
    assert int(o['checkpoint_epoch']) == 391
    sea = sea_audit['result']
    sp = rows(s.ROOT/'analysis/per_video.csv')
    op = [r for r in rows(ours/'analysis/per_video.csv') if int(r['k']) == 35]
    assert len(op) == len({r['sample_id'] for r in op}) == 50
    om = {r['sample_id']:r for r in op}
    assert set(om) == {r['sample_id'] for r in sp}
    base, cand, pairs = [], [], []
    trace_counts = []
    for g in range(4):
        ids = cfg['shard_ids'][str(g)]
        d = ours/'shards'/f'gpu{g}'/'K35'
        bm = read(d.parent/'baseline/run.json')
        gm = s.prior.validate_generation(d,prompt_ids=ids,method='ours',mode=s.prior.MODE,skip_budget=35)
        assert gm['gpu_uuid'] == bm['gpu_uuid'] == cfg['gpu_uuids'][str(g)]
        assert gm['protocol'] == bm['protocol'] == cfg['protocol']
        assert gm['policy_sha256'] == pc['checkpoint_sha256']
        b = read(d.parent/'baseline/components.json')['rows']
        c = read(d/'components.json')['rows']
        q = s.prior.quality_rows(d/'quality',ids)
        sq = rows(s.ROOT/'shards'/f'gpu{g}'/'seacache/quality/per_video.csv')
        bysq = {r['video_id']:r for r in sq}
        byq = {r['video_id']:r for r in q}
        byb = {r['sample_id']:r for r in b}
        assert len(c) == len(ids) and {r['sample_id'] for r in c} == set(ids)
        for r in c:
            sid = r['sample_id']
            assert int(om[sid]['gpu']) == g
            assert float(om[sid]['candidate_seconds']) == r['generate_seconds']
            assert float(om[sid]['baseline_seconds']) == byb[sid]['generate_seconds']
            assert byq[sid]['reference_sha256'] == bysq[sid]['reference_sha256']
            for m in s.prior.METRICS:
                assert float(om[sid][m]) == float(byq[sid][m+'_mean'])
            for method,folder in [('e391',d),('seacache',s.ROOT/'shards'/f'gpu{g}'/'seacache')]:
                trace = read(folder/'traces'/f'{sid}.json')
                timing = read(folder/'timings'/f'{sid}.json')
                assert timing['status'] == 'success' and len(timing['calls']) == 100
                if method == 'e391':
                    assert timing['pipeline_generate_wall_seconds'] == r['generate_seconds']
                for offset,branch in enumerate(('cond','uncond')):
                    reuse = trace['per_branch'][branch]['reuse_path']
                    assert len(reuse) == 35
                    for step in range(50):
                        assert timing['calls'][2*step+offset]['blocks_executed'] == (0 if step in reuse else 30)
                trace_counts.append(dict(method=method,sample_id=sid,reuse=35,recompute=15))
        base += b
        cand += c
    for k,v in s.prior.summarize(base,cand).items():
        assert math.isclose(v,float(o[k]),abs_tol=1e-9)
    for m in s.prior.METRICS:
        assert math.isclose(st.fmean(float(r[m]) for r in op),float(o[m]),abs_tol=1e-10)
    for r in sp:
        other=om[r['sample_id']]
        assert r['gpu'] == other['gpu']
        assert float(r['baseline_seconds']) == float(other['baseline_seconds'])
        p=dict(sample_id=r['sample_id'],gpu=int(r['gpu']))
        for m in (*s.prior.METRICS,'candidate_seconds'):
            p['seacache_'+m]=float(r[m])
            p['e391_'+m]=float(other[m])
            p['delta_'+m]=float(r[m])-float(other[m])
        pairs.append(p)
    paired={}
    for m in (*s.prior.METRICS,'candidate_seconds'):
        ds=[r['delta_'+m] for r in pairs]
        lower=m in ('lpips_alex_v0_1_spatial','candidate_seconds')
        paired[m]=dict(mean_delta_seacache_minus_e391=st.fmean(ds),median_delta=st.median(ds),
            min_delta=min(ds),max_delta=max(ds),seacache_better=sum(v<0 if lower else v>0 for v in ds),
            e391_better=sum(v>0 if lower else v<0 for v in ds),ties=sum(v==0 for v in ds))
    out=s.ROOT/'analysis/e391_comparison'
    out.mkdir(exist_ok=True)
    (out/'README.md').write_text('# SeaCache 3.0× / e391 K35\n\nreport.html为自包含报告，artifact.json为同源输入；paired.csv保留全部50个配对，\ncomparison.csv为汇总，AUDIT.json记录校验与描述性差值。\n脚本：SeaCache4Wan21/experiments/speed30_e391_readout_v1/compare.py。\n')
    s.prior.write_csv(out/'paired.csv',pairs)
    comparisons=[]
    for label,r in [('SeaCache',sea),('Dynamics128 e391 K35',o)]:
        comparisons.append(dict(method=label,**{k:float(r[k]) for k in sea if k in o and k not in ('method','target','threshold')}))
    s.prior.write_csv(out/'comparison.csv',comparisons)
    audit=dict(status='pass',checked_at=s.now(),seacache=sea_audit,paired=paired,
        e391_evidence_files=len(val['evidence_sha256']),paired_prompts=50,checked_branch_steps=10000,
        source_sha256={str(p):sha(p) for root in (s.ROOT,ours) for p in (root/'config.json',root/'COMPLETE.json',root/'analysis/results.csv',root/'analysis/per_video.csv',root/'analysis/VALIDATION.json')},
        limitations=['50 random prompts, generation seed42 only; descriptive comparison, no significance claim.',
            'VBench explicitly skipped in both frozen experiments; quality not redecoded in this readout.',
            'Timing runs were separate; a 0.3% difference is not evidence of a stable latency advantage.'])
    s.dump(out/'AUDIT.json',audit)
    print(json.dumps(dict(status='pass',output=str(out),paired=paired,comparison=comparisons),ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
