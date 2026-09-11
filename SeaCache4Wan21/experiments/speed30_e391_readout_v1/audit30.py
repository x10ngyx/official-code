"""Read-only final-result audit of SeaCache random50 nominal3.0x."""
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vbench50_speed30_v1'))
import run_suite30 as s


def audit():
    cfg=s.prepare(True); root=s.ROOT; done=s.read(root/'COMPLETE.json')
    assert done['status']=='complete' and done['candidates']==50
    for name,key in [('results.csv','results_sha256'),('VALIDATION.json','validation_sha256')]:
        assert s.sha(root/'analysis'/name)==done[key]
    validation=s.read(root/'analysis/VALIDATION.json')
    assert s.sha(root/'config.json')==validation['config_sha256']
    for path,digest in validation['evidence_sha256'].items(): assert s.sha(root/path)==digest
    results=s.pair.read_csv(root/'analysis/results.csv'); assert len(results)==1
    result=results[0]; assert float(result['target'])==3.0
    details=s.pair.read_csv(root/'analysis/per_video.csv')
    assert len(details)==len({r['sample_id'] for r in details})==50
    base=[]; candidate=[]; quality=[]; checked=0
    for g in s.prior.GPUS:
        ids=cfg['shard_ids'][str(g)]; d=root/'shards'/f'gpu{g}'/'seacache'
        b=s.read(Path(cfg['previous'])/'shards'/f'gpu{g}'/'baseline/components.json')['rows']
        c=s.pair.check_generation(d,g,'seacache',ids,cfg); q=s.prior.quality_rows(d/'quality',ids)
        base+=b; candidate+=c; quality+=q
        for sid in ids:
            t=s.read(d/'traces'/f'{sid}.json'); timing=s.read(d/'timings'/f'{sid}.json')
            for offset,(branch,tbranch) in enumerate((('cond','condition'),('uncond','uncondition'))):
                decisions=[r for r in t['decisions'] if r['branch']==branch]
                assert [r['step_index'] for r in decisions]==list(range(50))
                assert all(r['action'] in ('reuse','recompute') for r in decisions)
                assert [i for i,r in enumerate(decisions) if r['action']=='reuse']==t['per_branch'][branch]['reuse_path']
                for step,row in enumerate(decisions):
                    call=timing['calls'][2*step+offset]
                    assert call['cfg_branch']==tbranch and call['blocks_executed']==(0 if row['action']=='reuse' else 30)
                    checked+=1
        byb={r['sample_id']:r for r in b}; byc={r['sample_id']:r for r in c}; byq={r['video_id']:r for r in q}
        for detail in [r for r in details if int(r['gpu'])==g]:
            sid=detail['sample_id']; assert sid in ids
            assert float(detail['baseline_seconds'])==byb[sid]['generate_seconds']
            assert float(detail['candidate_seconds'])==byc[sid]['generate_seconds']
            for m in s.prior.METRICS: assert float(detail[m])==float(byq[sid][m+'_mean'])
    candidate=[dict(r,**{k:0. for k in s.prior.FIELDS[7:] if k not in r}) for r in candidate]
    for key,value in s.prior.summarize(base,candidate).items(): assert math.isclose(value,float(result[key]),abs_tol=1e-9)
    for m in s.prior.METRICS:
        values=[float(q[m+'_mean']) for q in quality]
        assert math.isclose(statistics.fmean(values),float(result[m]),abs_tol=1e-10)
        assert math.isclose(statistics.stdev(values),float(result[m+'_std_across_prompts']),abs_tol=1e-10)
    return dict(status='pass',completed_at=done['completed_at'],candidates=50,quality_frames=4050,
        source_files=len(cfg['source_sha256']),evidence_files=len(validation['evidence_sha256']),checked_branch_steps=checked,result=result)


if __name__=='__main__': print(json.dumps(audit(),ensure_ascii=False,indent=2))
