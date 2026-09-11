"""Freeze offline train baseline references and recompute the eight-hour budget."""
from pathlib import Path
import sys
import math
import statistics
from collections import Counter
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ours4wan21.contracts import EXP_ROOT, create_result, dump, sha256
from ours4wan21.online_train_reference import build_training_bundle
from ours4wan21.online_common import read,OnlineConfig,make_plan
from ours4wan21.policy import resolve_budget
NAME='ours21_sea7_e328_online_offline800_8h_v1'
SETUP=EXP_ROOT/(NAME+'_setup')
OLD=EXP_ROOT/'ours21_sea7_e328_online_8h_v1_setup'
BASE=EXP_ROOT/'ours21_offline_train800_baseline_reference_v1'
CACHE=EXP_ROOT/'ours21_random3000_12groups_v1_sea7_cache'

def main():
    ref=build_training_bundle(CACHE/'manifest.json',PROJECT/'data_collection/resources/prompts/openvidhd_balanced_5000.upstream.jsonl',
        EXP_ROOT/'wan21_random_threshold_collection_v1_stage1',BASE)
    old=read(OLD/'budget.json');cal=OLD/'calibration.json'
    # Reuse already verified empirical timing prior, retaining its source digest.
    for p,h in old['sources'].items():
        if sha256(p)!=h:raise ValueError('timing evidence changed: '+p)
    uuids=[read(EXP_ROOT/'ours21_dynamics128_e391_vbench50_random42_4gpu_v1'/'shards'/f'gpu{g}'/'baseline/run.json')['gpu_uuid'] for g in range(4)]
    slots={r['sample_id']:uuids.index(r['baseline_gpu_uuid']) for r in ref['rows']}
    by_k={e['skip_budget']:e['mean_candidate_seconds'] for e in read(cal)['entries']}
    warm=old['baseline_seconds']+old['model_init_seconds'];cases=[]
    for n in range(1,9):
        c=OnlineConfig(rounds=n,prompt_pool_size=len(ref['rows']));details=[]
        for r in range(1,n+1):
            plan=make_plan(ref['rows'],r,4,lambda t:resolve_budget(target_speedup=t,calibration=cal),c,gpu_slots=slots)
            work=[0.]*4
            for row in plan:work[row['slot']]+=by_k[row['skip_budget']]+4.
            # Conservative for any initialization lock order: slowest shard starts last.
            generation=4*warm+max(work)
            quality=25*old['quality_seconds_per_video']+20
            training=25*old['offline_epoch_seconds']*(120000+5000*r)/120000+25
            details.append(dict(round=r,trajectories=[sum(x['slot']==g for x in plan) for g in range(4)],new_baselines=0,
                generation_seconds=generation,quality_seconds=quality,training_seconds=training,subtotal_seconds=generation+quality+training))
        evals=[]
        for i,r in enumerate(c.evaluation_rounds()):
            # Old cases contain matching two-policy first and one-policy later evaluation costs.
            policies=2 if i==0 else 1
            prior=EXP_ROOT/'ours21_dynamics128_e391_vbench50_random42_4gpu_v1'
            rows=read(EXP_ROOT/'ours21_online_eval20_from_vbench50_random42_v1/manifest.json')['rows']
            counts=Counter(x['baseline_gpu_uuid'] for x in rows);works=[]
            for g in range(4):
                secs=sum(statistics.mean(x['generate_seconds'] for x in read(prior/'shards'/f'gpu{g}'/f'K{k}'/'components.json')['rows'])+4 for k in (23,29,35))
                works.append(counts[uuids[g]]*policies*secs)
            value=4*warm+max(works)+3*policies*(5*old['quality_seconds_per_video']+20)
            evals.append(dict(round=r,seconds=value))
        cases.append(dict(rounds=n,total_hours=(sum(x['subtotal_seconds'] for x in details)+sum(x['seconds'] for x in evals))/3600,round_details=details,evaluations=evals))
    best=max((x for x in cases if x['total_hours']<=8),key=lambda x:x['rounds'])
    create_result(SETUP,'# SEA7 offline-prompt online setup\n\nbudget.json records candidate-only four-GPU costs including loading, native warmup, quality, training and scheduled/final evaluation. train_baselines is a symlink to 800 offline train native references. calibration.json reuses the existing train-only timing prior.\n')
    (SETUP/'train_baselines').symlink_to(BASE,target_is_directory=True);(SETUP/'calibration.json').symlink_to(cal)
    dump(SETUP/'budget.json',dict(budget_hours=8,selected_rounds=best['rounds'],selected_estimated_hours=best['total_hours'],cases=cases,
        training_population=len(ref['rows']),sampling='100 uniform draws with replacement per round, deterministic independent prompt/target RNG',
        sources={str(OLD/'budget.json'):sha256(OLD/'budget.json'),str(BASE/'manifest.json'):sha256(BASE/'manifest.json')},
        assumptions='No new training baselines; 4 s candidate export; four serialized model/native warmups per phase; conservative slowest GPU starts last; four-GPU quality; cumulative replay; evaluation every four and final. Estimate is not a wall-time limit.'))
    print([(x['rounds'],round(x['total_hours'],3)) for x in cases],flush=True)
    print('SELECTED',best['rounds'],best['total_hours'],flush=True)
if __name__=='__main__':main()
