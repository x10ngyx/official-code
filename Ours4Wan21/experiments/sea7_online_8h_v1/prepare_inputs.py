"""Prepare SEA7 e328 inputs and estimate complete online rounds in eight hours."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from collections import defaultdict
from types import SimpleNamespace

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from ours4wan21.contracts import EXP_ROOT, PROTOCOL, create_result, dump, sha256
from ours4wan21.online_common import OnlineConfig, read, prompts, make_plan, balanced_training_slots
from ours4wan21.online_setup import build_pool
from ours4wan21.policy import resolve_budget

NAME='ours21_sea7_e328_online_8h_v1'
SETUP=EXP_ROOT/(NAME+'_setup')
CACHE=EXP_ROOT/'ours21_random3000_12groups_v1_sea7_cache'
REGISTRY=PROJECT/'data_collection/resources/prompts/openvidhd_balanced_5000.upstream.jsonl'
REFERENCE=EXP_ROOT/'ours21_online_eval20_from_vbench50_random42_v1'


def main():
    if SETUP.exists():
        raise FileExistsError('setup already exists; inspect its frozen inputs rather than overwrite')
    manifest=read(CACHE/'manifest.json')
    groups=defaultdict(list);evidence=[];quality=[]
    for source in manifest['sources']:
        if source['split']!='train':continue
        completion=next(Path(p) for p in source['files'] if '/completed/' in p)
        digest=sha256(completion)
        if digest!=source['files'][str(completion)]:raise ValueError('offline completion hash changed')
        data=read(completion);row=data['trajectory_row'];k=row['actual_both_reuse_steps']
        if (row['sample_id']!=source['sample_id'] or row['split']!='train' or
                len(data['step_rows'])!=50 or len(data['branch_rows'])!=100 or
                k!=sum(s['action']=='reuse' for s in data['step_rows']) or
                row['actual_reuse_branch_calls']!=2*k or
                any(s['cond_action']!=s['uncond_action'] for s in data['step_rows'])):
            raise ValueError('offline training trace/count mismatch')
        if row['timing_scope']!='pipeline_generate_wall_seconds':raise ValueError('wrong timing scope')
        bp=read(Path(row['baseline_video']).parent/'performance.json')
        cp=read(Path(row['trace_json']).parent/'performance.json')
        if (bp['flops_profile_sha256']!=cp['flops_profile_sha256'] or
                bp['pipeline_generate_wall_seconds']!=row['baseline_inference_seconds'] or
                cp['pipeline_generate_wall_seconds']!=row['candidate_inference_seconds'] or
                bp['full_compute_forward_calls']!=100 or cp['reuse_forward_calls']!=2*k):
            raise ValueError('source component measurement mismatch')
        entry=dict(sample_id=row['sample_id'],trajectory_id=row['trajectory_id'],skip_budget=k,
            baseline_seconds=row['baseline_inference_seconds'],candidate_seconds=row['candidate_inference_seconds'],
            shard=row['shard_index'],completion=str(completion),sha256=digest,
            baseline_performance_sha256=sha256(Path(row['baseline_video']).parent/'performance.json'),
            candidate_performance_sha256=sha256(Path(row['trace_json']).parent/'performance.json'))
        groups[k].append(entry);evidence.append(entry)
        quality.append(row['video_metrics_evaluation_seconds'])
    create_result(SETUP,'# SEA7 e328 online setup\n\ncalibration.json is an empirical training-only timing prior grouped by observed reuse K. calibration_evidence.json retains source hashes; budget.json records the eight-hour round estimate. No GPU generation is performed by this script.')
    entries=[]
    for k,rows in sorted(groups.items()):
        if k<16:continue
        entries.append(dict(skip_budget=k,samples=len(rows),
            calibrated_speedup=sum(r['baseline_seconds'] for r in rows)/sum(r['candidate_seconds'] for r in rows),
            mean_baseline_seconds=statistics.mean(r['baseline_seconds'] for r in rows),
            mean_candidate_seconds=statistics.mean(r['candidate_seconds'] for r in rows)))
    if any(a['calibrated_speedup']>b['calibrated_speedup'] for a,b in zip(entries,entries[1:])):
        raise ValueError('observed timing prior is not monotone; no smoothing or theoretical extrapolation')
    if entries[0]['calibrated_speedup']>1.5 or entries[-1]['calibrated_speedup']<3.5:
        raise ValueError('observed counts do not cover requested target range')
    dump(SETUP/'calibration_evidence.json',evidence)
    dump(SETUP/'calibration.json',dict(schema='ours4wan21_speed_to_k_v1',status='calibrated',
        protocol=PROTOCOL,forced_steps=[0,49],entries=entries,
        source_method='offline train random_continuous_seacache_threshold, grouped by realized K',
        actor_specific_calibration=False,source_dataset=str(CACHE/'manifest.json'),
        source_dataset_sha256=sha256(CACHE/'manifest.json'),
        evidence_sha256=sha256(SETUP/'calibration_evidence.json'),
        note='Empirical ratio of paired complete generate times, train split only, observed K16–38; no fitted/theoretical points. Same frozen archive shard pairs. This is a timing prior for online SEA7, not an actor-specific measured guarantee; report actual speeds.'))
    pool_dir=SETUP/(NAME+'_pool')
    build_pool(SimpleNamespace(dataset=CACHE,prompt_registry=REGISTRY,
        eval_prompts=REFERENCE/'eval_prompts.jsonl',output_dir=pool_dir))
    pool=prompts(pool_dir/'train_prompts.jsonl')
    base_seconds=statistics.mean(r['baseline_seconds'] for r in evidence)
    quality_seconds=statistics.mean(quality)
    train_log=EXP_ROOT/'ours21_random3000_12groups_v1_sea7_train/epoch_metrics.jsonl'
    epochs=[json.loads(line) for line in train_log.read_text().splitlines()]
    epoch_seconds=epochs[-1]['elapsed_seconds']/len(epochs)
    prior=EXP_ROOT/'ours21_dynamics128_e391_vbench50_random42_4gpu_v1'
    init_seconds=statistics.mean(read(prior/'shards'/f'gpu{g}'/'baseline/run.json')['pipeline_init_seconds'] for g in range(4))
    warmup=base_seconds+init_seconds
    by_k={e['skip_budget']:e['mean_candidate_seconds'] for e in entries}
    measured_eval={}
    for k in (23,29,35):
        values=[]
        for g in range(4):values.extend(read(prior/'shards'/f'gpu{g}'/f'K{k}'/'components.json')['rows'])
        measured_eval[k]=statistics.mean(r['generate_seconds'] for r in values)
    cases=[]
    for rounds in range(1,5):
        c=OnlineConfig(rounds=rounds);slots=balanced_training_slots(pool,4,c)
        seen=set();round_rows=[];total=0
        for r in range(1,rounds+1):
            plan=make_plan(pool,r,4,lambda t:resolve_budget(target_speedup=t,calibration=SETUP/'calibration.json'),c,gpu_slots=slots)
            work=[0.]*4;new=[0]*4;counts=[0]*4
            for row in plan:
                g=row['slot'];sid=row['sample_id'];counts[g]+=1
                work[g]+=by_k[row['skip_budget']]+4. # MP4/trace export, outside reported inference
                if sid not in seen:work[g]+=base_seconds+4.;new[g]+=1;seen.add(sid)
            # Model loading and native warmup are serialized, generation overlaps later worker starts.
            generation=max((g+1)*warmup+work[g] for g in range(4))
            quality_wall=math.ceil(100/4)*quality_seconds+20.
            training=25*epoch_seconds*(120000+5000*r)/120000+25.
            subtotal=generation+quality_wall+training
            round_rows.append(dict(round=r,trajectories=counts,new_baselines=new,
                generation_seconds=generation,quality_seconds=quality_wall,training_seconds=training,
                subtotal_seconds=subtotal))
            total+=subtotal
        # One combined generation dispatch per evaluation; offline start generated only at first evaluation.
        eval_counts=defaultdict(int)
        for row in read(REFERENCE/'manifest.json')['rows']:eval_counts[row['baseline_gpu_uuid']]+=1
        eval_seconds=[]
        for i,r in enumerate(c.evaluation_rounds()):
            policies=2 if i==0 else 1
            generation=4*warmup+max(eval_counts.values())*policies*sum(measured_eval[k]+4. for k in (23,29,35))
            quality_wall=3*policies*(math.ceil(20/4)*quality_seconds+20.)
            value=generation+quality_wall;eval_seconds.append(dict(round=r,seconds=value));total+=value
        cases.append(dict(rounds=rounds,total_hours=total/3600,round_details=round_rows,evaluations=eval_seconds))
    eligible=[x for x in cases if x['total_hours']<=8.]
    selected=max(eligible,key=lambda x:x['rounds']) if eligible else cases[0]
    dump(SETUP/'budget.json',dict(budget_hours=8.,selected_rounds=selected['rounds'],
        selected_estimated_hours=selected['total_hours'],cases=cases,
        baseline_seconds=base_seconds,quality_seconds_per_video=quality_seconds,
        offline_epoch_seconds=epoch_seconds,model_init_seconds=init_seconds,
        sources={str(p):sha256(p) for p in [train_log,SETUP/'calibration.json',SETUP/'calibration_evidence.json']},
        assumptions='Four otherwise idle GPUs; native loads/warmups serialized, balanced prompt slots, four-GPU quality, 4s export/video. Includes final evaluation; runtime estimate, not a hard eight-hour termination.'))
    refresh_evaluation_estimate()


def refresh_evaluation_estimate():
    """Use the original per-GPU evaluation counts and staggered worker starts."""
    budget=read(SETUP/'budget.json')
    prior=EXP_ROOT/'ours21_dynamics128_e391_vbench50_random42_4gpu_v1'
    counts=defaultdict(int)
    for row in read(REFERENCE/'manifest.json')['rows']:counts[row['baseline_gpu_uuid']]+=1
    warmup=budget['baseline_seconds']+budget['model_init_seconds']
    for case in budget['cases']:
        for i,entry in enumerate(case['evaluations']):
            policies=2 if i==0 else 1;gpu_finish=[]
            for g in range(4):
                source=prior/'shards'/f'gpu{g}'/'baseline/run.json'
                uuid=read(source)['gpu_uuid'];budget['sources'][str(source)]=sha256(source)
                candidate=0.
                for k in (23,29,35):
                    source=prior/'shards'/f'gpu{g}'/f'K{k}'/'components.json'
                    rows=read(source)['rows'];budget['sources'][str(source)]=sha256(source)
                    candidate+=statistics.mean(r['generate_seconds'] for r in rows)+4.
                gpu_finish.append((g+1)*warmup+counts[uuid]*policies*candidate)
            quality=3*policies*(math.ceil(20/4)*budget['quality_seconds_per_video']+20.)
            entry.update(seconds=max(gpu_finish)+quality,gpu_finish_seconds=gpu_finish)
        case['total_hours']=(sum(r['subtotal_seconds'] for r in case['round_details'])+
                             sum(e['seconds'] for e in case['evaluations']))/3600
    eligible=[case for case in budget['cases'] if case['total_hours']<=budget['budget_hours']]
    selected=max(eligible,key=lambda case:case['rounds'])
    budget.update(selected_rounds=selected['rounds'],selected_estimated_hours=selected['total_hours'],
        uncertainty='Initialization lock order, CPU/IO contention and GPU temperature can shift wall time; estimate is not a deadline guarantee.')
    dump(SETUP/'budget.json',budget)
    print(json.dumps(dict(selected_rounds=selected['rounds'],estimated_hours=selected['total_hours'],
                         cases=[(x['rounds'],x['total_hours']) for x in budget['cases']])))


if __name__=='__main__':
    if '--refresh-estimate' in sys.argv:refresh_evaluation_estimate()
    else:main()
