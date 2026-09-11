#!/usr/bin/env python3
"""Frozen random-50 Dynamics128 e391 benchmark, four GPUs, fixed Exact-K."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime, timezone
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

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'experiments/vbench5_speed_targets_v1'))
from ours4wan21.contracts import EXP_ROOT, MODEL_ROOT, OFFICIAL, PROTOCOL, WORKSPACE, create_result, dump, sha256
from pipeline_lib import read_json, validate_generation

NAME = 'ours21_dynamics128_e391_vbench50_random42_4gpu_v1'
MODE = 'sea7_dynamics_raw_sea128'
GPUS, KS, TARGETS = (0,1,2,3), (23,29,35), (1.8,2.4,3.0)
LABELS = ('baseline','K23','K29','K35')
METRICS = ('psnr_rgb_db','ssim_rgb','lpips_alex_v0_1_spatial')
FIELDS = ('generate_seconds','dit_cuda_seconds','t5_cuda_seconds','vae_decode_cuda_seconds',
          'dit_tflops','estimated_t5_tflops_per_video','estimated_vae_decode_tflops_per_video',
          'predictor_tflops','predictor_network_cuda_seconds','predictor_decision_wall_seconds',
          'latent_feature_wall_seconds')


def now():
    return datetime.now(timezone.utc).isoformat()


def select_prompts(rows):
    if len(rows)!=200 or len({r['sample_id'] for r in rows})!=200 or len({r['prompt_en'] for r in rows})!=200:
        raise ValueError('need 200 unique source IDs and prompts')
    return sorted(random.Random(42).sample(rows,50),key=lambda r:r['sample_id'])


def partition(rows):
    if len(rows)!=50 or len({r['sample_id'] for r in rows})!=50:
        raise ValueError('need 50 unique selected prompts')
    return {g:rows[g::4] for g in GPUS}


def quality_rows(path, ids):
    with (path/'per_video.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    summary=read_json(path/'summary.json')
    if len(rows)!=len(ids) or {r['video_id'] for r in rows}!=set(ids):
        raise ValueError('quality coverage or duplicate mismatch')
    if summary['video_count']!=len(ids) or summary['frame_count_total']!=81*len(ids):
        raise ValueError('quality count mismatch')
    for row in rows:
        if (int(row['frames']),int(row['height']),int(row['width']))!=(81,480,832):
            raise ValueError('quality video shape mismatch')
        for m in METRICS:
            if not math.isfinite(float(row[m+'_mean'])):
                raise ValueError('nonfinite quality')
        for side in ('reference','candidate'):
            if sha256(Path(row[side]))!=row[side+'_sha256']:
                raise ValueError('quality video SHA mismatch')
    for m in METRICS:
        if not math.isclose(statistics.fmean(float(r[m+'_mean']) for r in rows),summary['metrics'][m]['mean'],abs_tol=1e-10):
            raise ValueError('quality mean mismatch')
    return rows


def summarize(base, candidates):
    if len(base)!=50 or len(candidates)!=50 or len({r['sample_id'] for r in base})!=50 or len({r['sample_id'] for r in candidates})!=50:
        raise ValueError('need 50 unique paired performance rows')
    if {r['sample_id'] for r in base}!={r['sample_id'] for r in candidates}:
        raise ValueError('performance identity mismatch')
    # Native baseline has no predictor/latent-feature fields by design.
    base=[dict(r,**{k:0.0 for k in FIELDS[7:] if k not in r}) for r in base]
    for r in base+candidates:
        if any(not math.isfinite(r[k]) or r[k]<0 for k in FIELDS) or r['generate_seconds']<=0 or r['dit_tflops']<=0:
            raise ValueError('invalid performance')
    out={f'{side}_{k}_mean':statistics.fmean(r[k] for r in rows)
         for side,rows in [('baseline',base),('candidate',candidates)] for k in FIELDS}
    out['latency_speedup']=sum(r['generate_seconds'] for r in base)/sum(r['generate_seconds'] for r in candidates)
    out['dit_tflops_speedup']=sum(r['dit_tflops'] for r in base)/sum(r['dit_tflops'] for r in candidates)
    return out


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def vbench_enabled(root):
    marker=root/'VBENCH_SKIPPED_BY_USER.json'
    if not marker.exists():return True
    scope=read_json(marker)
    if scope.get('status')!='skipped_by_user' or scope.get('video_metrics_enabled') is not True:
        raise ValueError('invalid user evaluation scope')
    return False


def validate_resume_config(root,current):
    frozen=read_json(root/'config.json')
    if not vbench_enabled(root):
        scope=read_json(root/'VBENCH_SKIPPED_BY_USER.json')
        script=str(Path(__file__))
        if (frozen['source_sha256'][script]!=scope['previous_runner_sha256'] or
                current['source_sha256'][script]!=scope['updated_runner_sha256']):
            raise ValueError('scope-change runner hash mismatch')
        # Only the explicitly authorized orchestrator revision is exempted.
        current={**current,'source_sha256':{**current['source_sha256'],script:scope['previous_runner_sha256']}}
    if frozen!=current:raise ValueError('frozen inputs changed')


def freeze(resume=False):
    root=EXP_ROOT/NAME
    prompts_path=OFFICIAL/'Vbench200/prompts.jsonl'
    full_path=OFFICIAL/'Vbench200/VBench200_full_info.json'
    selected=select_prompts([json.loads(x) for x in prompts_path.read_text().splitlines() if x.strip()])
    shards=partition(selected)
    prompts={r['prompt_en'] for r in selected}
    subset=[r for r in json.loads(full_path.read_text()) if r['prompt_en'] in prompts]
    counts=Counter(d for r in subset for d in r['dimension'])
    dims=OFFICIAL/'VbenchEvaluation/dimensions.json'
    if {r['prompt_en'] for r in subset}!=prompts or set(counts)!=set(read_json(dims)['dimensions']):
        raise ValueError('random draw lacks full metadata/dimension coverage; do not silently redraw')
    selection_path=EXP_ROOT/f'ours21_random3000_12groups_v1_{MODE}_analysis/checkpoint_selection.json'
    selection=read_json(selection_path)
    checkpoint=Path(selection['checkpoint']).resolve(strict=True)
    if selection['checkpoint_epoch']!=391 or sha256(checkpoint)!=selection['checkpoint_sha256']:
        raise ValueError('selected Dynamics128 e391 checkpoint mismatch')
    profile_path=EXP_ROOT/'wan21_seacache_threshold_collection_v1/calflops_profile.json'
    profile=read_json(profile_path)
    if profile['input']['video_shape_fhw']!=[81,480,832] or profile['input']['transformer_blocks']!=30:
        raise ValueError('FLOPs profile mismatch')
    sources=[prompts_path,full_path,dims,selection_path,profile_path,Path(__file__),
             PROJECT/'ours4wan21/inference.py',PROJECT/'ours4wan21/runtime.py',
             OFFICIAL/'VideoMetrics/video_metrics/core.py',OFFICIAL/'VideoMetrics/evaluate.py',
             OFFICIAL/'VideoMetrics/video_metrics/evaluator.py',OFFICIAL/'VideoMetrics/video_metrics/video.py',
             OFFICIAL/'VbenchEvaluation/evaluate_vbench.py',OFFICIAL/'VbenchEvaluation/prepare_videos.py',
             OFFICIAL/'VbenchEvaluation/aggregate_vbench_scores.py']
    config=dict(schema='dynamics128_vbench50_4gpu_v1',mode=MODE,protocol=PROTOCOL,
        checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),selected_epoch=391,
        sampling_seed=42,sampling_algorithm='Python random.Random(42).sample(source_order, 50); then sort sample_id',
        source_count=200,prompt_count=50,selected_ids=[r['sample_id'] for r in selected],
        original_five_overlap=sorted({r['sample_id'] for r in selected}&{f'vbench200_{s}' for s in ('001','016','056','135','159')}),
        dimension_record_counts=dict(counts),shard_ids={str(g):[r['sample_id'] for r in rs] for g,rs in shards.items()},
        gpu_ids=list(GPUS),skip_budgets=list(KS),targets=list(TARGETS),adaptive_rescan=False,
        vbench_enabled=True,vbench_scope='random 50-prompt subset of VBench200; not full leaderboard',
        flops_profile=str(profile_path),source_sha256={str(p):sha256(p) for p in sources})
    if resume:
        validate_resume_config(root,config)
    else:
        create_result(root,'# Dynamics128 e391 random VBench50\n\nSee config.json, prompts/, shards/, merged/, vbench/, analysis/, logs/ and status.json. Native50 + candidates150; fixed K23/K29/K35.')
        for name in ('prompts','shards','merged','vbench','analysis','logs'):
            (root/name).mkdir();(root/name/'README.md').write_text(f'# {name}\n\nDynamics128 random-50 suite {name}; see ../README.md.\n')
        dump(root/'config.json',config)
        for name,rs in [('selected',selected)]+[(f'gpu{g}',rs) for g,rs in shards.items()]:
            (root/'prompts'/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
        dump(root/'prompts/full_info.json',subset)
        dump(root/'status.json',dict(status='prepared',updated_at=now()))
    for name,rs in [('selected',selected)]+[(f'gpu{g}',rs) for g,rs in shards.items()]:
        if [json.loads(x) for x in (root/'prompts'/f'{name}.jsonl').read_text().splitlines()]!=rs:
            raise ValueError('frozen prompt file changed')
    if json.loads((root/'prompts/full_info.json').read_text())!=subset:
        raise ValueError('frozen full-info changed')
    return root,config


def run(root,config):
    state=dict(status='waiting_for_gpus',stages={},updated_at=now());lock=threading.Lock()
    def update(**values):
        with lock:
            state.update(values,updated_at=now());dump(root/'status.json',state)
    env=dict(os.environ,OURS4WAN21_WORKSPACE=str(WORKSPACE),OURS4WAN21_EXP_BASE=str(EXP_ROOT),
        EXP_BASE=str(EXP_ROOT),PYTHON_BIN=sys.executable,PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',
        TORCH_HOME=str(MODEL_ROOT/'torch-cache'),VBENCH_CACHE_DIR=str(MODEL_ROOT/'VBench'),
        HF_HOME=str(MODEL_ROOT/'VBench/huggingface'),XDG_CACHE_HOME=str(MODEL_ROOT/'VBench/xdg'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    def execute(label,cmd,g):
        with lock:
            state['stages'][label]=dict(status='running',gpu=g,started_at=now());dump(root/'status.json',state)
        with (root/'logs'/f'{label}.log').open('ab') as log:
            subprocess.run(cmd,cwd=PROJECT,env=dict(env,CUDA_VISIBLE_DEVICES=str(g)),stdout=log,stderr=subprocess.STDOUT,check=True)
        with lock:
            state['stages'][label].update(status='complete',completed_at=now());dump(root/'status.json',state)
    def parallel(fn,values):
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fn,values))
    def folder(g,label):return root/'shards'/f'gpu{g}'/label
    def generation(g):
        parent=root/'shards'/f'gpu{g}';parent.mkdir(exist_ok=True)
        (parent/'README.md').write_text('# GPU shard\n\nMatched native baseline and K23/K29/K35 generation, timing, trace and quality.\n')
        ids=config['shard_ids'][str(g)]
        for label in LABELS:
            d=folder(g,label);k=None if label=='baseline' else int(label[1:])
            cmd=[sys.executable,'generate.py','--wan21-root',str(WORKSPACE/'data/source/Wan2.1-65386b2'),
                '--checkpoint-dir',str(MODEL_ROOT/'Wan2.1-T2V-1.3B'),'--prompts',str(root/'prompts'/f'gpu{g}.jsonl'),
                '--flops-profile',config['flops_profile'],'--output-dir',str(d),'--result-parent',str(root)]
            cmd+=['--baseline'] if k is None else ['--policy-checkpoint',config['checkpoint'],'--state-mode',MODE,'--skip-budget',str(k)]
            if not d.exists():execute(f'generate_gpu{g}_{label}',cmd,g)
            m=validate_generation(d,prompt_ids=ids,method='baseline' if k is None else 'ours',mode=MODE,skip_budget=k)
            if m['protocol']!=PROTOCOL or m['flops_profile_sha256']!=config['source_sha256'][config['flops_profile']]:
                raise ValueError('generation protocol/profile mismatch')
            if k is not None:
                bm=read_json(folder(g,'baseline')/'run.json')
                if bm['gpu_uuid']!=m['gpu_uuid'] or m['policy_sha256']!=config['checkpoint_sha256']:
                    raise ValueError('physical GPU/checkpoint mismatch')
                for sid in ids:
                    trace=read_json(d/'traces'/f'{sid}.json')
                    if trace['step_reuse']!=k or trace['total_steps']!=50:raise ValueError('Exact-K mismatch')
                if not (d/'performance.json').exists():
                    execute(f'performance_gpu{g}_{label}',[sys.executable,'evaluate.py','summarize','--baseline-dir',str(folder(g,'baseline')),
                        '--candidate-dir',str(d),'--profile',config['flops_profile']],g)
    def quality(g):
        for k in KS:
            d=folder(g,f'K{k}');q=d/'quality'
            if not q.exists():
                execute(f'quality_gpu{g}_K{k}',[sys.executable,str(OFFICIAL/'VideoMetrics/evaluate.py'),
                    '--reference-dir',str(folder(g,'baseline')/'videos'),'--candidate-dir',str(d/'videos'),
                    '--expected-frames','81','--device','cuda:0','--output-dir',str(q)],g)
            quality_rows(q,config['shard_ids'][str(g)])
    def vbench(job):
        g,label=job;out=root/'vbench'/label;out.mkdir(exist_ok=True)
        (out/'README.md').write_text('# VBench50 condition\n\nStaged prompt-named symlinks, scores/, aggregate.json, scope.json and COMPLETE.json.\n')
        if not (out/'COMPLETE.json').exists():
            vb=OFFICIAL/'VbenchEvaluation'
            execute(f'vbench_prepare_{label}',[sys.executable,str(vb/'prepare_videos.py'),'--videos-dir',str(root/'merged'/label/'videos'),
                '--staging-dir',str(out/'staged_videos'),'--manifest',str(out/'staging_manifest.json'),
                '--prompts-jsonl',str(root/'prompts/selected.jsonl'),'--expected-seeds','1'],g)
            execute(f'vbench_score_{label}',[sys.executable,str(vb/'evaluate_vbench.py'),'--videos-dir',str(out/'staged_videos'),
                '--output-dir',str(out/'scores'),'--full-info',str(root/'prompts/full_info.json'),
                '--name-prefix','vbench50','--load-ckpt-from-local'],g)
            execute(f'vbench_aggregate_{label}',[sys.executable,str(vb/'aggregate_vbench_scores.py'),'--score-dir',str(out/'scores'),
                '--output',str(out/'aggregate.json'),'--label','Dynamics128 random VBench50 '+label],g)
            dump(out/'scope.json',dict(prompt_count=50,source_count=200,sampling_seed=42,official_full_vbench_score=False,
                full_info_sha256=sha256(root/'prompts/full_info.json'),dimensions=16))
            dump(out/'COMPLETE.json',dict(status='complete',aggregate_sha256=sha256(out/'aggregate.json')))
        scores=read_json(out/'aggregate.json')
        if (set(scores['raw_dimension_scores'])!=set(config['dimension_record_counts']) or scores['official_full_vbench_score'] is not False
                or read_json(out/'COMPLETE.json')['aggregate_sha256']!=sha256(out/'aggregate.json')):
            raise ValueError('VBench score scope/hash mismatch')
    try:
        update(status='waiting_for_gpus')
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():time.sleep(30)
        update(status='generating');parallel(generation,GPUS)
        update(status='video_quality');parallel(quality,GPUS)
        analyze(root,config,folder,require_vbench=False)
        if not vbench_enabled(root):
            update(status='analyzing',vbench_status='skipped_by_user')
            analyze(root,config,folder,require_vbench=False,final=True)
            dump(root/'COMPLETE.json',dict(status='complete',completed_at=now(),prompts=50,
                baseline_videos=50,candidate_videos=150,vbench_status='skipped_by_user',
                scope_sha256=sha256(root/'VBENCH_SKIPPED_BY_USER.json'),
                results_sha256=sha256(root/'analysis/results.csv'),validation_sha256=sha256(root/'analysis/VALIDATION.json')))
            update(status='complete',vbench_status='skipped_by_user')
            return
        for label in LABELS:
            parent=root/'merged'/label;videos=parent/'videos';videos.mkdir(parents=True,exist_ok=True)
            (parent/'README.md').write_text('# Merged videos\n\nExactly 50 symlinks to the corresponding per-GPU condition.\n')
            for g in GPUS:
                for sid in config['shard_ids'][str(g)]:
                    src=folder(g,label)/'videos'/f'{sid}.mp4';dst=videos/src.name
                    if dst.is_symlink() and dst.resolve()==src.resolve():continue
                    if dst.exists() or dst.is_symlink():raise ValueError('merged video conflict')
                    dst.symlink_to(src)
            if {p.stem for p in videos.glob('*.mp4')}!=set(config['selected_ids']):raise ValueError('merged video coverage mismatch')
        update(status='vbench');parallel(vbench,list(zip(GPUS,LABELS)))
        update(status='analyzing');analyze(root,config,folder,require_vbench=True)
        dump(root/'COMPLETE.json',dict(status='complete',completed_at=now(),prompts=50,baseline_videos=50,candidate_videos=150,
            results_sha256=sha256(root/'analysis/results.csv'),validation_sha256=sha256(root/'analysis/VALIDATION.json')))
        update(status='complete')
    except BaseException as error:
        update(status='failed',error=repr(error));dump(root/'FAILED.json',dict(error=repr(error),failed_at=now()));raise


def analyze(root,config,folder,require_vbench,final=False):
    results=[];per_video=[];evidence={}
    for k,target in zip(KS,TARGETS):
        base=[];candidate=[];quality=[]
        for g in GPUS:
            d=folder(g,f'K{k}');p=read_json(d/'performance.json');qs=quality_rows(d/'quality',config['shard_ids'][str(g)])
            b={r['sample_id']:r for r in p['baseline']};c={r['sample_id']:r for r in p['candidate']}
            if len(b)!=len(p['baseline']) or len(c)!=len(p['candidate']) or set(b)!=set(c) or set(b)!=set(config['shard_ids'][str(g)]):
                raise ValueError('performance shard pairing mismatch')
            base.extend(p['baseline']);candidate.extend(p['candidate']);quality.extend(qs)
            for q in qs:
                sid=q['video_id'];per_video.append(dict(k=k,gpu=g,sample_id=sid,baseline_seconds=b[sid]['generate_seconds'],
                    candidate_seconds=c[sid]['generate_seconds'],**{m:float(q[m+'_mean']) for m in METRICS}))
            for name in ('performance.json','quality/summary.json','quality/per_video.csv'):
                evidence[str((d/name).relative_to(root))]=sha256(d/name)
        result=dict(label='Dynamics128',checkpoint_epoch=391,k=k,target_speedup=target,**summarize(base,candidate))
        for m in METRICS:
            values=[float(q[m+'_mean']) for q in quality]
            result[m]=statistics.fmean(values);result[m+'_std_across_prompts']=statistics.stdev(values)
        if require_vbench:
            scores=read_json(root/'vbench'/f'K{k}'/'aggregate.json')['aggregate_scores']
            baseline_scores=read_json(root/'vbench/baseline/aggregate.json')['aggregate_scores']
            result.update(vbench50_total_score=scores['total_score'],vbench50_quality_score=scores['quality_score'],
                vbench50_semantic_score=scores['semantic_score'],baseline_vbench50_total_score=baseline_scores['total_score'])
        results.append(result)
    if len(per_video)!=150:raise ValueError('incomplete 150 quality pairs')
    out=root/'analysis';name='results' if require_vbench or final else 'quality_results'
    write_csv(out/f'{name}.csv',results);write_csv(out/'per_video.csv',per_video)
    if require_vbench:
        for label in LABELS:evidence[f'vbench/{label}/aggregate.json']=sha256(root/'vbench'/label/'aggregate.json')
    dump(out/('VALIDATION.json' if require_vbench or final else 'QUALITY_VALIDATION.json'),dict(status='pass',conditions=3,
        baseline_videos=50,candidate_videos=150,quality_frames=12150,vbench_complete=require_vbench,
        vbench_status='complete' if require_vbench else 'pending' if vbench_enabled(root) else 'skipped_by_user',
        official_full_vbench_score=False,config_sha256=sha256(root/'config.json'),evidence_sha256=evidence))
    report=['# Dynamics128 e391 随机 VBench50', '', '从VBench200一次随机无放回抽50条，抽样seed42；生成seed42。',
        'K23/K29/K35固定不补测；同物理GPU baseline配对。PSNR先81帧均值、再50prompt等权均值。',
        '完整generate耗时排除模型加载、native warmup、保存与评测。VBench为50条子集，非全量排行榜。', '',
        '|K|名义目标|实测加速|秒/视频|PSNR dB|SSIM|LPIPS|VBench50 %|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        score=f"{100*r['vbench50_total_score']:.3f}" if require_vbench else '待评测' if vbench_enabled(root) else '按用户要求跳过'
        report.append(f"|{r['k']}|{r['target_speedup']}|{r['latency_speedup']:.4f}|{r['candidate_generate_seconds_mean']:.3f}|{r['psnr_rgb_db']:.4f}|{r['ssim_rgb']:.5f}|{r['lpips_alex_v0_1_spatial']:.5f}|{score}|")
    if not require_vbench and not vbench_enabled(root):
        report[4]='完整generate耗时排除模型加载、native warmup、保存与评测；VBench评分按用户要求跳过。'
    (out/('RESULTS.md' if require_vbench or final else 'QUALITY_RESULTS.md')).write_text('\n'.join(report)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if 'wan2.2' not in Path(sys.prefix).name.lower():raise ValueError('use Wan2.2 environment')
    root,config=freeze(args.resume)
    if args.prepare_only:
        print(json.dumps(dict(status='prepared',root=str(root),epoch=391,prompts=50,baseline_videos=50,candidate_videos=150,dimensions=config['dimension_record_counts'])));return
    with (root/'queue.lock').open('a') as local_lock, (EXP_ROOT/'wan21_benchmark_4gpu.lock').open('a') as gpu_lock:
        fcntl.flock(local_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'COMPLETE.json').exists():return
        fcntl.flock(gpu_lock,fcntl.LOCK_EX)
        run(root,config)


if __name__=='__main__':main()
