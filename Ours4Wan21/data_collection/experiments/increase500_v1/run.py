#!/usr/bin/env python3
"""Frozen increase augmentation, using the original collector and publication tables."""
import argparse
from collections import Counter
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
OFFICIAL = PROJECT.parent
EXP = Path('/mnt/hdd/xiongyuxiang/tmp/exp')
SOURCE = EXP / 'wan21_random_threshold_collection_v1_stage1'
ROOT = EXP / 'ours21_increase500_v1'
PYTHON = Path('/mnt/hdd/xiongyuxiang/tmp/data/environments/Wan2.2-conda-env/bin/python')
FAMILY = 'linear_increase_seacache_threshold'
SCALES = [.02, .04, .08, .12, .16, .20, .26, .34]
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['OURS4WAN21_EXP_BASE'] = str(EXP)
sys.path[:0] = [str(PROJECT / 'data_collection/src'), str(PROJECT), str(OFFICIAL / 'VideoMetrics')]
from ours4wan21_data.collector import atomic_json, baseline_paths, baseline_complete, candidate_paths, candidate_complete
from ours4wan21_data.manifest import PROTOCOL
from ours4wan21_data.source_lock import file_sha256 as sha


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def freeze(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'frozen input changed: {path}')
    else:
        atomic_json(path, value)


def jsonl(path, records):
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise ValueError(f'frozen manifest changed: {path}')
    if not path.exists():
        path.write_text(text)


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError(f'link mismatch: {target}')
    elif target.exists():
        raise FileExistsError(target)
    else:
        target.symlink_to(source.resolve())


def path_for(scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('positive finite increase scale required')
    return [scale * (1 + 4 * step / 49) for step in range(50)]


def make_row(source, index, scale=None, target=None, calibration=False):
    keys = ('sample_id', 'prompt', 'prompt_rank', 'split', 'shard_index', 'part',
            'content_group', 'length_group', 'motion_group', 'topic_tag')
    row = {k: source[k] for k in keys if k in source}
    row.update(schema='ours4wan21_increase_runnable_v1' if scale else 'ours4wan21_increase_plan_v1',
               release_index=index, trajectory_id=f"{source['sample_id']}__{'inc_cal' if calibration else 'increase'}{index:04d}",
               candidate_index_for_prompt=3, num_shards=4, policy_family=FAMILY,
               protocol=PROTOCOL, forced_recompute_steps=[0, 49], q=None,
               target_speedup=target, target_speedup_distribution='stratified_uniform_[1.5,3.5]',
               threshold_path=None, mean_threshold=None, manifest_seed=42,
               source_prompt_rank=source['prompt_rank'], calibration_only=calibration)
    if scale is not None:
        row.update(threshold_path=path_for(scale), mean_threshold=3*scale,
                   threshold_min=scale, threshold_max=5*scale, increase_scale=scale,
                   calibration_status='calibration_probe' if calibration else 'calibrated')
    return row


def sample_plan(prompts):
    rng = random.Random(42)
    selected, calibration = [], []
    for shard in range(4):
        for split, count in [('train', 100), ('val', 13 if shard < 2 else 12),
                             ('test', 12 if shard < 2 else 13)]:
            pool = sorted([p for p in prompts if p['split'] == split and p['shard_index'] == shard],
                          key=lambda p: p['sample_id'])
            chosen = rng.sample(pool, count + (1 if split == 'train' else 0))
            selected.extend(chosen[:count])
            if split == 'train':
                calibration.append(chosen[-1])
    # Stratification independently within split: every 0.4x bin has 80/10/10 rows.
    plan = []
    for split in ('train', 'val', 'test'):
        group = [p for p in selected if p['split'] == split]
        rng.shuffle(group)
        n = len(group)
        targets = [1.5 + 2*(i + rng.random())/n for i in range(n)]
        targets[0], targets[-1] = 1.5, 3.5
        rng.shuffle(targets)
        start = len(plan)
        plan.extend(make_row(p, start+i, target=t) for i, (p,t) in enumerate(zip(group, targets)))
    return plan, calibration


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'README.md').write_text('# Increase500 augmentation\n\nmanifests/ freezes 500 unique prompts and approximate speeds from the existing mean-threshold mapping; no new calibration. shared_baselines/ links the original 1000 references; shards/ and completed/ use the original collection contract; published/ contains standard tables; mixed/ indexes 3000 random + 500 increase without copying latents; logs/ and STATUS.json track execution.\n')
    link(ROOT, PROJECT / 'experiment_results' / ROOT.name)
    source_manifest = SOURCE / 'manifests/random_runnable.jsonl'
    original = [r for r in rows(source_manifest) if r['prompt_rank'] < 1000]
    prompts = list({r['sample_id']: r for r in original}.values())
    if len(original) != 3000 or len(prompts) != 1000:
        raise ValueError('requires original 1000-prompt / 3000-candidate stage')
    for p in prompts:
        if not baseline_complete(baseline_paths(SOURCE, p['sample_id']), p):
            raise ValueError(f"invalid source baseline: {p['sample_id']}")
    plan, calibration = sample_plan(prompts)
    jsonl(ROOT / 'manifests/plan.jsonl', plan)
    link(SOURCE / 'shared_baselines', ROOT / 'shared_baselines')
    cfg = read(SOURCE / 'shards/shard_00/candidate_config_prefix_001000.json')
    identities = {}
    for p in prompts:
        b = baseline_paths(SOURCE, p['sample_id'])
        identities[p['sample_id']] = {k: {'path': str(b[k]), 'sha256': sha(b[k])}
                                     for k in ('complete', 'video', 'timing', 'performance', 'trace')}
    freeze(ROOT / 'baseline_sources.json', identities)
    freeze(ROOT / 'config.json', dict(schema='ours21_increase500_v1', source=str(SOURCE),
        source_manifest_sha256=sha(source_manifest), source_baselines_sha256=sha(ROOT/'baseline_sources.json'),
        plan_sha256=sha(ROOT/'manifests/plan.jsonl'), new_calibration=False,
        speed_estimation='existing_mean_threshold_mapping',
        protocol=PROTOCOL, collection_config=cfg, scales=SCALES, split_counts=dict(Counter(p['split'] for p in plan)),
        count=500, seed=42, policy_family=FAMILY, vbench_required=True,
        source_hashes={str(p):sha(p) for p in [Path(__file__), PROJECT/'ours4wan21/selection.py',
            PROJECT/'ours4wan21/data.py', *sorted((PROJECT/'data_collection/src/ours4wan21_data').glob('*.py'))]}))
    print(json.dumps({'prepared':500, 'new_calibration':False, 'split_counts':dict(Counter(p['split'] for p in plan))}), flush=True)


def verify():
    cfg = read(ROOT / 'config.json')
    for file, expected in cfg['source_hashes'].items():
        if sha(Path(file)) != expected:
            raise ValueError(f'frozen implementation changed: {file}')
    for file, key in [('plan.jsonl','plan_sha256')]:
        if sha(ROOT/'manifests'/file) != cfg[key]:
            raise ValueError('manifest changed')
    if sha(ROOT/'baseline_sources.json') != cfg['source_baselines_sha256']:
        raise ValueError('baseline source inventory changed')
    if sha(SOURCE/'manifests/random_runnable.jsonl') != cfg['source_manifest_sha256']:
        raise ValueError('original manifest changed')
    return cfg


def worker(manifest, parent, shard):
    cfg = verify()['collection_config']
    selected = [r for r in rows(manifest) if r['shard_index'] == shard]
    if any(r.get('calibration_only') for r in selected):
        raise ValueError('new calibration disabled by user; only formal increase rows may run')
    from ours4wan21_data.collector import collect_one, validate_flops_profile
    from ours4wan21_data.runtime import create_pipeline, Wan21DataRuntime
    from ours4wan21_data.metrics import FullReferenceMetricEvaluator
    import torch
    if torch.cuda.device_count() != 1 or torch.cuda.mem_get_info()[0] < 40*1024**3:
        raise ValueError('requires one idle 48GB GPU with no offload')
    profile = Path(cfg['flops_profile'])
    if sha(profile) != cfg['flops_profile_sha256']:
        raise ValueError('FLOPs profile changed')
    validate_flops_profile(profile)
    sources = read(ROOT/'baseline_sources.json')
    for row in selected:
        for record in sources[row['sample_id']].values():
            if sha(Path(record['path'])) != record['sha256']:
                raise ValueError('baseline changed')
    pending = [r for r in selected if not candidate_complete(candidate_paths(parent, shard, r['trajectory_id']))]
    if not pending:
        return
    with (ROOT/'initialization.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pipe, elapsed, cache_video = create_pipeline(Path(cfg['wan21_root']), Path(cfg['checkpoint_dir']))
        # Excluded native warmup, before installing the latent capture runtime.
        with torch.no_grad():
            warm = pipe.generate(pending[0]['prompt'], size=(832,480), frame_num=81, shift=5.,
                                sample_solver='unipc', sampling_steps=50, guide_scale=5., seed=42, offload_model=False)
        del warm
        torch.cuda.synchronize()
    runtime = Wan21DataRuntime(pipe, 'candidate')
    evaluator = FullReferenceMetricEvaluator(device='cuda:0', lpips_batch_size=8,
        model_cache=Path(cfg['full_reference_metrics']['torch_home']))
    args = argparse.Namespace(mode='candidate', parent_root=parent, shard_index=shard,
                              resume=True, ffprobe_bin='ffprobe')
    for row in pending:
        with torch.no_grad():
            collect_one(args, row, pipe, runtime, elapsed, cache_video, profile, evaluator)
        print(json.dumps({'completed':row['trajectory_id']}), flush=True)


def dispatch(manifest, parent):
    children = []
    for shard in range(4):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES':str(shard)}
        log = ROOT/'logs'/f'{parent.name}_shard_{shard}.log'
        log.parent.mkdir(exist_ok=True)
        handle = log.open('a')
        proc = subprocess.Popen([str(PYTHON), '-u', str(__file__), 'worker', '--manifest', str(manifest),
                                 '--parent', str(parent), '--shard', str(shard)], env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
        handle.close()
        children.append(proc)
    codes = [p.wait() for p in children]
    if any(codes):
        raise RuntimeError(f'collection workers failed: {codes}; inspect logs; completed bundles preserved')


def interpolate(points, target):
    ordered = sorted(points, key=lambda p: p['scale'])
    if any(b['speedup'] <= a['speedup'] for a,b in zip(ordered, ordered[1:])):
        raise ValueError('calibration must be strictly increasing; do not silently fit nonmonotonic measurements')
    if not ordered[0]['speedup'] <= target <= ordered[-1]['speedup']:
        raise ValueError('calibration does not bracket target; extrapolation prohibited')
    for a,b in zip(ordered, ordered[1:]):
        if a['speedup'] <= target <= b['speedup']:
            f = (target-a['speedup'])/(b['speedup']-a['speedup'])
            return a['scale'] + f*(b['scale']-a['scale'])
    raise ValueError('unreachable interpolation')


def materialize():
    verify()
    original = rows(SOURCE/'manifests/random_runnable.jsonl')[0]
    source = Path(original['calibration_file'])
    if sha(source) != original['calibration_sha256']:
        raise ValueError('existing mean-threshold mapping changed')
    mapping = read(source)['mapping']
    points = [dict(scale=mean/3, speedup=speed) for speed,mean in
              zip(mapping['speedups'],mapping['mean_thresholds'],strict=True)]
    estimate = ROOT/'manifests/mean_threshold_speed_estimate.json'
    freeze(estimate, dict(source=str(source),source_sha256=sha(source),points=points,
        schedule='scale*(1+4*step/49)',mean_threshold='3*scale',
        speed_status='approximate_from_existing_mean_threshold_mapping',
        new_calibration=False,quality_used_for_fit=False))
    runnable = []
    for p in rows(ROOT/'manifests/plan.jsonl'):
        scale = interpolate(points, p['target_speedup'])
        r = make_row(p, p['release_index'], scale=scale, target=p['target_speedup'])
        r.update(calibration_status='estimated_from_existing_mean_threshold',
                 calibration_file=str(estimate), calibration_sha256=sha(estimate),
                 speedup_is_estimate=True, new_calibration=False)
        runnable.append(r)
    jsonl(ROOT/'manifests/increase_runnable.jsonl', runnable)


def audit_one(parent, row, profile):
    from ours4wan21_data.audit import validate_candidate
    from ours4wan21_data.publisher import load_completion
    from ours4wan21.data import load_completion as load_training, episode
    paths = candidate_paths(parent, row['shard_index'], row['trajectory_id'])
    completion = load_completion(parent, row)
    if completion is None or not candidate_complete(paths):
        raise ValueError('incomplete candidate')
    if any(completion['trajectory_row'].get(k) != v for k,v in row.items()):
        raise ValueError('candidate differs from frozen manifest')
    validate_candidate(parent, row, profile,
        read(baseline_paths(parent, row['sample_id'])['performance']), deep_latents=True)
    loaded, decisions, quality, _ = load_training(paths['complete'])
    episode(decisions, quality, 'sea7')
    calls = read(paths['timing'])['calls']
    for d, c in zip(decisions, calls):
        if c['blocks_executed'] != (0 if d['action']=='reuse' else 30):
            raise ValueError('trace/actual blocks mismatch')
        if d['requested_threshold'] != row['threshold_path'][d['step_index']]:
            raise ValueError('executed threshold differs from increase schedule')
    return loaded


def publish_and_mix():
    from ours4wan21_data.publisher import write_snapshot
    cfg = verify()
    manifest = ROOT/'manifests/increase_runnable.jsonl'
    increase = rows(manifest)
    if len(increase)!=500 or len({r['sample_id'] for r in increase})!=500:
        raise ValueError('requires exactly 500 distinct increase prompts')
    profile = read(Path(cfg['collection_config']['flops_profile']))
    measured = [audit_one(ROOT, r, profile) for r in increase]
    destination = ROOT/'published/snapshots/prefix_000000500'
    if not destination.exists():
        temp = destination.with_name(f'.building.{os.getpid()}')
        write_snapshot(increase, ROOT, temp, manifest, expected_candidate_count=500)
        temp.rename(destination)
    atomic_json(ROOT/'published/CURRENT.json', dict(schema='ours4wan21_current_publication_v3',
        snapshot=str(destination), published_candidate_count=500, published_step_count=25000,
        published_branch_transition_count=50000, complete=True, summary_sha256=sha(destination/'SUMMARY.json')))
    original = [r for r in rows(SOURCE/'manifests/random_runnable.jsonl') if r['prompt_rank']<1000]
    mix = ROOT/'mixed'
    mix.mkdir(exist_ok=True)
    (mix/'README.md').write_text('# Mixed training source\n\nselection.json and manifests/mixed_runnable.jsonl bind 3000 original random and 500 increase records, with original prompt splits. completed/ contains small completion records; videos/latents remain in source archives. Use prepare_features.py / build_dataset.py with this collection root and selection.json.\n')
    mixed_rows, selection_rows, split_by_prompt = [], [], {}
    for index, (parent, r) in enumerate([(SOURCE,r) for r in original]+[(ROOT,r) for r in increase]):
        tid, sid = r['trajectory_id'], r['sample_id']
        if sid in split_by_prompt and split_by_prompt[sid] != r['split']:
            raise ValueError('mixed source prompt leakage')
        split_by_prompt[sid] = r['split']
        source = parent/'completed'/f'{tid}.json'
        c = read(source)
        c = copy.deepcopy(c)
        c['release_index'] = index
        for table in [c['trajectory_row'], *c['step_rows'], *c['branch_rows']]:
            table['source_release_index'] = table['release_index']
            table['release_index'] = index
        c['augmentation_provenance'] = dict(source_complete=str(source), source_complete_sha256=sha(source))
        dest = mix/'completed'/f'{tid}.json'
        freeze(dest, c)
        mixed_rows.append({**r, 'source_release_index':r['release_index'], 'release_index':index})
        selection_rows.append(dict(trajectory_id=tid, sample_id=sid, split=r['split'], policy_family=r['policy_family'],
            complete_path=str(dest.relative_to(mix)), complete_sha256=sha(dest)))
    jsonl(mix/'manifests/mixed_runnable.jsonl', mixed_rows)
    freeze(mix/'selection.json', dict(schema='ours4wan21_mixed_subset_v1', seed=42,
        strategy='all_random3000_plus_increase500', selected_count=3500, source_count=3500, completed_count=3500,
        source_manifest='manifests/mixed_runnable.jsonl', source_manifest_sha256=sha(mix/'manifests/mixed_runnable.jsonl'),
        family_counts=dict(Counter(r['policy_family'] for r in mixed_rows)),
        split_counts=dict(Counter(r['split'] for r in mixed_rows)), rows=selection_rows))
    from ours4wan21.selection import selected_paths
    if len(selected_paths(mix/'selection.json', mix)) != 3500:
        raise ValueError('mixed training loader verification failed')
    atomic_json(ROOT/'DATA_COMPLETE.json', dict(candidates=500, mixed_trajectories=3500,
        split_counts=dict(Counter(r['split'] for r in increase)),
        measured_speedup_min=min(r['inference_latency_speedup'] for r in measured),
        measured_speedup_max=max(r['inference_latency_speedup'] for r in measured),
        mean_psnr=sum(r['mean_psnr'] for r in measured)/500,
        mixed_selection_sha256=sha(mix/'selection.json'), vbench_status='pending'))


def vbench():
    control = ROOT/'VBENCH_SKIPPED_BY_USER.json'
    if control.exists():
        if read(control).get('skip_vbench') is not True:
            raise ValueError('invalid scoped VBench control')
        data = read(ROOT/'DATA_COMPLETE.json')
        data['vbench_status'] = 'skipped_by_user'
        atomic_json(ROOT/'DATA_COMPLETE.json', data)
        atomic_json(ROOT/'COMPLETE.json',dict(data=data, vbench_status='skipped_by_user',
            baseline_vbench=None,candidate_vbench=None,control_sha256=sha(control)))
        return
    from ours4wan21_data.vbench import ensure_link, read_score
    records = rows(ROOT/'manifests/increase_runnable.jsonl')
    for condition in ('baseline', 'candidate'):
        quality = ROOT/'quality'
        staging = quality/'staging'/condition
        mapping = {}
        for r in records:
            path = (baseline_paths(ROOT,r['sample_id'])['video'] if condition=='baseline' else
                    candidate_paths(ROOT,r['shard_index'],r['trajectory_id'])['video'])
            name = r['trajectory_id']+'.mp4'
            ensure_link(path, staging/name)
            mapping[name]=r['prompt']
        map_path=quality/f'{condition}_prompt_map.json'
        freeze(map_path,mapping)
        out=quality/f'vbench_{condition}'
        score=out/'vbench_custom_aggregate_scores.json'
        if not score.exists():
            env={**os.environ,'PYTHON_BIN':str(PYTHON),'VBENCH_CACHE_DIR':'/mnt/hdd/xiongyuxiang/tmp/models/VBench'}
            subprocess.run(['bash',str(OFFICIAL/'VbenchEvaluation/run_custom_vbench.sh'),str(staging),str(out),str(map_path)],env=env,check=True)
        read_score(score)
    atomic_json(ROOT/'COMPLETE.json',dict(data=read(ROOT/'DATA_COMPLETE.json'),
        baseline_vbench=read_score(ROOT/'quality/vbench_baseline/vbench_custom_aggregate_scores.json'),
        candidate_vbench=read_score(ROOT/'quality/vbench_candidate/vbench_custom_aggregate_scores.json')))


def run():
    verify()
    with (ROOT/'pipeline.lock').open('a') as local, (EXP/'wan21_benchmark_4gpu.lock').open('a') as shared:
        fcntl.flock(local, fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(ROOT/'STATUS.json',dict(phase='waiting_for_gpus'))
        fcntl.flock(shared, fcntl.LOCK_EX)
        idle=0
        while idle<3:
            output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            free=all(int(a)<1000 and int(b)<5 for a,b in (line.split(',') for line in output.splitlines()))
            idle=idle+1 if free else 0
            time.sleep(20)
        for phase, operation in [('materialize',materialize),
                ('collection',lambda:dispatch(ROOT/'manifests/increase_runnable.jsonl',ROOT)),
                ('audit_and_mix',publish_and_mix),('vbench',vbench)]:
            atomic_json(ROOT/'STATUS.json',dict(phase=phase))
            operation()
        atomic_json(ROOT/'STATUS.json',dict(phase='complete'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','run','worker','materialize','finalize'])
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--shard',type=int)
    a=parser.parse_args()
    try:
        if a.phase=='worker':worker(a.manifest,a.parent,a.shard)
        else:{'prepare':prepare,'run':run,'materialize':materialize,'finalize':publish_and_mix}[a.phase]()
    except Exception as exc:
        if a.phase=='run':atomic_json(ROOT/'LAST_ERROR.json',dict(error=repr(exc)))
        raise
