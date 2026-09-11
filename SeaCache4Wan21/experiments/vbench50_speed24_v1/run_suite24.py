"""Fixed calibrated SeaCache nominal2.4x on the existing random50, four GPUs."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OFFICIAL = PROJECT.parent
sys.path.insert(0, str(OFFICIAL / 'Ours4Wan21/experiments/vbench50_speed36_pair_v1'))
import run_pair as pair
prior = pair.prior
ROOT = pair.EXP / 'seacache_wan21_vbench50_speed24_4gpu_v1'
WORKER = pair.HERE / 'sea_worker.py'
read, dump, sha, now = pair.read, pair.dump, pair.sha, pair.now


def threshold_for(mapping, target=2.4):
    if mapping['calibration_status'] != 'calibrated':
        raise ValueError('uncalibrated mapping')
    xs, ys = mapping['mapping']['speedups'], mapping['mapping']['mean_thresholds']
    for a,b,c,d in zip(xs,xs[1:],ys,ys[1:]):
        if a <= target <= b:
            return c+(d-c)*(target-a)/(b-a)
    raise ValueError('target outside calibrated domain')


def validate_sources(cfg):
    for path, digest in {**cfg['source_sha256'], **cfg['baseline_sha256']}.items():
        if sha(Path(path)) != digest:
            raise ValueError(f'frozen source changed: {path}')
    for gpu, ids in cfg['shard_ids'].items():
        prompts = [json.loads(s) for s in (ROOT/'prompts'/f'gpu{gpu}.jsonl').read_text().splitlines()]
        if [r['sample_id'] for r in prompts] != ids:
            raise ValueError('prompt shard mismatch')
        for sid in ids:
            ref = ROOT/'references'/f'gpu{gpu}'/'videos'/f'{sid}.mp4'
            expected = Path(cfg['previous'])/'shards'/f'gpu{gpu}'/'baseline/videos'/f'{sid}.mp4'
            if ref.resolve() != expected.resolve():
                raise ValueError('reference symlink mismatch')


def prepare(resume=False):
    if resume:
        cfg = read(ROOT/'config.json'); validate_sources(cfg); return cfg
    old = read(pair.ROOT/'config.json')
    done = read(pair.ROOT/'COMPLETE.json')
    if done['status'] != 'complete' or sha(pair.ROOT/'analysis/results.csv') != done['results_sha256']:
        raise ValueError('previous pair experiment incomplete')
    for p,h in {**old['source_sha256'], **old['baseline_sha256']}.items():
        if sha(Path(p)) != h: raise ValueError(f'prior source changed: {p}')
    mapping_path = pair.EXP/'wan21_seacache_speedup_calibration_v1/analysis/speed_threshold_mapping.calibrated.json'
    threshold = threshold_for(read(mapping_path))
    previous = Path(old['previous'])
    rows = [json.loads(x) for x in (previous/'prompts/selected.jsonl').read_text().splitlines()]
    shards = prior.partition(rows)
    sources = dict(old['source_sha256'])
    for path in (Path(__file__), WORKER, Path(pair.__file__), mapping_path, pair.ROOT/'config.json', pair.ROOT/'COMPLETE.json'):
        sources[str(path)] = sha(path)
    cfg = dict(schema='seacache_random50_speed24_v1', previous=str(previous), protocol=prior.PROTOCOL,
               target=2.4, threshold=threshold, prompt_count=50, gpu_uuids=old['gpu_uuids'],
               shard_ids=old['shard_ids'], flops_profile=old['flops_profile'],
               baseline_sha256=old['baseline_sha256'], source_sha256=sources,
               vbench_enabled=False, metrics=list(prior.METRICS),
               calibration_policy='reuse previously calibrated2.4 mapping, fixed threshold, no adaptive rescan')
    if cfg['shard_ids'] != {str(g):[r['sample_id'] for r in rs] for g,rs in shards.items()}:
        raise ValueError('original partition mismatch')
    prior.create_result(ROOT, '# SeaCache nominal2.4x random50\n\nFrozen same-GPU references; prompts/, references/, shards/, analysis/, logs/. Only VideoMetrics PSNR/SSIM/LPIPS.\n')
    link = PROJECT/'experiment_results'/ROOT.name
    if not link.is_symlink(): link.symlink_to(ROOT, target_is_directory=True)
    dump(ROOT/'config.json',cfg)
    for name in ('prompts','references','shards','analysis','logs'):
        (ROOT/name).mkdir(); (ROOT/name/'README.md').write_text(f'# {name}\n\nSeaCache nominal2.4x random50 {name}; see ../README.md.\n')
    for g, rs in shards.items():
        (ROOT/'prompts'/f'gpu{g}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
        refs=ROOT/'references'/f'gpu{g}'; refs.mkdir(); (refs/'videos').mkdir()
        (refs/'README.md').write_text('# Same-GPU baseline\n\nNative video symlinks, no reference regeneration.\n')
        for r in rs:
            (refs/'videos'/(r['sample_id']+'.mp4')).symlink_to(previous/'shards'/f'gpu{g}'/'baseline/videos'/(r['sample_id']+'.mp4'))
    validate_sources(cfg)
    dump(ROOT/'status.json', dict(status='prepared',updated_at=now(),vbench_status='skipped_by_user'))
    return cfg


def finalize(cfg):
    base=[]; cand=[]; qs=[]; details=[]; evidence={}
    for g in prior.GPUS:
        ids=cfg['shard_ids'][str(g)]; d=ROOT/'shards'/f'gpu{g}'/'seacache'
        b=read(Path(cfg['previous'])/'shards'/f'gpu{g}'/'baseline/components.json')['rows']
        c=pair.check_generation(d,g,'seacache',ids,cfg); q=prior.quality_rows(d/'quality',ids)
        base+=b; cand += [dict(r,**{k:0. for k in prior.FIELDS[7:] if k not in r}) for r in c]; qs+=q
        byb={r['sample_id']:r for r in b}; byc={r['sample_id']:r for r in c}
        for row in q:
            sid=row['video_id']; details.append(dict(sample_id=sid,gpu=g,baseline_seconds=byb[sid]['generate_seconds'],
                candidate_seconds=byc[sid]['generate_seconds'],**{m:float(row[m+'_mean']) for m in prior.METRICS}))
        for file in ('run.json','components.json','COMPLETE.json','quality/summary.json','quality/per_video.csv'):
            evidence[str((d/file).relative_to(ROOT))]=sha(d/file)
    if len(details)!=50 or len({r['sample_id'] for r in details})!=50: raise ValueError('need50 pairs')
    result=dict(method='seacache',target=2.4,threshold=cfg['threshold'],**prior.summarize(base,cand))
    for m in prior.METRICS:
        values=[float(q[m+'_mean']) for q in qs]; result[m]=statistics.fmean(values); result[m+'_std_across_prompts']=statistics.stdev(values)
    validate_sources(cfg)
    out=ROOT/'analysis'; prior.write_csv(out/'results.csv',[result]); prior.write_csv(out/'per_video.csv',details)
    dump(out/'VALIDATION.json',dict(status='pass',candidates=50,baseline_reused=50,quality_frames=4050,
        config_sha256=sha(ROOT/'config.json'),evidence_sha256=evidence,vbench_status='skipped_by_user'))
    (out/'RESULTS.md').write_text('# SeaCache 同50prompt名义2.4×\n\n同GPU baseline、固定标定阈值；无VBench或自适应补测。\n\n'
        '|实测加速|秒/视频|DiT TFLOPs|PSNR|SSIM|LPIPS|\n|---:|---:|---:|---:|---:|---:|\n'
        +f"|{result['latency_speedup']:.4f}|{result['candidate_generate_seconds_mean']:.3f}|{result['candidate_dit_tflops_mean']:.3f}|{result['psnr_rgb_db']:.4f}|{result['ssim_rgb']:.5f}|{result['lpips_alex_v0_1_spatial']:.5f}|\n")
    dump(ROOT/'COMPLETE.json',dict(status='complete',completed_at=now(),candidates=50,baseline_reused=50,
        results_sha256=sha(out/'results.csv'),validation_sha256=sha(out/'VALIDATION.json'),vbench_status='skipped_by_user'))


def execute(cfg):
    state=dict(status='waiting_for_gpus',stages={},vbench_status='skipped_by_user'); lock=threading.Lock()
    def update(**kw):
        with lock: state.update(kw,updated_at=now()); dump(ROOT/'status.json',state)
    env=dict(os.environ,OURS4WAN21_WORKSPACE=str(pair.WORKSPACE),OURS4WAN21_EXP_BASE=str(pair.EXP),EXP_BASE=str(pair.EXP),
        CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',TORCH_HOME=str(prior.MODEL_ROOT/'torch-cache'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    def run(name, cmd, g):
        with lock: state['stages'][name]=dict(status='running',started_at=now()); dump(ROOT/'status.json',state)
        with (ROOT/'logs'/f'{name}.log').open('ab') as log:
            subprocess.run(cmd,cwd=pair.PROJECT,env=dict(env,CUDA_VISIBLE_DEVICES=cfg['gpu_uuids'][str(g)]),stdout=log,stderr=subprocess.STDOUT,check=True)
        with lock: state['stages'][name].update(status='complete',completed_at=now()); dump(ROOT/'status.json',state)
    def generation(g):
        parent=ROOT/'shards'/f'gpu{g}'; parent.mkdir(exist_ok=True)
        (parent/'README.md').write_text('# GPU shard\n\nseacache/: candidate videos, trace, timing, quality.\n')
        d=parent/'seacache'
        if not d.exists():
            run(f'generate_gpu{g}',[sys.executable,str(WORKER),'--config',str(ROOT/'config.json'),
                '--prompts',str(ROOT/'prompts'/f'gpu{g}.jsonl'),'--output-dir',str(d),'--result-parent',str(ROOT),'--gpu',str(g)],g)
            (d/'README.md').write_text('# SeaCache nominal2.4x\n\nrun.json freezes settings; videos/traces/timings/components.json. Native warmup excluded.\n')
        pair.check_generation(d,g,'seacache',cfg['shard_ids'][str(g)],cfg)
    def quality(g):
        d=ROOT/'shards'/f'gpu{g}'/'seacache'; q=d/'quality'
        if not q.exists(): run(f'quality_gpu{g}',[sys.executable,str(OFFICIAL/'VideoMetrics/evaluate.py'),
            '--reference-dir',str(ROOT/'references'/f'gpu{g}'/'videos'),'--candidate-dir',str(d/'videos'),
            '--expected-frames','81','--device','cuda:0','--output-dir',str(q)],g)
        prior.quality_rows(q,cfg['shard_ids'][str(g)])
    try:
        update()
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(): time.sleep(30)
        validate_sources(cfg); update(status='generating')
        with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(generation,prior.GPUS))
        update(status='video_quality')
        with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(quality,prior.GPUS))
        update(status='analyzing'); finalize(cfg); update(status='complete')
    except BaseException as exc:
        update(status='failed',error=repr(exc)); dump(ROOT/'FAILED.json',dict(error=repr(exc),failed_at=now())); raise


def main():
    p=argparse.ArgumentParser(); p.add_argument('--resume',action='store_true'); p.add_argument('--prepare-only',action='store_true'); args=p.parse_args()
    if 'wan2.2' not in Path(sys.prefix).name.lower(): raise ValueError('use Wan2.2 environment')
    cfg=prepare(args.resume)
    if args.prepare_only: print(json.dumps(dict(root=str(ROOT),threshold=cfg['threshold'],prompts=50))); return
    with (ROOT/'queue.lock').open('a') as local,(pair.EXP/'wan21_benchmark_4gpu.lock').open('a') as shared:
        fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'COMPLETE.json').exists(): return
        fcntl.flock(shared,fcntl.LOCK_EX); execute(cfg)


if __name__=='__main__': main()
