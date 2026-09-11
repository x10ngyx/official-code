"""Fixed-threshold SeaCache addition to the completed Ours five-prompt matrix."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
OFFICIAL = PROJECT.parent
OURS = OFFICIAL / 'Ours4Wan21'
sys.path.insert(0, str(OURS))
from ours4wan21.contracts import EXP_ROOT, WORKSPACE, PROTOCOL, dump, sha256

ROOT = EXP_ROOT / 'seacache_wan21_vbench5_ours_comparison_v1'
PREVIOUS = EXP_ROOT / 'ours21_random3000_12groups_v1_vbench5_speed_targets_v1'
PROFILE = EXP_ROOT / 'wan21_seacache_threshold_collection_v1/calflops_profile.json'
MAPPING = EXP_ROOT / 'wan21_seacache_speedup_calibration_v1/analysis/speed_threshold_mapping.calibrated.json'
TARGETS = (1.8, 2.4, 3.0)


def read(p):
    return json.loads(p.read_text())


def threshold_for(mapping, target):
    if mapping['calibration_status'] != 'calibrated':
        raise ValueError('uncalibrated threshold mapping')
    xs, ys = mapping['mapping']['speedups'], mapping['mapping']['mean_thresholds']
    if not xs[0] <= target <= xs[-1]:
        raise ValueError('refuse extrapolation')
    for a, b, c, d in zip(xs, xs[1:], ys, ys[1:]):
        if a <= target <= b:
            return c + (d-c) * (target-a)/(b-a)
    raise ValueError('invalid mapping')


def quality_means(directory, ids):
    summary = read(directory / 'summary.json')
    with (directory / 'per_video.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 5 or {r['video_id'] for r in rows} != set(ids):
        raise ValueError('quality prompt mismatch')
    if summary['video_count'] != 5 or summary['frame_count_total'] != 405:
        raise ValueError('quality coverage mismatch')
    result = {}
    for key in ('psnr_rgb_db', 'ssim_rgb', 'lpips_alex_v0_1_spatial'):
        value = statistics.fmean(float(r[key + '_mean']) for r in rows)
        if not math.isfinite(value) or abs(value-summary['metrics'][key]['mean']) > 1e-10:
            raise ValueError('quality aggregate mismatch')
        result[key] = value
    for row in rows:
        if (int(row['frames']), int(row['height']), int(row['width'])) != (81, 480, 832):
            raise ValueError('quality shape mismatch')
        for side in ('reference', 'candidate'):
            if sha256(Path(row[side])) != row[side+'_sha256']:
                raise ValueError('quality video hash mismatch')
    return result


def worker(index, config):
    import torch
    from ours4wan21.shared import benchmark
    benchmark()
    from protocol import checkpoint, prepare_resident, source_lock
    from metrics import flops_for_calls
    from reporting import extract_component_latency, extract_component_tflops
    from ours4wan21.inference import physical_gpu_uuid
    sys.path.insert(0, str(PROJECT))
    from inference_timing import _PipelineProfiler
    from wan21_integration import apply_seacache
    job = config['jobs'][index]
    out = ROOT / f'target_{index}'
    if out.exists():
        raise FileExistsError('partial generation must be inspected, not overwritten: '+str(out))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('select exactly one GPU')
    if torch.cuda.get_device_properties(0).total_memory < 44*1024**3:
        raise ValueError('need 48GB class resident GPU')
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    uuid = physical_gpu_uuid(torch)
    if uuid != job['gpu_uuid']:
        raise ValueError('baseline physical GPU mismatch')
    source = WORKSPACE / 'data/source/Wan2.1-65386b2'
    source_lock(source)
    model = checkpoint(WORKSPACE / 'models/Wan2.1-T2V-1.3B')
    sys.path.insert(0, str(source))
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import cache_video
    out.mkdir()
    for name in ('videos', 'timings', 'traces'):
        (out/name).mkdir()
    (out/'README.md').write_text('# SeaCache condition\n\nrun.json freezes configuration; videos/, timings/, traces/ and quality/ contain per-prompt evidence.\n')
    gen = dict(size=(832,480), frame_num=81, shift=5., sample_solver='unipc', sampling_steps=50,
               guide_scale=5., seed=42, offload_model=False)
    started = time.perf_counter()
    pipe = wan.WanT2V(config=WAN_CONFIGS['t2v-1.3B'], checkpoint_dir=str(model), device_id=0,
                     rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False)
    prepare_resident(pipe)
    if pipe.param_dtype != torch.bfloat16 or pipe.t5_cpu:
        raise ValueError('wrong dtype/residency')
    torch.cuda.synchronize()
    init_seconds = time.perf_counter()-started
    with torch.no_grad():
        warmup = pipe.generate(config['prompts'][0]['prompt'], **gen)
    del warmup
    torch.cuda.synchronize()
    apply_seacache(pipe, task='t2v-1.3B', threshold=job['threshold'], trace_path=None, use_ret_steps=False)
    dump(out/'run.json', dict(protocol=PROTOCOL, method='seacache', gpu_uuid=uuid,
         prompts=config['prompts'], job=job, warmup='one unmeasured full native generation',
         config_sha256=sha256(ROOT/'config.json')))
    profile = read(PROFILE)
    rows = []
    for prompt in config['prompts']:
        sid = prompt['sample_id']
        profiler = _PipelineProfiler(pipe, init_wall_seconds=init_seconds,
                    output_path=out/'timings'/f'{sid}.json', implementation='seacache')
        profiler.install()
        with torch.no_grad():
            video = pipe.generate(prompt['prompt'], **gen)
        trace = pipe.model.seacache_controller.summary()
        dump(out/'traces'/f'{sid}.json', trace)
        timing = read(out/'timings'/f'{sid}.json')
        if timing['status'] != 'success' or len(timing['calls']) != 100:
            raise ValueError('incomplete timing')
        rows.append(dict(sample_id=sid, generate_seconds=timing['pipeline_generate_wall_seconds'],
                         dit_tflops=flops_for_calls(timing['calls'],profile,'seacache'),
                         **extract_component_latency(timing), **extract_component_tflops(profile)))
        cache_video(tensor=video[None], save_file=str(out/'videos'/f'{sid}.mp4'),fps=16,nrow=1,normalize=True,value_range=(-1,1))
        del video
        print(json.dumps(dict(target=job['target'], completed=len(rows), sample_id=sid)), flush=True)
    dump(out/'components.json',dict(rows=rows))
    dump(out/'COMPLETE.json',dict(status='generation_complete',videos=5))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--worker',type=int,choices=range(3))
    args = p.parse_args()
    if 'wan2.2' not in Path(sys.prefix).name.lower():
        raise ValueError('use Wan2.2 environment')
    mapping = read(MAPPING)
    source_hashes = {str(path):sha256(path) for path in (Path(__file__), PROJECT/'seacache.py',
                    PROJECT/'wan21_integration.py', PROJECT/'inference_timing.py', PROFILE, MAPPING)}
    jobs = []
    prompts = read(PREVIOUS/'baselines/gpu0/run.json')['prompts']
    if [r['sample_id'] for r in prompts] != [f'vbench200_{s}' for s in ('001','016','056','135','159')]:
        raise ValueError('wrong prompt subset')
    for gpu,target in enumerate(TARGETS):
        base = PREVIOUS/'baselines'/f'gpu{gpu}'
        manifest = read(base/'run.json')
        if manifest['prompts'] != prompts or manifest['protocol'] != PROTOCOL or read(base/'COMPLETE.json')['videos'] != 5:
            raise ValueError('baseline protocol/prompt mismatch')
        if manifest['flops_profile_sha256'] != sha256(PROFILE):
            raise ValueError('baseline profile mismatch')
        for path in (base/'run.json',base/'components.json',*[base/'videos'/f"{r['sample_id']}.mp4" for r in prompts]):
            source_hashes[str(path)] = sha256(path)
        jobs.append(dict(gpu=gpu,target=target,threshold=threshold_for(mapping,target),baseline=str(base),gpu_uuid=manifest['gpu_uuid']))
    config = dict(protocol=PROTOCOL,prompts=prompts,jobs=jobs,source_sha256=source_hashes,
                  vbench_enabled=False,adaptive_rescan=False,comparison='same nominal targets; report measured speedups, not speed matched')
    if args.resume or args.worker is not None:
        if read(ROOT/'config.json') != config:
            raise ValueError('frozen input mismatch')
    else:
        link = PROJECT/'experiment_results'/ROOT.name
        if ROOT.exists() or link.exists() or link.is_symlink():
            raise FileExistsError('use --resume')
        ROOT.mkdir()
        link.symlink_to(ROOT,target_is_directory=True)
        for name in ('logs','analysis'):
            (ROOT/name).mkdir()
            (ROOT/name/'README.md').write_text(f'# {name}\n\nSeaCache five-prompt comparison {name} artifacts.\n')
        (ROOT/'README.md').write_text('# SeaCache five-prompt comparison\n\nconfig.json freezes inputs; target_*/ contains videos, trace, timings, quality; analysis/ contains merged 13-group comparison; logs/ and status.json describe execution.\n')
        dump(ROOT/'config.json',config)
        dump(ROOT/'status.json',dict(status='prepared',candidate_videos=15))
    if args.prepare_only:
        print(json.dumps(config['jobs'],indent=2));return
    if args.worker is not None:
        worker(args.worker,config);return
    with (ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                 TORCH_HOME=str(WORKSPACE/'models/torch-cache'),OPENBLAS_NUM_THREADS='1',
                 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        def run_job(index):
            job=jobs[index];out=ROOT/f'target_{index}'
            child_env=dict(env,CUDA_VISIBLE_DEVICES=str(job['gpu']))
            with (ROOT/'logs'/f'target_{index}.log').open('ab') as log:
                if not (out/'COMPLETE.json').exists():
                    subprocess.run([sys.executable,'-u',str(Path(__file__)),'--worker',str(index)],env=child_env,stdout=log,stderr=subprocess.STDOUT,check=True)
                if not (out/'quality').exists():
                    subprocess.run([sys.executable,str(OFFICIAL/'VideoMetrics/evaluate.py'),'--reference-dir',job['baseline']+'/videos',
                        '--candidate-dir',str(out/'videos'),'--expected-frames','81','--device','cuda:0','--output-dir',str(out/'quality')],
                        env=child_env,stdout=log,stderr=subprocess.STDOUT,check=True)
            metrics=quality_means(out/'quality',[r['sample_id'] for r in prompts])
            br=read(Path(job['baseline'])/'components.json')['rows']; cr=read(out/'components.json')['rows']
            if len(cr)!=5 or {r['sample_id'] for r in cr}!={r['sample_id'] for r in prompts}:
                raise ValueError('incomplete performance')
            r=dict(label='SeaCache',epoch=None,k=None,target=job['target'],threshold=job['threshold'],
                   speedup=sum(x['generate_seconds'] for x in br)/sum(x['generate_seconds'] for x in cr),
                   seconds=statistics.fmean(x['generate_seconds'] for x in cr),**metrics)
            for key in ('dit_tflops','t5_cuda_seconds','dit_cuda_seconds','vae_decode_cuda_seconds','estimated_t5_tflops_per_video','estimated_vae_decode_tflops_per_video'):
                r[key]=statistics.fmean(x[key] for x in cr)
            dump(out/'RESULT.json',r)
            return r
        try:
            if (ROOT/'COMPLETE.json').exists():return
            # Respect any already-running compute workload; no process is killed.
            dump(ROOT/'status.json',dict(status='waiting_for_gpus'))
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
                time.sleep(30)
            dump(ROOT/'status.json',dict(status='running',jobs=jobs))
            with ThreadPoolExecutor(max_workers=3) as pool:
                sea=list(pool.map(run_job,range(3)))
            checked=json.loads(subprocess.check_output([sys.executable,str(OURS/'experiments/vbench5_speed_targets_v1/readout_completed.py')],env=env,text=True))
            result=checked['results']
            for r in result:
                r['target']=dict(zip((23,29,35),TARGETS))[r['k']]
                r['threshold']=None
            result+=sea
            keys=list(dict.fromkeys(key for row in result for key in row))
            with (ROOT/'analysis/comparison.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(result)
            text=['# 五提示词 SeaCache / Ours 对照','',
                  '相同名义目标，不是严格等速。5条固定prompt、单seed42，未计算VBench。',
                  'SeaCache为仓库corrected filtered-boundary实现，独立CFG分支；Ours使用Exact-K。',
                  '加速比为同卡baseline/候选完整generate总时间比，不含加载、warmup、保存和评测。','',
                  '|方法|名义目标|实测加速|秒/视频|PSNR dB|SSIM|LPIPS|','|---|---:|---:|---:|---:|---:|---:|']
            for r in sorted(result,key=lambda r:(r['target'],r['label'])):
                text.append(f"|{r['label']}|{r['target']:.1f}|{r['speedup']:.3f}|{r['seconds']:.3f}|{r['psnr_rgb_db']:.3f}|{r['ssim_rgb']:.4f}|{r['lpips_alex_v0_1_spatial']:.4f}|")
            (ROOT/'analysis/COMPARISON.md').write_text('\n'.join(text)+'\n')
            dump(ROOT/'analysis/VALIDATION.json',dict(status='pass',conditions=39,sea_video_pairs=15,sea_frame_pairs=1215,
                source_sha256=config['source_sha256'],quality_sha256={str(ROOT/f'target_{i}/quality/summary.json'):sha256(ROOT/f'target_{i}/quality/summary.json') for i in range(3)}))
            dump(ROOT/'COMPLETE.json',dict(status='complete',comparison_sha256=sha256(ROOT/'analysis/comparison.csv')))
            dump(ROOT/'status.json',dict(status='complete',sea_results=sea))
        except BaseException as error:
            dump(ROOT/'FAILED.json',dict(error=repr(error)))
            dump(ROOT/'status.json',dict(status='failed',error=repr(error)))
            raise


if __name__=='__main__':
    main()
