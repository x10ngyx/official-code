"""Remote-ready collection/replay/training/VBench20 orchestration."""
import argparse
import importlib.util
import shutil
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .contracts import (EXP_ROOT, MODEL_ROOT, OFFICIAL, PROJECT, PROTOCOL,
                        create_result, dump, sha256, state_contract, under)
from .online_common import (OnlineConfig, THREAD_KEYS, atomic_torch_save, checkpoint_identity,
    freeze_json, gpu_slot, inventory, isolate_prompts, make_plan, prepare_directory, prompts, read,
    require_environment, seal, verified)
from .online_generation import job_identity


def source_hashes():
    paths=list((PROJECT/'ours4wan21').glob('*.py'))+[PROJECT/'online.py']
    for name in ('SeaCache4Wan21','VideoMetrics','VbenchEvaluation','Wan21Benchmark','ComponentMetrics'):
        paths.extend(p for p in (OFFICIAL/name).rglob('*.py') if 'tests' not in p.parts)
    return {str(p.relative_to(OFFICIAL)):sha256(p) for p in sorted(set(paths))}


def environment(gpu):
    env=dict(os.environ)
    env.update({k:'1' for k in THREAD_KEYS})
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHON_BIN=sys.executable,
        TORCH_HOME=str(MODEL_ROOT/'torch-cache'),VBENCH_CACHE_DIR=str(MODEL_ROOT/'VBench'),
        HF_HOME=str(MODEL_ROOT/'VBench/huggingface'),XDG_CACHE_HOME=str(MODEL_ROOT/'VBench/xdg'))
    return env


def invoke(command, log, gpu):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        stream.write(json.dumps([str(x) for x in command])+'\n');stream.flush()
        subprocess.run([str(x) for x in command],env=environment(gpu),stdout=stream,
                       stderr=subprocess.STDOUT,check=True)


def dataset_file(path):
    path=Path(path).resolve()
    return path/'transitions.pt' if path.is_dir() else path


def prepare(args):
    require_environment()
    import torch
    from .policy import Policy,resolve_budget
    from .train import validate_bundle
    from .online_training import fixed_support
    from .shared import benchmark
    benchmark()
    from protocol import source_lock,checkpoint
    config=OnlineConfig()
    missing=[name for name in ('lpips','vbench','imageio','cv2') if importlib.util.find_spec(name) is None]
    missing += [name for name in ('ffmpeg','ffprobe') if shutil.which(name) is None]
    if missing:
        raise ValueError('missing generation/quality dependencies: '+', '.join(missing))
    data=dataset_file(args.dataset)
    parent_path=under(args.start_checkpoint,MODEL_ROOT)
    Policy(parent_path,device='cpu')  # shared architecture/normalizer/protocol checks
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    mode=parent['state']['mode']
    if 'online' in parent:
        raise ValueError('prepare must start from an offline checkpoint; use run to resume online')
    bundle=torch.load(data,map_location='cpu',weights_only=False)
    validate_bundle(bundle,mode)
    if parent['dataset_manifest']!=bundle['manifest']:
        raise ValueError('offline dataset differs from the checkpoint training manifest')
    if bundle['manifest'].get('trajectories')!=3000:
        raise ValueError('production offline source must be the frozen random3000 bundle')
    pool,evaluation,registry=[prompts(p) for p in (args.train_prompts,args.eval_prompts,args.prompt_registry)]
    if len(pool)!=config.prompt_pool_size:
        raise ValueError('freeze exactly 3000 eligible online pool prompts')
    isolate_prompts(pool,evaluation,registry,bundle['manifest']['prompt_splits'])
    # Verify registry text against the archived collection manifest when available.
    # A separate registry remains required for portable bundles whose old paths are unavailable.
    calibration=Path(args.calibration).resolve(); profile=Path(args.flops_profile).resolve()
    entries=read(calibration)['entries']
    if not entries or any(not math.isfinite(float(e['calibrated_speedup'])) for e in entries):
        raise ValueError('empty or nonfinite calibration')
    speeds=[float(e['calibrated_speedup']) for e in entries]
    if min(speeds)>config.target_min or max(speeds)<config.target_max:
        raise ValueError('calibration must cover the complete [1.3,3.5] target range')
    for target in [config.target_min,config.target_max,*config.evaluation_targets]:
        resolve_budget(target_speedup=target,calibration=calibration)
    if len({e['skip_budget'] for e in entries})!=len(entries):
        raise ValueError('duplicate K values in calibration')
    f=read(profile)
    if f['input']['video_shape_fhw']!=[81,480,832] or f['input']['transformer_blocks']!=30:
        raise ValueError('wrong Wan21 FLOPs profile')
    from reporting import extract_component_tflops
    extract_component_tflops(f)
    root=Path(args.wan21_root).resolve(); lock=source_lock(root)
    wan_checkpoint=checkpoint(Path(args.checkpoint_dir).resolve())
    gpus=args.gpus.split(',')
    if not gpus or len(gpus)!=len(set(gpus)) or any(not g.strip() for g in gpus):
        raise ValueError('provide distinct GPUs/UUIDs')
    out=under(args.output_dir,EXP_ROOT); weights=under(args.weights_dir,MODEL_ROOT)
    inputs={name:dict(path=str(Path(path).resolve()),sha256=sha256(path)) for name,path in dict(
        dataset=data,start_checkpoint=parent_path,train_prompts=args.train_prompts,
        eval_prompts=args.eval_prompts,prompt_registry=args.prompt_registry,
        calibration=calibration,flops_profile=profile).items()}
    manifest=dict(schema='ours4wan21_online_pipeline_v1',config=config.payload(),mode=mode,
        inputs=inputs,source_hashes=source_hashes(),protocol=PROTOCOL,gpus=gpus,
        paths=dict(dataset=str(data),start_checkpoint=str(parent_path),wan21_root=str(root),
                   wan_checkpoint=str(wan_checkpoint),flops_profile=str(profile),
                   calibration=str(calibration),weights=str(weights)),
        source_lock=lock,model_inventory=inventory(wan_checkpoint),pool=pool,evaluation=evaluation,
        evaluation_score='vbench_custom_input_raw_mean_v1: mean of ten custom-input dimensions',
        optimizer_policy='R1 reset all three optimizers; R2+ inherit selected checkpoint and RNG')
    if out.exists():
        if read(out/'manifest.json')!=manifest or not verified(out/'inputs'):
            raise ValueError('existing run differs or has incomplete preparation; use a fresh result name')
        print(out);return
    if weights.exists():
        raise FileExistsError('use a fresh models directory')
    create_result(out,'# Ours4Wan21 online fine-tuning\n\nmanifest.json freezes inputs. rounds/ holds collection/replay/training/evaluation. artifacts/ holds reusable per-video results. commands/ and logs/ support restart. Model checkpoints are linked from models/.')
    weights.mkdir(parents=True);(weights/'README.md').write_text('Online checkpoints by round; each round selected.pt carries optimizer/RNG lineage.\n')
    (out/'model_weights').symlink_to(weights,target_is_directory=True)
    dump(out/'manifest.json',manifest)
    inp=out/'inputs';inp.mkdir()
    (inp/'README.md').write_text('Frozen prompt lists and train-only actor state support. No model weights.\n')
    atomic_torch_save(fixed_support(bundle,config),inp/'fixed_states.pt')
    dump(inp/'pool.json',pool);dump(inp/'vbench20.json',evaluation)
    seal(inp,['fixed_states.pt','pool.json','vbench20.json'],identity=dict(inputs=inputs,mode=mode))
    dump(out/'STATUS.json',dict(phase='prepared',round=0))
    print(out)


def check_run(run):
    m=read(run/'manifest.json')
    if m['config']!=OnlineConfig().payload() or m['source_hashes']!=source_hashes():
        raise ValueError('frozen online configuration or source code changed')
    if inventory(m['paths']['wan_checkpoint'])!=m['model_inventory']:
        raise ValueError('Wan model inventory changed')
    if m['protocol']!=PROTOCOL or not verified(run/'inputs'):
        raise ValueError('invalid frozen run inputs')
    for item in m['inputs'].values():
        if sha256(item['path'])!=item['sha256']:
            raise ValueError('input changed: '+item['path'])
    return m


def base_job(run,m,row):
    return dict(kind='baseline',sample_id=row['sample_id'],prompt=row['prompt'],
        slot=gpu_slot(row['sample_id'],len(m['gpus'])),
        output=str(run/'artifacts/baselines'/row['sample_id']))


def candidate_job(run,m,row,parent,output,*,sample=False):
    job=dict(kind='collection' if sample else 'evaluation',sample_id=row['sample_id'],
        prompt=row['prompt'],slot=gpu_slot(row['sample_id'],len(m['gpus'])),
        skip_budget=row['skip_budget'],target_speedup=row['target_speedup'],
        checkpoint=checkpoint_identity(parent),output=str(output))
    if sample:
        job.update(trajectory_id=row['trajectory_id'],sampling_seed=row['sampling_seed'])
    return job


def dispatch(run,m,jobs,label):
    # Preserve stable physical GPU assignment for every baseline/candidate pair.
    unique={}
    for job in jobs:
        if job['output'] in unique and unique[job['output']]!=job:
            raise ValueError('conflicting jobs share an output directory')
        unique[job['output']]=job
    pending=[j for j in unique.values() if not verified(j['output'],job_identity(j,m))]
    if not pending:
        return
    folder=run/'commands'/label;folder.mkdir(parents=True,exist_ok=True)
    active=[];streams=[]
    try:
        for slot,gpu in enumerate(m['gpus']):
            shard=[j for j in pending if j['slot']==slot]
            if not shard:
                continue
            path=folder/f'gpu_{slot}.json';dump(path,shard)
            log=run/'logs'/f'{label}_gpu{slot}.log';log.parent.mkdir(exist_ok=True)
            stream=log.open('a');streams.append(stream)
            command=[sys.executable,str(PROJECT/'online.py'),'worker','--run-dir',str(run),'--jobs',str(path)]
            active.append(subprocess.Popen(command,env=environment(gpu),stdout=stream,stderr=subprocess.STDOUT))
        while active:
            for proc in list(active):
                result=proc.poll()
                if result is not None:
                    active.remove(proc)
                    if result:
                        raise RuntimeError(f'generation worker failed ({result}); inspect {run / "logs"}')
            if active:
                time.sleep(1)
    finally:
        for proc in active:
            proc.terminate()
        for proc in active:
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        for stream in streams:stream.close()
    for job in jobs:
        if not verified(job['output'],job_identity(job,m)):
            raise ValueError('unsealed generation job')


def stage_link(path,target):
    path=Path(path);target=Path(target).resolve(strict=True)
    if path.is_symlink():
        if path.resolve()!=target:raise ValueError('staging symlink changed')
    elif path.exists():
        raise FileExistsError(path)
    else:path.symlink_to(target)


def paired_quality(run,m,pairs,output):
    identity=dict(pairs=[dict(id=sid,baseline=sha256(Path(base)/'video.mp4'),
                    candidate=sha256(Path(cand)/'video.mp4')) for sid,base,cand in pairs],protocol='rgb_full_reference_v1')
    for _,base,cand in pairs:
        if not verified(base) or not verified(cand):
            raise ValueError('quality requires sealed video artifacts')
        b,c=read(Path(base)/'generation.json'),read(Path(cand)/'generation.json')
        if b['gpu_uuid']!=c['gpu_uuid'] or b['protocol']!=c['protocol']:
            raise ValueError('baseline/candidate physical GPU or protocol mismatch')
        bj,cj=b['identity']['job'],c['identity']['job']
        if bj['prompt']!=cj['prompt'] or bj['sample_id']!=cj['sample_id']:
            raise ValueError('baseline/candidate prompt mismatch')
    output=Path(output)
    if prepare_directory(output,identity):
        for folder in ('reference','candidate'):(output/folder).mkdir()
        for sid,base,cand in pairs:
            stage_link(output/'reference'/f'{sid}.mp4',Path(base)/'video.mp4')
            stage_link(output/'candidate'/f'{sid}.mp4',Path(cand)/'video.mp4')
        invoke([sys.executable,OFFICIAL/'VideoMetrics/evaluate.py',
            '--reference-dir',output/'reference','--candidate-dir',output/'candidate',
            '--expected-frames','81','--device','cuda:0','--model-cache',MODEL_ROOT/'torch-cache',
            '--output-dir',output/'metrics'],output/'command.log',m['gpus'][0])
        with (output/'metrics/per_video.csv').open() as f:
            rows=list(csv.DictReader(f))
        if (len(rows)!=len(pairs) or {r['video_id'] for r in rows}!={p[0] for p in pairs} or
                any((int(r['frames']),int(r['height']),int(r['width']))!=(81,480,832) for r in rows)):
            raise ValueError('invalid paired video coverage or geometry')
        seal(output,['identity.json','metrics/per_frame.csv','metrics/per_video.csv','metrics/summary.json'],identity=identity)
    with (output/'metrics/per_video.csv').open() as f:
        return {r['video_id']:r for r in csv.DictReader(f)}


def vbench(run,m,rows,output):
    identity=dict(videos=[dict(id=r['sample_id'],prompt=r['prompt'],sha256=sha256(Path(r['output'])/'video.mp4')) for r in rows],
                  protocol='vbench_custom_input_raw_mean_v1')
    output=Path(output)
    if prepare_directory(output,identity):
        staging=output/'videos';staging.mkdir()
        for row in rows:stage_link(staging/f'{row["sample_id"]}.mp4',Path(row['output'])/'video.mp4')
        dump(output/'prompt_map.json',{r['sample_id']+'.mp4':r['prompt'] for r in rows})
        invoke(['bash',OFFICIAL/'VbenchEvaluation/run_custom_vbench.sh',staging,
                output/'scores',output/'prompt_map.json'],output/'command.log',m['gpus'][0])
        score=read(output/'scores/vbench_custom_aggregate_scores.json')
        if len(score['raw_dimension_scores'])!=10 or not math.isfinite(score['vbench_score']):
            raise ValueError('incomplete VBench custom score')
        files=['identity.json','prompt_map.json',*map(lambda p:str(p.relative_to(output)),(output/'scores').rglob('*.json'))]
        seal(output,files,identity=identity)
    return read(output/'scores/vbench_custom_aggregate_scores.json')


def collect_round(run,m,parent,r,config):
    import torch
    from .policy import resolve_budget
    from .online_training import trace_transitions
    root=run/'rounds'/f'round_{r:03d}';root.mkdir(parents=True,exist_ok=True)
    (root/'README.md').write_text('plan.json fixes sampled prompts, targets and RNG; collection/, replay.pt, training/ and optional evaluation/ follow.\n')
    plan=make_plan(m['pool'],r,len(m['gpus']),
        lambda t:resolve_budget(target_speedup=t,calibration=m['paths']['calibration']),config)
    freeze_json(root/'plan.json',dict(round=r,parent=checkpoint_identity(parent),rows=plan))
    jobs=[];pairs=[]
    for row in plan:
        base=base_job(run,m,row);jobs.append(base)
        cand=candidate_job(run,m,row,parent,root/'collection'/row['trajectory_id'],sample=True)
        jobs.append(cand);pairs.append((row['trajectory_id'],base['output'],cand['output']))
    dispatch(run,m,jobs,f'collect_r{r:03d}')
    quality=paired_quality(run,m,pairs,root/'quality')
    identity=dict(plan=sha256(root/'plan.json'),quality=sha256(root/'quality/COMPLETE.json'),
        traces=[sha256(Path(c)/'trace.json') for _,_,c in pairs],state=state_contract(m['mode']))
    replay_dir=root/'replay'
    if prepare_directory(replay_dir,identity):
        episodes=[]
        for sid,base,cand in pairs:
            psnr=float(quality[sid]['psnr_rgb_db_mean'])
            episodes.append(trace_transitions(read(Path(cand)/'trace.json'),psnr,m['mode']))
        data=dict(round=r,state=state_contract(m['mode']),plan=plan,
            parent=checkpoint_identity(parent),tensors={k:torch.cat([e[k] for e in episodes]) for k in episodes[0]})
        atomic_torch_save(data,replay_dir/'transitions.pt')
        seal(replay_dir,['identity.json','transitions.pt'],identity=identity)
    return replay_dir/'transitions.pt'


def aggregate_target(pairs,quality,score):
    rows=[]
    for sid,base,cand in pairs:
        b,c=read(Path(base)/'measurement.json'),read(Path(cand)/'measurement.json')
        rows.append(dict(sample_id=sid,baseline=b,candidate=c,quality=quality[sid],
            speedup=b['generate_seconds']/c['generate_seconds']))
    fields=('psnr_rgb_db_mean','ssim_rgb_mean','lpips_alex_v0_1_spatial_mean')
    return dict(rows=rows,mean_quality={k:sum(float(q[k]) for q in quality.values())/len(quality) for k in fields},
        generate_speedup=sum(x['baseline']['generate_seconds'] for x in rows)/sum(x['candidate']['generate_seconds'] for x in rows),
        mean_generate_seconds=sum(x['candidate']['generate_seconds'] for x in rows)/len(rows),
        mean_dit_tflops=sum(x['candidate']['dit_tflops'] for x in rows)/len(rows),
        vbench_score=score['vbench_score'],vbench_dimensions=score['raw_dimension_scores'])


def evaluate_round(run,m,parent,r,config):
    from .policy import resolve_budget
    root=run/'rounds'/f'round_{r:03d}'/'evaluation';root.mkdir(parents=True,exist_ok=True)
    (root/'README.md').write_text('VBench20: 20 remote-selected prompts × five targets; ten custom-input dimensions and paired PSNR/SSIM/LPIPS.\n')
    frozen=Path(m['paths']['start_checkpoint'])
    identity=dict(checkpoint=checkpoint_identity(parent),reference=checkpoint_identity(frozen),
                  prompts=m['evaluation'],targets=list(config.evaluation_targets),round=r)
    freeze_json(root/'identity.json',identity)
    results=[]
    baselines=[base_job(run,m,row) for row in m['evaluation']]
    dispatch(run,m,baselines,f'eval_baseline_r{r:03d}')
    baseline_score=vbench(run,m,baselines,run/'evaluation_reference/baseline_vbench20')
    for target in config.evaluation_targets:
        tag=f'target_{target:g}x'
        budget=resolve_budget(target_speedup=target,calibration=m['paths']['calibration'])
        comparisons={}
        for label,checkpoint,out in (
            ('offline',frozen,run/'evaluation_reference'/tag),('online',parent,root/tag)):
            jobs=[];pairs=[]
            for prompt,base in zip(m['evaluation'],baselines):
                row=dict(**prompt,target_speedup=target,skip_budget=budget)
                job=candidate_job(run,m,row,checkpoint,out/'candidates'/row['sample_id'])
                jobs.append(job);pairs.append((row['sample_id'],base['output'],job['output']))
            dispatch(run,m,jobs,f'eval_{label}_{tag}_r{r:03d}')
            quality=paired_quality(run,m,pairs,out/'quality')
            score=vbench(run,m,jobs,out/'vbench20')
            comparisons[label]=aggregate_target(pairs,quality,score)
        results.append(dict(target_speedup=target,skip_budget=budget,**comparisons,
            delta_psnr=comparisons['online']['mean_quality']['psnr_rgb_db_mean']-comparisons['offline']['mean_quality']['psnr_rgb_db_mean'],
            delta_vbench_score=comparisons['online']['vbench_score']-comparisons['offline']['vbench_score']))
    dump(root/'metrics.json',dict(round=r,prompts=20,candidate_cells=100,reference_cells=100,
        protocol=PROTOCOL,score_protocol='vbench_custom_input_raw_mean_v1',
        baseline_vbench_score=baseline_score['vbench_score'],per_target=results))
    lines=['# VBench20 online comparison', '', 'Ten custom-input VBench dimensions; scores below are raw means.', '',
        '| Target | Online latency (s) | Online speedup | Online PSNR | Offline PSNR | SSIM | LPIPS | DiT TFLOPs | VBench score |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in results:
        a,b=item['online'],item['offline'];q=a['mean_quality']
        lines.append(f"| {item['target_speedup']:g}x | {a['mean_generate_seconds']:.3f} | {a['generate_speedup']:.4f} | {q['psnr_rgb_db_mean']:.4f} | {b['mean_quality']['psnr_rgb_db_mean']:.4f} | {q['ssim_rgb_mean']:.6f} | {q['lpips_alex_v0_1_spatial_mean']:.6f} | {a['mean_dit_tflops']:.3f} | {a['vbench_score']:.6f} |")
    (root/'READOUT.md').write_text('\n'.join(lines)+'\n')
    seal(root,['identity.json','metrics.json','READOUT.md'],identity=identity)


def run_pipeline(args):
    require_environment();run=under(args.run_dir,EXP_ROOT)
    with (run/'pipeline.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('another orchestrator already owns this run')
        m=check_run(run);c=OnlineConfig();parent=Path(m['paths']['start_checkpoint']);online=[]
        try:
            for r in range(1,c.rounds+1):
                dump(run/'STATUS.json',dict(round=r,phase='collection'))
                online.append(collect_round(run,m,parent,r,c))
                dump(run/'STATUS.json',dict(round=r,phase='training'))
                invoke([sys.executable,PROJECT/'online.py','train-round','--run-dir',run,'--round',str(r)],
                       run/'logs'/f'train_r{r:03d}.log',m['gpus'][0])
                selected=read(run/'rounds'/f'round_{r:03d}'/'training/selected.json')
                if sha256(selected['path'])!=selected['sha256']:raise ValueError('checkpoint corrupted')
                parent=Path(selected['path'])
                if r%c.evaluation_every==0:
                    dump(run/'STATUS.json',dict(round=r,phase='vbench20'))
                    evaluate_round(run,m,parent,r,c)
                dump(run/'STATUS.json',dict(round=r,phase='round_complete',checkpoint=selected))
            dump(run/'RESULT.json',dict(status='complete',rounds=c.rounds,
                final_checkpoint=checkpoint_identity(parent),evaluated_rounds=list(range(c.evaluation_every,c.rounds+1,c.evaluation_every))))
            dump(run/'STATUS.json',dict(round=c.rounds,phase='complete'))
        except BaseException as exc:
            dump(run/'LAST_ERROR.json',dict(error=repr(exc),status=read(run/'STATUS.json')))
            raise


def train_worker(args):
    require_environment()
    import torch
    from .online_training import train_round
    run=under(args.run_dir,EXP_ROOT);m=check_run(run);r=args.round;c=OnlineConfig()
    if not 1<=r<=c.rounds:raise ValueError('invalid round')
    torch.set_num_threads(1)
    parent=Path(m['paths']['start_checkpoint']) if r==1 else Path(read(run/'rounds'/f'round_{r-1:03d}'/'training/selected.json')['path'])
    online=[run/'rounds'/f'round_{i:03d}'/'replay/transitions.pt' for i in range(1,r+1)]
    train_round(parent,m['paths']['dataset'],online,run/'inputs/fixed_states.pt',
        run/'rounds'/f'round_{r:03d}'/'training',Path(m['paths']['weights'])/f'round_{r:03d}',r,c,'cuda:0')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare',help='CPU preflight and freeze remote inputs; does not generate video')
    for name in ('dataset','start-checkpoint','train-prompts','eval-prompts','prompt-registry',
                 'calibration','flops-profile','wan21-root','checkpoint-dir','output-dir','weights-dir'):
        q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--gpus',default='0,1,2,3',help='stable physical GPU IDs or UUIDs')
    q=sub.add_parser('run',help='start or resume the frozen pipeline');q.add_argument('--run-dir',type=Path,required=True)
    q=sub.add_parser('worker');q.add_argument('--run-dir',type=Path,required=True);q.add_argument('--jobs',type=Path,required=True)
    q=sub.add_parser('train-round');q.add_argument('--run-dir',type=Path,required=True);q.add_argument('--round',type=int,required=True)
    q=sub.add_parser('build-pool',help='exclude offline held-out and VBench20 prompts from a full registry')
    for name in ('dataset','prompt-registry','eval-prompts','output-dir'):
        q.add_argument('--'+name,type=Path,required=True)
    q=sub.add_parser('build-calibration',help='export measured fixed-K runs from generate.py')
    q.add_argument('--baseline-dir',type=Path,required=True)
    q.add_argument('--candidate-dirs',type=Path,nargs='+',required=True)
    q.add_argument('--output-dir',type=Path,required=True)
    sub.add_parser('show-config')
    args=p.parse_args()
    if args.action=='show-config':print(json.dumps(OnlineConfig().payload(),indent=2))
    elif args.action in ('build-pool','build-calibration'):
        from .online_setup import build_pool,build_calibration
        (build_pool if args.action=='build-pool' else build_calibration)(args)
    elif args.action=='prepare':prepare(args)
    elif args.action=='run':run_pipeline(args)
    elif args.action=='train-round':train_worker(args)
    else:
        from .online_generation import generate_jobs
        generate_jobs(args.run_dir,args.jobs)
