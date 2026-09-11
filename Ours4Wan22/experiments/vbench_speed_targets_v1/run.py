"""Single-GPU VBench target suite using the formal Ours4Wan22 CLI."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
from ours4wan22.shared import OFFICIAL, MODELS, read, write, sha256, under, result_dir, implementation_hashes, verify_sources
from ours4wan22.contracts import PROTOCOL
from ours4wan22.policy import DEFAULT_CALIBRATION, resolve_budget
from ours4wan22.generation import validate_jobs
from ours4wan22.evaluation import paired, validate_vbench_jobs

DEFAULT_IDS = ['vbench200_001','vbench200_016','vbench200_056','vbench200_135','vbench200_159']


def quality_values(quality, prompt_count):
    if quality['video_count']!=prompt_count or quality['frame_count_total']!=45*prompt_count:
        raise ValueError('quality sample/frame coverage mismatch')
    values = {name:quality['metrics'][key]['mean'] for name,key in
              [('psnr_rgb_db','psnr_rgb_db'),('ssim','ssim_rgb'),('lpips','lpips_alex_v0_1_spatial')]}
    if not all(math.isfinite(v) for v in values.values()):raise ValueError('nonfinite quality score')
    return values


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--wan22-root',type=Path,required=True)
    p.add_argument('--checkpoint-dir',type=Path,default=MODELS/'Wan2.2-T2V-A14B')
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--metric-models',type=Path,default=MODELS/'evaluation')
    p.add_argument('--calibration',type=Path,default=DEFAULT_CALIBRATION)
    p.add_argument('--targets',type=float,nargs='+',default=[1.8,2.4,3.0])
    group = p.add_mutually_exclusive_group()
    group.add_argument('--prompt-ids',nargs='+')
    group.add_argument('--all-prompts',action='store_true')
    p.add_argument('--baseline',type=Path,help='reuse a complete native Ours4Wan22 CLI baseline with the same jobs/protocol/GPU')
    p.add_argument('--gpu',required=True,help='one physical GPU index or UUID')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true',help='print validated plan and commands; no GPU work or output directory')
    p.add_argument('--resume',action='store_true',help='reuse sealed complete stages; never overwrite incomplete stages')
    return p


def jobs_for(ids=None, all_prompts=False):
    rows = [json.loads(line) for line in (OFFICIAL/'Vbench200/prompts.jsonl').read_text().splitlines()]
    index = {r['sample_id']:r['prompt_en'] for r in rows}
    ids = list(index) if all_prompts else (DEFAULT_IDS if ids is None else ids)
    if len(set(ids)) != len(ids) or any(i not in index for i in ids):
        raise ValueError('prompt IDs must be unique members of VBench200')
    return validate_jobs([dict(sample_id=i,prompt=index[i]) for i in ids])


def build_plan(a):
    if not a.gpu.strip() or ',' in a.gpu:
        raise ValueError('select exactly one physical GPU')
    jobs = jobs_for(a.prompt_ids,a.all_prompts)
    mode = 'vbench200' if a.all_prompts else 'custom'
    validate_vbench_jobs(jobs,mode)
    if len(set(a.targets)) != len(a.targets):
        raise ValueError('duplicate targets')
    targets = [dict(target=t,K=resolve_budget(target_speedup=t,calibration=a.calibration)) for t in a.targets]
    output = under(a.output,Path('/all/yiran07-disk3/huteng_data/exp'))
    policy = under(a.policy,MODELS)
    checkpoint = under(a.checkpoint_dir,MODELS)
    metric_models = under(a.metric_models,MODELS)
    source = a.wan22_root.resolve()
    inputs = [policy,a.profile.resolve(),a.calibration.resolve(),source/'.seacache4wan22_prepared.json',PROJECT/'source_lock.json',OFFICIAL/'Vbench200/prompts.jsonl']
    base = a.baseline.resolve() if a.baseline else output/(output.name+'_baseline')
    if a.baseline:
        inputs.append(base/'manifest.json')
        m = read(base/'manifest.json')
        if m.get('mode') != 'baseline' or m.get('status') != 'complete' or m.get('protocol') != PROTOCOL or m.get('jobs') != jobs:
            raise ValueError('reused baseline must be a complete matching native Ours4Wan22 baseline')
    common = [sys.executable,str(PROJECT/'main.py'),'generate','--wan22-root',str(source),
              '--checkpoint-dir',str(a.checkpoint_dir.absolute()),'--jobs',str(output/'jobs.json'),'--profile',str(a.profile.resolve())]
    stages = []
    if not a.baseline:
        stages.append(dict(name='baseline',kind='generate',output=str(base),command=common+['--output',str(base)]))
    for i,row in enumerate(targets):
        label=f'target_{i:02d}'
        candidate = output/(output.name+'_'+label)
        evaluation = output/(output.name+'_'+label+'_metrics')
        row.update(candidate=str(candidate),evaluation=str(evaluation))
        stages.append(dict(name=label,kind='generate',output=str(candidate),target=row['target'],K=row['K'],
            command=common+['--policy',str(a.policy.absolute()),'--target-speedup',str(row['target']),
                           '--calibration',str(a.calibration.resolve()),'--output',str(candidate)]))
        stages.append(dict(name=label+'_metrics',kind='evaluate',output=str(evaluation),candidate=str(candidate),
            command=[sys.executable,str(PROJECT/'main.py'),'evaluate','--baseline',str(base),
                     '--candidate',str(candidate),'--output',str(evaluation),'--metric-models',str(a.metric_models.absolute()),
                     '--device','cuda:0','--vbench-mode',mode]))
    return dict(schema='ours22_vbench_speed_targets_v1',protocol=PROTOCOL,jobs=jobs,targets=targets,
                gpu=a.gpu,output=str(output),baseline=str(base),baseline_reused=bool(a.baseline),
                policy=str(policy),profile=str(a.profile.resolve()),checkpoint_dir=str(checkpoint),
                calibration=str(a.calibration.resolve()),vbench_mode=mode,stages=stages,
                source=str(source),input_sha256={str(p):sha256(p) for p in inputs},
                implementation_sha256=implementation_hashes(),warmup_videos_per_generation_process=1)


def validate_generation(path, plan, gpu_uuid, target=None):
    m = read(path/'manifest.json')
    expected = dict(status='complete',protocol=PROTOCOL,jobs=plan['jobs'],gpu_uuid=gpu_uuid,
                    checkpoint_dir=plan['checkpoint_dir'],profile_sha256=plan['input_sha256'][plan['profile']],
                    mode='baseline' if target is None else 'ours')
    if target is not None:
        expected.update(target_speedup=target['target'],skip_budget=target['K'],
                        policy_sha256=plan['input_sha256'][plan['policy']],
                        calibration_sha256=plan['input_sha256'][plan['calibration']])
    for key,value in expected.items():
        if m.get(key) != value:raise ValueError('generation identity mismatch: '+key)
    if m.get('source_manifest') != read(Path(plan['source'])/'.seacache4wan22_prepared.json'):
        raise ValueError('generation prepared source mismatch')
    if [r['sample_id'] for r in m['rows']] != [j['sample_id'] for j in plan['jobs']]:
        raise ValueError('generation sample coverage mismatch')
    for r in m['rows']:
        for folder,suffix,key in [('videos','.mp4','video'),('timings','.json','timing')]+(
                [('traces','.json','trace'),('features','.pt','feature')] if target else []):
            if sha256(path/folder/(r['sample_id']+suffix)) != r[key+'_sha256']:
                raise ValueError('generation artifact changed')
    return m


def execute_stage(stage, plan, env, gpu_uuid):
    for path,expected in plan.get('input_sha256',{}).items():
        if sha256(path)!=expected:raise ValueError('frozen input changed: '+path)
    out=Path(stage['output']);seal=Path(plan['output'])/'stages'/(stage['name']+'.json')
    if out.exists():
        if not seal.exists():
            raise ValueError(f'incomplete/unsealed stage retained at {out}; use a new suite output to rerun')
        sealed=read(seal)
        for relative,h in sealed['files'].items():
            if sha256(out/relative)!=h:raise ValueError('sealed stage artifact changed: '+relative)
    else:
        with (Path(plan['output'])/'logs'/(stage['name']+'.log')).open('x') as log:
            subprocess.run(stage['command'],env=env,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
    if stage['kind']=='generate':
        validate_generation(out,plan,gpu_uuid,stage if 'target' in stage else None)
        files=['manifest.json','run.json','performance.json']
    else:
        paired(Path(plan['baseline']),Path(stage['candidate']))
        s=read(out/'summary.json')
        if s.get('status')!='complete' or s.get('vbench_mode')!=plan['vbench_mode']:
            raise ValueError('evaluation incomplete or mode mismatch')
        files=[str(f.relative_to(out)) for f in out.rglob('*.json')]
        files += [str(f.relative_to(out)) for f in out.rglob('*.csv')]
    if not seal.exists():write(seal,dict(files={f:sha256(out/f) for f in files}))


def run(a):
    if Path(sys.prefix).name!='wan2.2':raise ValueError('use conda wan2.2')
    plan=build_plan(a)
    if a.dry_run:
        print(json.dumps(plan,indent=2,ensure_ascii=False));return
    verify_sources()
    from ours4wan22.policy import Policy
    Policy(a.policy,device='cpu')
    subprocess.run([sys.executable,str(OFFICIAL/'SeaCache4Wan22/scripts/validate_prepared_tree.py'),
                    '--source',plan['source'],'--mode','prepared'],check=True)
    gpu_uuid=subprocess.check_output(['nvidia-smi','--id='+a.gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if not gpu_uuid.startswith('GPU-') or '\n' in gpu_uuid:raise ValueError('cannot uniquely resolve physical GPU')
    output=Path(plan['output'])
    if a.resume:
        if read(output/'plan.json')!=plan:raise ValueError('resume plan/input/code changed')
        if read(output/'gpu.json')['uuid']!=gpu_uuid:raise ValueError('resume physical GPU changed')
    else:
        result_dir(output,'# Ours4Wan22 VBench target suite\n\nplan.json freezes inputs/commands; jobs.json fixes prompts; logs/ contains stage logs; stages/ seals completed work; target folders contain videos, component metrics and quality; summary.json/results.csv/REPORT.md summarize all targets.')
        write(output/'plan.json',plan);write(output/'jobs.json',plan['jobs']);write(output/'gpu.json',dict(uuid=gpu_uuid))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_uuid)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    with (output/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (output/'logs').mkdir(exist_ok=True)
        if plan['baseline_reused']:validate_generation(Path(plan['baseline']),plan,gpu_uuid)
        try:
            for stage in plan['stages']:
                write(output/'STATUS.json',dict(state='running',stage=stage['name']))
                execute_stage(stage,plan,env,gpu_uuid)
            summaries=[];rows=[]
            for t in plan['targets']:
                s=read(Path(t['evaluation'])/'summary.json');summaries.append(s)
                rows.append(dict(target_speedup=t['target'],K=t['K'],
                    actual_speedup=s['latency_speedup_ratio_of_sums'],
                    **quality_values(s['quality'],len(plan['jobs'])),
                    dit_tflops_speedup=s['dit_tflops_speedup_ratio_of_sums'],
                    vbench_mode=s['vbench_mode'],baseline_vbench=s['vbench']['baseline']['vbench_score'],
                    candidate_vbench=s['vbench']['candidate']['vbench_score'],evaluation=t['evaluation']))
            write(output/'summary.json',dict(status='complete',targets=rows,evaluations=summaries,plan_sha256=sha256(output/'plan.json')))
            with (output/'results.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            report=['# Ours4Wan22 VBench speed targets','',f"{len(plan['jobs'])} prompts; mode={plan['vbench_mode']}; same-GPU baseline shared across targets.",'',
                    '| Target | K | Actual speedup | RGB PSNR | SSIM | LPIPS | Candidate VBench |','|---:|---:|---:|---:|---:|---:|---:|']
            report += [f"| {r['target_speedup']} | {r['K']} | {r['actual_speedup']:.4f} | {r['psnr_rgb_db']:.4f} | {r['ssim']:.5f} | {r['lpips']:.5f} | {r['candidate_vbench']:.6f} |" for r in rows]
            report += ['', 'Per-target evaluations retain RGB PSNR/SSIM/LPIPS, component latency/TFLOPs and VBench details in summary.json. Custom scores are 10-dimension diagnostics, not VBench200 aggregate scores. The SeaCache target calibration excludes Ours controller overhead; actual speedup above is measured full generate.']
            (output/'REPORT.md').write_text('\n'.join(report)+'\n')
            write(output/'COMPLETE.json',dict(status='complete',summary_sha256=sha256(output/'summary.json')))
            write(output/'STATUS.json',dict(state='complete'))
        except BaseException as e:
            write(output/'STATUS.json',dict(state='failed',error=repr(e)));raise


if __name__=='__main__':run(parser().parse_args())
