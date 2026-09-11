"""Four-GPU training, validation selection, resident evaluation and reporting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from common import *
from ours4wan21.contracts import create_result

HERE=Path(__file__).resolve().parent
SCHEDULE={0:['sea7_a1','dynamics128_a1'],1:['sea7_a2','dynamics128_a2'],
          2:['sea7_a4','dynamics128_a4'],3:['dynamics128_a3','sea7_a3']}

def initialize():
    reference=load_evaluation_bundle(REFERENCE);uuids=gpu_uuids(['0','1','2','3'])
    prompts=freeze_prompts(reference,uuids)
    groups=[group(f,a) for f in FEATURES for a in LEVELS]
    inputs={}
    for f in FEATURES:
        g=group(f,'a1');cache=Path(g['cache']);marker=read(cache/'COMPLETE.json')
        digest=sha256(cache/'transitions.pt')
        assert marker['status']=='complete' and marker['sha256']==digest
        assert read(cache/'manifest.json')==read(Path(g['baseline'])/'dataset_manifest.json')
        inputs[f]=dict(path=str(cache/'transitions.pt'),sha256=digest)
    flops=read(OLD/'config.json')['flops_profile']
    inputs['flops_profile']=dict(path=flops,sha256=sha256(flops))
    inputs['reference']=dict(path=str(REFERENCE/'manifest.json'),sha256=sha256(REFERENCE/'manifest.json'))
    adoption=read(LEGACY/'SUPERSEDED.json')
    assert adoption['replacement']==str(ROOT)
    running=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    for row in running.splitlines():
        pid,uuid=[v.strip() for v in row.split(',')]
        assert int(pid)==adoption['training_pid'] and uuid==uuids[3],('GPU already occupied',row)
    adopted=group('dynamics128','a3');actual=read(Path(adopted['training'])/'config.json')
    assert all(actual[k]==v for k,v in adopted['training_config'].items())
    sources=[*sorted((PROJECT/'ours4wan21').glob('*.py')),*sorted(HERE.glob('*.py')),HERE/'README.md',
        OFFICIAL/'SeaCache4Wan21/seacache.py',OFFICIAL/'SeaCache4Wan21/wan21_integration.py',PROJECT/'local_training_lock.json']
    config=dict(schema='ours21_iql_aggressiveness_2x4_v1',protocol=PROTOCOL,groups=groups,
        targets=TARGETS,skip_budgets=BUDGETS,vbench_enabled=False,reference=str(REFERENCE),
        prompt_selection='seed42; within baseline GPU strata take 3/3/2/2 of frozen online20; preserve source order',
        prompts=prompts,gpu_uuids=uuids,schedule=SCHEDULE,adoption=adoption,inputs=inputs,
        source_hashes={str(p):sha256(p) for p in sources},python=sys.executable,
        paths=dict(wan21_root=str(WORKSPACE/'data/source/Wan2.1-65386b2'),
            wan_checkpoint=str(MODEL_ROOT/'Wan2.1-T2V-1.3B'),flops_profile=flops))
    create_result(ROOT,(HERE/'README.md').read_text())
    dump(ROOT/'config.json',config)
    for folder in ('groups','logs','jobs','evaluation','figures'):
        d=ROOT/folder;d.mkdir();(d/'README.md').write_text(f'# {folder}\n\nSee ../README.md and config.json for the frozen two-feature/four-level suite.\n')
    for g in groups:
        d=ROOT/'groups'/g['name'];d.mkdir();(d/'README.md').write_text(f"# {g['name']}\n\n{g['profile']}, 400 epochs. Links point to external training, analysis, cache and model weights.\n")
        for key in ('training','analysis','cache','weights'):(d/key).symlink_to(Path(g[key]).resolve(),target_is_directory=True)
        dump(d/'config.json',g)
    return config

def process_state(pid,start):
    p=Path(f'/proc/{pid}/stat')
    if not p.exists():return None
    fields=p.read_text().split()
    if fields[21]!=str(start):return None
    return fields[2]

def acquire_gpu_lock(path,timeout=10.,poll=.05):
    """Allow a terminated previous owner time to release its open file descriptor."""
    lock=Path(path).open('a');deadline=time.monotonic()+timeout
    try:
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                return lock
            except BlockingIOError:
                if time.monotonic()>=deadline:
                    raise TimeoutError(f'GPU lock remains held after {timeout}s: {path}')
                time.sleep(poll)
    except BaseException:
        lock.close()
        raise

def finish_adopted(config,g):
    a=config['adoption'];training=Path(g['training'])
    while not (training/'TRAINING_COMPLETE.json').exists():
        if (training/'FAILED.json').exists():raise RuntimeError('adopted training failed')
        if process_state(a['training_pid'],a['training_starttime']) in (None,'Z'):
            raise RuntimeError('adopted training exited before completion')
        time.sleep(10)
    verify_training(g)
    while process_state(a['training_pid'],a['training_starttime']) not in (None,'Z'):time.sleep(1)
    # The stopped old parent must never dispatch its obsolete online20 evaluation.
    if process_state(a['parent_pid'],a['parent_starttime']) is not None:
        assert b'dynamics128_aggressive_iql_v1/run.py' in Path(f"/proc/{a['parent_pid']}/cmdline").read_bytes()
        os.kill(a['parent_pid'],signal.SIGKILL)
    dump(LEGACY/'STATUS.json',dict(status='training_complete_adopted_by_suite',replacement=str(ROOT),
        obsolete_evaluation20='superseded_by_user_10prompt_suite'))

def execute(config,label,command,gpu):
    verify_sources(config)
    with (ROOT/'logs'/f'{label}.log').open('a') as stream:
        subprocess.run(command,cwd=PROJECT,env=environment(config['gpu_uuids'][gpu]),
            stdout=stream,stderr=subprocess.STDOUT,check=True)

def train_gpu(config,gpu):
    lock=None
    for name in SCHEDULE[gpu]:
        g=next(g for g in config['groups'] if g['name']==name);d=ROOT/'groups'/name
        if g['adopted']:
            dump(d/'STATUS.json',dict(status='training',adopted=True))
            finish_adopted(config,g)
        if lock is None:
            lock=acquire_gpu_lock(EXP_ROOT/f'.ours21_gpu{gpu}.lock')
        if not (Path(g['training'])/'TRAINING_COMPLETE.json').exists():
            if Path(g['training']).exists():raise RuntimeError('partial training requires explicit recovery; refusing overwrite '+name)
            dump(d/'STATUS.json',dict(status='training',gpu=gpu))
            execute(config,name+'_train',[sys.executable,'train.py','--dataset',g['cache'],'--state-mode',g['mode'],
                '--output-dir',g['training'],'--checkpoint-dir',g['weights'],'--device','cuda',
                '--training-seed','42','--iql-profile',g['profile']],gpu)
        verify_training(g)
        dump(d/'STATUS.json',dict(status='checkpoint_selection',gpu=gpu))
        if not (Path(g['analysis'])/'checkpoint_selection.json').exists():
            execute(config,name+'_selection',[sys.executable,'analyze_training.py','--training-result',g['training'],
                '--dataset',g['cache'],'--checkpoint-dir',g['weights'],'--output-dir',g['analysis'],'--device','cuda'],gpu)
        selected=read(Path(g['analysis'])/'checkpoint_selection.json')
        assert 300<selected['checkpoint_epoch']<400 and sha256(selected['checkpoint'])==selected['checkpoint_sha256']
        dump(d/'STATUS.json',dict(status='q_analysis',gpu=gpu,selected_epoch=selected['checkpoint_epoch']))
        if not (Path(g['analysis'])/'extra/COMPLETE.json').exists():
            execute(config,name+'_q',[sys.executable,str(HERE/'q_worker.py'),'--group',name],gpu)
        dump(d/'TRAINING_COMPLETE.json',dict(status='complete',epochs=400,selected_epoch=selected['checkpoint_epoch'],
            checkpoint=selected['checkpoint'],checkpoint_sha256=selected['checkpoint_sha256'],gate=selected['gate']))
        dump(d/'STATUS.json',dict(status='training_and_analysis_complete'))
    if lock is not None:lock.close()

def evaluation_jobs(config):
    queues={g:[] for g in range(4)}
    for group_config in config['groups']:
        selected=read(Path(group_config['analysis'])/'checkpoint_selection.json')
        for k in BUDGETS:
            for row in config['prompts']:
                gpu=config['gpu_uuids'].index(row['baseline_gpu_uuid'])
                output=ROOT/'evaluation/candidates'/group_config['name']/f'K{k}'/row['sample_id']
                queues[gpu].append(dict(kind='candidate',group=group_config['name'],state_mode=group_config['mode'],
                    sample_id=row['sample_id'],prompt=row['prompt'],skip_budget=k,expected_gpu_uuid=row['baseline_gpu_uuid'],
                    checkpoint=dict(path=selected['checkpoint'],sha256=selected['checkpoint_sha256']),output=str(output)))
    assert sum(map(len,queues.values()))==240
    for gpu,jobs in queues.items():dump(ROOT/'jobs'/f'gpu{gpu}.json',jobs)
    for p in (ROOT/'evaluation/candidates',ROOT/'evaluation/quality',ROOT/'evaluation/quality_inputs'):
        p.mkdir(exist_ok=True);(p/'README.md').write_text('# Evaluation artifacts\n\n240 candidate pairs, same ten prompts across eight groups. Per-video seals and source hashes are mandatory.\n')
    for name in [g['name'] for g in config['groups']]:
        p=ROOT/'evaluation/candidates'/name;p.mkdir(exist_ok=True)
        (p/'README.md').write_text('# '+name+'\n\nThree fixed skip budgets, ten sealed candidate videos each.\n')
        for k in BUDGETS:
            q=p/f'K{k}';q.mkdir(exist_ok=True);(q/'README.md').write_text(f'# K{k}\n\nTen measured candidates.\n')
    return queues

def quality_gpu(config,gpu,jobs):
    from ours4wan21.online_common import verified
    from generate_worker import job_identity
    d=ROOT/'evaluation/quality_inputs'/f'gpu{gpu}';d.mkdir(exist_ok=True)
    (d/'README.md').write_text('# Quality pairs\n\nUnique group-K-prompt IDs link to sealed candidates and reused same-GPU native references.\n')
    for name in ('reference','candidate'):(d/name).mkdir(exist_ok=True)
    for j in jobs:
        assert verified(j['output'],job_identity(j,config))
        key=f"{j['group']}_K{j['skip_budget']}_{j['sample_id']}"
        for side,target in [('reference',REFERENCE/'baselines'/j['sample_id']/'video.mp4'),('candidate',Path(j['output'])/'video.mp4')]:
            link=d/side/(key+'.mp4')
            if not link.is_symlink():link.symlink_to(target)
            assert link.resolve()==target.resolve()
    out=ROOT/'evaluation/quality'/f'gpu{gpu}'
    if (out/'summary.json').exists():
        assert read(out/'summary.json')['video_count']==len(jobs)
        return
    execute(config,f'quality_gpu{gpu}',[sys.executable,str(OFFICIAL/'VideoMetrics/evaluate.py'),
        '--reference-dir',str(d/'reference'),'--candidate-dir',str(d/'candidate'),'--expected-frames','81',
        '--device','cuda:0','--model-cache',str(MODEL_ROOT/'torch-cache'),'--output-dir',str(out)],gpu)

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    if 'wan2.2' not in Path(sys.prefix).name.lower():raise ValueError('use Wan2.2 environment')
    config=read(ROOT/'config.json') if a.resume else initialize()
    verify_sources(config)
    suite_lock=(ROOT/'suite.lock').open('a');fcntl.flock(suite_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'COMPLETE.json').exists():return
    try:
        dump(ROOT/'STATUS.json',dict(status='running',stage='training_2x4',pid=os.getpid(),started=time.time()))
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda gpu:train_gpu(config,gpu),range(4)))
        dump(ROOT/'TRAINING_COMPLETE.json',dict(status='complete',groups=8,epochs=3200,checkpoints=3200,
            selections={g['name']:read(ROOT/'groups'/g['name']/'TRAINING_COMPLETE.json') for g in config['groups']}))
        queues=evaluation_jobs(config)
        dump(ROOT/'STATUS.json',dict(status='running',stage='generation',candidates=240))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda gpu:execute(config,f'generate_gpu{gpu}',[sys.executable,str(HERE/'generate_worker.py'),
                '--jobs',str(ROOT/'jobs'/f'gpu{gpu}.json')],gpu),range(4)))
        dump(ROOT/'STATUS.json',dict(status='running',stage='video_quality'))
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda gpu:quality_gpu(config,gpu,queues[gpu]),range(4)))
        dump(ROOT/'STATUS.json',dict(status='running',stage='report'))
        execute(config,'report',[sys.executable,str(HERE/'report.py')],0)
        done=read(ROOT/'COMPLETE.json');assert done['status']=='complete' and done['candidates']==240
        dump(ROOT/'STATUS.json',done)
    except BaseException as exc:
        dump(ROOT/'FAILED.json',dict(error=repr(exc),time=time.time(),traceback=traceback.format_exc()))
        dump(ROOT/'STATUS.json',dict(status='failed',error=repr(exc)))
        raise

if __name__=='__main__':main()
