"""Overlap the remaining single-GPU training with three ready-group evaluators."""
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
from dispatch import *
import suite

HERE=Path(__file__).resolve().parent

def setup(config):
    status=read(ROOT/'STATUS.json');parent=status['pid']
    assert b'iql_aggressiveness_2x4_v1/suite.py' in Path(f'/proc/{parent}/cmdline').read_bytes()
    remaining=next(g for g in config['groups'] if g['name']=='sea7_a3')
    children=child_pids(parent)
    candidates=[int(p) for p in children if b'train.py' in Path(f'/proc/{p}/cmdline').read_bytes()
        and remaining['training'].encode() in Path(f'/proc/{p}/cmdline').read_bytes()]
    assert len(candidates)==1,'expected the already-running SEA7 A3 trainer'
    child=candidates[0]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    for row in apps.splitlines():
        pid,uuid=[v.strip() for v in row.split(',')]
        assert int(pid)==child and uuid==config['gpu_uuids'][3],('unexpected GPU workload',row)
    OVERLAP.mkdir();(OVERLAP/'README.md').write_text((HERE/'README.md').read_text())
    dump(OVERLAP/'previous_STATUS.json',status)
    metadata=dict(parent_pid=parent,parent_starttime=Path(f'/proc/{parent}/stat').read_text().split()[21],
        training_pid=child,training_starttime=Path(f'/proc/{child}/stat').read_text().split()[21],
        adopted_group='sea7_a3',source_hashes={str(p):sha256(p) for p in [*sorted(HERE.glob('*.py')),HERE/'README.md']},
        base_config_sha256=sha256(ROOT/'config.json'),expected_counts=[72,72,48,48],
        request='Three idle GPUs evaluate selected groups concurrently with the remaining SEA7 A3 training')
    dump(OVERLAP/'CONFIG.json',metadata)
    # Preserve the trainer; prevent its old parent from later dispatching duplicate evaluators.
    os.kill(parent,signal.SIGSTOP)
    for folder in ('evaluation/candidates','evaluation/quality','evaluation/quality_inputs'):
        d=ROOT/folder;d.mkdir(exist_ok=True);(d/'README.md').write_text('# Evaluation artifacts\n\nFrozen 10-prompt, eight-group, three-budget evaluation; per-video identities and hashes required.\n')
    for g in config['groups']:
        d=ROOT/'evaluation/candidates'/g['name'];d.mkdir(exist_ok=True)
        (d/'README.md').write_text('# '+g['name']+'\n\nThree budgets and ten sealed videos per budget.\n')
        for k in BUDGETS:
            p=d/f'K{k}';p.mkdir(exist_ok=True);(p/'README.md').write_text(f'# K{k}\n\nSame ten frozen prompts.\n')
    queues,ready=publish_jobs(config)
    assert len(ready)==7 and [len(queues[g]) for g in range(4)]==[63,63,42,42]
    return metadata

def execute(config,gpu,label,cmd):
    verify_overlap()
    with (ROOT/'logs'/f'overlap_{label}.log').open('a') as f:
        subprocess.run(cmd,cwd=PROJECT,env=environment(config['gpu_uuids'][gpu]),
            stdout=f,stderr=subprocess.STDOUT,check=True)

def evaluate_gpu(config,metadata,gpu):
    with suite.acquire_gpu_lock(EXP_ROOT/f'.ours21_gpu{gpu}.lock'):
        dump(OVERLAP/f'gpu{gpu}_STATUS.json',dict(status='generation',expected=metadata['expected_counts'][gpu]))
        execute(config,gpu,f'generate_gpu{gpu}',[sys.executable,str(HERE/'worker.py'),
            '--jobs',str(ROOT/'jobs'/f'gpu{gpu}.json'),'--expected',str(metadata['expected_counts'][gpu])])
        jobs=read(ROOT/'jobs'/f'gpu{gpu}.json');assert len(jobs)==metadata['expected_counts'][gpu]
        dump(OVERLAP/f'gpu{gpu}_STATUS.json',dict(status='quality',videos=len(jobs)))
        suite.quality_gpu(config,gpu,jobs)
        dump(OVERLAP/f'gpu{gpu}_STATUS.json',dict(status='complete',videos=len(jobs)))

def finish_training_then_evaluate(config,metadata,held):
    g=next(g for g in config['groups'] if g['name']=='sea7_a3');training=Path(g['training'])
    dump(OVERLAP/'gpu3_STATUS.json',dict(status='training',adopted_pid=metadata['training_pid']))
    while not (training/'TRAINING_COMPLETE.json').exists():
        if (training/'FAILED.json').exists():raise RuntimeError('remaining SEA7 A3 training failed')
        if suite.process_state(metadata['training_pid'],metadata['training_starttime']) in (None,'Z'):
            raise RuntimeError('remaining trainer exited before completion')
        time.sleep(10)
    verify_training(g)
    while suite.process_state(metadata['training_pid'],metadata['training_starttime']) not in (None,'Z'):time.sleep(.1)
    if suite.process_state(metadata['parent_pid'],metadata['parent_starttime']) is not None:
        assert b'iql_aggressiveness_2x4_v1/suite.py' in Path(f"/proc/{metadata['parent_pid']}/cmdline").read_bytes()
        os.kill(metadata['parent_pid'],signal.SIGKILL)
    # Wait for kernel lock release; keep the suite guard until every evaluator and report finish.
    held.append(suite.acquire_gpu_lock(ROOT/'suite.lock'))
    dump(OVERLAP/'gpu3_STATUS.json',dict(status='checkpoint_selection_and_q'))
    suite.train_gpu(config,3)
    queues,ready=publish_jobs(config)
    assert len(ready)==8 and sum(map(len,queues.values()))==240
    dump(ROOT/'TRAINING_COMPLETE.json',dict(status='complete',groups=8,epochs=3200,checkpoints=3200,
        selections={g['name']:read(ROOT/'groups'/g['name']/'TRAINING_COMPLETE.json') for g in config['groups']}))
    dump(ROOT/'STATUS.json',dict(status='running',stage='overlapped_evaluation',pid=os.getpid()))
    evaluate_gpu(config,metadata,3)

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    config=read(ROOT/'config.json');verify_sources(config)
    metadata=read(OVERLAP/'CONFIG.json') if a.resume else setup(config)
    verify_overlap();assert sha256(ROOT/'config.json')==metadata['base_config_sha256']
    lock=(OVERLAP/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    held=[]
    if a.resume and (OVERLAP/'FAILED.json').exists():
        (OVERLAP/'FAILED.json').rename(OVERLAP/f'previous_failure_{time.time_ns()}.json')
    def guarded(fn,*args):
        try:return fn(*args)
        except BaseException as exc:
            failure=dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),time=time.time())
            dump(OVERLAP/'FAILED.json',failure);dump(ROOT/'STATUS.json',failure)
            raise
    try:
        dump(ROOT/'STATUS.json',dict(status='running',stage='training_and_evaluation_overlap',pid=os.getpid()))
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(guarded,evaluate_gpu,config,metadata,gpu) for gpu in range(3)]
            futures.append(pool.submit(guarded,finish_training_then_evaluate,config,metadata,held))
            for f in futures:f.result()
        dump(ROOT/'STATUS.json',dict(status='running',stage='report',pid=os.getpid()))
        execute(config,0,'report',[sys.executable,str(BASE/'report.py')])
        done=read(ROOT/'COMPLETE.json');assert done['status']=='complete' and done['candidates']==240
        dump(ROOT/'STATUS.json',done);dump(OVERLAP/'COMPLETE.json',dict(status='complete',candidates=240))
    except BaseException as exc:
        failure=dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),time=time.time())
        dump(OVERLAP/'FAILED.json',failure);dump(ROOT/'STATUS.json',failure)
        raise
    finally:
        for f in held:f.close()

if __name__=='__main__':main()
