#!/usr/bin/env python3
"""Adopt surviving workers; resume only GPU1; preserve frozen original runner."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,fcntl,signal,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vbench200_balanced_gpu123'))
import rebalance_gpu123 as original

def proc(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,state=s[0],start=s[19],cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError:return None

def done(state,actual,expected):
    if state.get('pid')!=expected['pid']:raise RuntimeError('Worker status PID changed')
    if actual and actual['start']!=expected['start']:raise RuntimeError('Worker PID reused')
    if state.get('status')=='failed':raise RuntimeError('Worker failed: '+str(state))
    alive=actual is not None and actual['state']!='Z'
    if not alive and state.get('status')!='complete':raise RuntimeError('Worker exited without completion')
    return not alive and state.get('status')=='complete'

def self_test():
    e={'pid':12,'start':'100'};running=dict(e,state='R');zombie=dict(e,state='Z')
    assert not done({'pid':12,'status':'running'},running,e)
    assert not done({'pid':12,'status':'complete'},running,e)
    assert done({'pid':12,'status':'complete'},zombie,e)
    assert done({'pid':12,'status':'complete'},None,e)
    for state,p in [({'pid':13,'status':'complete'},None),({'pid':12,'status':'running'},None),({'pid':12,'status':'failed'},running),({'pid':12,'status':'complete'},dict(zombie,start='101'))]:
        try:done(state,p,e)
        except RuntimeError:pass
        else:raise AssertionError('unsafe adoption accepted')
    print('PASS 8 adoption/completion guards')

def main(root,oldpid):
    original.check_environment();root=original.external_output(root);rdir=root/'gpu1_recovery';rdir.mkdir(exist_ok=True)
    with (root/'.gpu1_recovery.lock').open('a') as lock:
      fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
      old=proc(oldpid)
      assert old and old['state']=='T' and 'rebalance_gpu123.py' in old['cmd'] and '--worker' not in old['cmd'] and str(root) in old['cmd'],old
      contexts=original.target_context(root);assignment=original.read_json(root/'balanced_assignment.json')
      for record in assignment['experiment_source'].values():assert original.sha256(record['path'])==record['sha256']
      expected={};before={}
      for gpu in ('2','3'):
        state=original.read_json(root/'balanced_worker_status'/f'gpu_{gpu}.json');actual=proc(state['pid']);assert actual and str(root) in actual['cmd'] and f'--worker {gpu}' in actual['cmd'];done(state,actual,actual);expected[gpu]=actual;before[gpu]=state
      assert not subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU1 busy'
      valid=[]
      for job in assignment['assignments']['1']:
        c=contexts[job['target']];path=original.job_directory(c['root'],c['prompts'][job['sample_id']],c['condition'])/'manifest.json'
        if path.exists():original.validate_any_gpu(path,c);valid.append(original.artifact(path))
      original.atomic_json(rdir/'PRECHECK.json',dict(status='pass',old_coordinator=old,adopted_workers=expected,before=before,gpu1_validated_completed=len(valid),gpu1_remaining=len(assignment['assignments']['1'])-len(valid),completed_manifests=valid,source_hashes='pass',assignment=original.artifact(root/'balanced_assignment.json')))
      worker,log=original.spawn(root,['--worker','1','--resume'],rdir/'worker_gpu1.log','1');expected['1']=proc(worker.pid);assert expected['1']
      record=dict(status='running',phase='generation',coordinator_pid=os.getpid(),old_coordinator=old,workers=expected,gpu1_preexisting_completed=len(valid),script=original.artifact(Path(__file__).resolve()))
      original.atomic_json(rdir/'RECOVERY.json',record)
      original.atomic_json(root/'status.json',dict(status='running',phase='balanced_generation_recovered_gpu1',coordinator_pid=os.getpid(),recovery=str(rdir/'RECOVERY.json')))
      try:
        while True:
          all_done=True;states={};worker.poll()
          for gpu,e in expected.items():
            state=original.read_json(root/'balanced_worker_status'/f'gpu_{gpu}.json')
            if gpu=='1' and state.get('pid')!=e['pid']:
                assert worker.poll() is None,'GPU1 failed before status initialization';all_done=False;states[gpu]={'status':'validating_existing_results','pid':e['pid']};continue
            complete=done(state,proc(e['pid']),e);all_done &= complete;states[gpu]=state
          paused=proc(oldpid);assert paused and paused['start']==old['start'] and paused['state']=='T','Old coordinator unexpectedly changed'
          original.atomic_json(rdir/'HEARTBEAT.json',dict(status='running',phase='generation',time_unix=time.time(),coordinator_pid=os.getpid(),workers=states,old_coordinator_safely_paused=True))
          if all_done:break
          time.sleep(10)
        assert worker.wait()==0;log.close()
        # Retire stale parent only after all GPU generation workers have exited.
        for gpu,e in expected.items():assert done(states[gpu],proc(e['pid']),e)
        os.kill(oldpid,signal.SIGKILL)
        with (root/'.balanced_runner.lock').open('a') as balanced_lock:
          # Death releases this lock; retry briefly for kernel/parent cleanup.
          deadline=time.monotonic()+15
          while True:
            try:fcntl.flock(balanced_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
              if time.monotonic()>deadline:raise
              time.sleep(.1)
          for c in contexts.values():
            for row in c['prompts'].values():original.validate_any_gpu(original.job_directory(c['root'],row,c['condition'])/'manifest.json',c)
          original.atomic_json(root/'status.json',dict(status='running',phase='parallel_quality_evaluation',coordinator_pid=os.getpid()))
          processes=[]
          for target in original.TARGET_DIRS:
            gpu=str(contexts[target]['plan']['gpu_ids'][0]);p,l=original.spawn(root,['--finalize-target',target],rdir/f'finalize_{target}.log',gpu);processes.append((target,p,l))
          while any(p.poll() is None for _,p,_ in processes):
            original.atomic_json(rdir/'HEARTBEAT.json',dict(status='running',phase='parallel_quality_evaluation',time_unix=time.time(),finalizers={n:dict(pid=p.pid,returncode=p.poll()) for n,p,_ in processes}));time.sleep(10)
          for name,p,l in processes:l.close()
          failures=[(n,p.returncode) for n,p,_ in processes if p.returncode];assert not failures,failures
          original.complete_root(root)
          original.atomic_json(rdir/'RECOVERY.json',dict(record,status='complete',phase='complete'))
          original.atomic_json(rdir/'HEARTBEAT.json',dict(status='complete',phase='complete',time_unix=time.time()))
      except BaseException:
        original.atomic_json(rdir/'RECOVERY.json',dict(record,status='failed',error=traceback.format_exc()))
        original.atomic_json(root/'status.json',dict(status='failed',phase='recovery_supervision',error=traceback.format_exc(),healthy_workers_preserved=True));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path);ap.add_argument('--old-coordinator',type=int,default=2099428);ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    if a.self_test:self_test()
    else:main(a.output_dir,a.old_coordinator)
