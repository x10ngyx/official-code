"""Replace only the stopped old scheduler after its generation workers finish."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import run_suite as suite


def main():
    root=suite.EXP_ROOT/suite.NAME
    scope=suite.read_json(root/'VBENCH_SKIPPED_BY_USER.json')
    if suite.vbench_enabled(root):raise ValueError('explicit skip marker required')
    suite.freeze(resume=True)
    pid=scope['previous_scheduler_pid']
    proc=Path(f'/proc/{pid}')
    expected='experiments/dynamics128_vbench50_4gpu_v1/run_suite.py'
    def identity():
        if not proc.exists():return False
        if expected not in (proc/'cmdline').read_bytes().decode().replace('\0',' '):
            raise ValueError('previous scheduler PID changed identity')
        # /proc stat starttime prevents same-command PID reuse.
        if (proc/'stat').read_text().split(') ',1)[1].split()[19]!=scope['previous_scheduler_starttime']:
            raise ValueError('previous scheduler PID reused')
        return True
    if identity() and (proc/'stat').read_text().split(') ',1)[1].split()[0]!='T':
        raise ValueError('old scheduler must be stopped')
    cfg=suite.read_json(root/'config.json')
    while True:
        pending=[]
        for g in suite.GPUS:
            for label in suite.LABELS:
                d=root/'shards'/f'gpu{g}'/label
                if not (d/'COMPLETE.json').exists():pending.append(str(d))
        suite.dump(root/'HANDOVER.json',dict(status='waiting_for_existing_generation' if pending else 'generation_complete',
            pending=pending,vbench_status='skipped_by_user',updated_at=suite.now()))
        if not pending:break
        time.sleep(15)
    for g in suite.GPUS:
        for label in suite.LABELS:
            k=None if label=='baseline' else int(label[1:])
            suite.validate_generation(root/'shards'/f'gpu{g}'/label,prompt_ids=cfg['shard_ids'][str(g)],
                method='baseline' if k is None else 'ours',mode=suite.MODE,skip_budget=k)
    # Let the last worker exit after publishing its COMPLETE marker.
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        time.sleep(5)
    if identity():
        os.kill(pid,signal.SIGTERM)
        os.kill(pid,signal.SIGCONT)  # deliver pending TERM; do not resume old workflow
        for _ in range(60):
            if not proc.exists():break
            time.sleep(1)
        if proc.exists():raise RuntimeError('old scheduler did not exit; refusing competing launch')
    suite.dump(root/'HANDOVER.json',dict(status='quality_only_scheduler_started',vbench_status='skipped_by_user',updated_at=suite.now()))
    os.execv(sys.executable,[sys.executable,str(Path(suite.__file__)),'--resume'])


if __name__=='__main__':main()
