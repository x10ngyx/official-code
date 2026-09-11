"""Monotonic ready-checkpoint queues layered over the unchanged frozen suite."""
from pathlib import Path
import sys
import time

BASE=Path(__file__).resolve().parents[1]/'iql_aggressiveness_2x4_v1'
sys.path.insert(0,str(BASE))
from common import *

OVERLAP=ROOT/'overlap'

def child_pids(parent,proc=Path('/proc')):
    """Subprocesses created by executor threads are listed under their owning task."""
    return sorted({int(pid) for p in (proc/str(parent)/'task').glob('*/children')
        for pid in p.read_text().split()})

def verify_overlap():
    config=read(OVERLAP/'CONFIG.json')
    verify_sources(read(ROOT/'config.json'))
    for p,h in config['source_hashes'].items():assert sha256(p)==h,('overlap source changed',p)
    return config

def build_jobs(config):
    queues={gpu:[] for gpu in range(4)};ready=[]
    for g in config['groups']:
        p=Path(g['analysis'])/'checkpoint_selection.json'
        if not p.exists():continue
        selected=read(p)
        assert 300<selected['checkpoint_epoch']<400
        assert sha256(selected['checkpoint'])==selected['checkpoint_sha256']
        ready.append(g['name'])
        for k in BUDGETS:
            for row in config['prompts']:
                gpu=config['gpu_uuids'].index(row['baseline_gpu_uuid'])
                output=ROOT/'evaluation/candidates'/g['name']/f'K{k}'/row['sample_id']
                queues[gpu].append(dict(kind='candidate',group=g['name'],state_mode=g['mode'],
                    sample_id=row['sample_id'],prompt=row['prompt'],skip_budget=k,expected_gpu_uuid=row['baseline_gpu_uuid'],
                    checkpoint=dict(path=selected['checkpoint'],sha256=selected['checkpoint_sha256']),output=str(output)))
    return queues,ready

def publish_jobs(config):
    queues,ready=build_jobs(config)
    for gpu,jobs in queues.items():
        path=ROOT/'jobs'/f'gpu{gpu}.json'
        if path.exists():
            old={j['output']:j for j in read(path)};new={j['output']:j for j in jobs}
            assert all(new.get(key)==job for key,job in old.items()),'queued job identity changed'
        dump(path,jobs)
    dump(OVERLAP/'READY.json',dict(groups=ready,counts={str(g):len(j) for g,j in queues.items()}))
    return queues,ready

def live_jobs(path,expected,poll=5.):
    """Follow atomic job-file appends without repeating or silently changing a job."""
    known={};visited=set()
    while True:
        if (OVERLAP/'FAILED.json').exists():raise RuntimeError('overlap coordinator reported a failure')
        rows=read(path);current={j['output']:j for j in rows}
        if len(current)!=len(rows) or len(rows)>expected:raise ValueError('duplicate/excess evaluation jobs')
        if any(current.get(k)!=v for k,v in known.items()):raise ValueError('queued job removed or changed')
        known=current
        for job in rows:
            if job['output'] not in visited:
                yield job
                visited.add(job['output'])
        if len(visited)==expected:return
        time.sleep(poll)
