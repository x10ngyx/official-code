#!/usr/bin/env python3
"""Drain existing inference workers, then replace a paused coordinator without VBench."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ours4wan21.contracts import dump
from pipeline_lib import read_json, validate_generation


def process(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(state=fields[0], start=fields[19])
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coordinator-pid', type=int, required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root.resolve(strict=True)
    parent = process(args.coordinator_pid)
    if parent is None or parent['state'] != 'T':
        raise RuntimeError('coordinator must be paused before handover')
    proc = Path('/proc') / str(args.coordinator_pid)
    command = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
    if not any(x.endswith('run_pipeline.py') for x in command):
        raise RuntimeError('unexpected coordinator command')
    env = dict(x.decode().split('=', 1) for x in (proc / 'environ').read_bytes().split(b'\0') if x)
    workers = []
    # subprocess children may belong to any one of the coordinator's threads.
    pids = set()
    for children in proc.glob('task/*/children'):
        pids.update(int(x) for x in children.read_text().split())
    config = read_json(root / 'config.json')
    for pid in sorted(pids):
        wp = Path('/proc') / str(pid)
        argv = [x.decode() for x in (wp / 'cmdline').read_bytes().split(b'\0') if x]
        if 'generate.py' not in argv:
            raise RuntimeError(f'unexpected worker {pid}')
        output = Path(argv[argv.index('--output-dir') + 1]).resolve()
        if not output.is_relative_to(root / 'candidates'):
            raise RuntimeError('worker output outside candidate scope')
        workers.append(dict(pid=pid, start=process(pid)['start'], output=str(output),
            mode=argv[argv.index('--state-mode') + 1], k=int(argv[argv.index('--skip-budget') + 1])))
    dump(root / 'VBENCH_SKIPPED_BY_USER.json', dict(status='skipped_by_user',
        video_metrics_enabled=True, reason='User requested no VBench scoring for this diagnostic test'))
    record = dict(status='draining_inference', coordinator_pid=args.coordinator_pid, workers=workers,
                  vbench_enabled=False, video_metrics_enabled=True)
    dump(root / 'HANDOVER.json', record)
    deadline = time.monotonic() + 3600
    while True:
        states = [process(w['pid']) for w in workers]
        if all(s is None or s['start'] != w['start'] or s['state'] == 'Z'
               for w, s in zip(workers, states)):
            break
        if time.monotonic() > deadline:
            raise RuntimeError('workers did not drain within one hour; paused coordinator retained')
        time.sleep(10)
    for worker in workers:
        validate_generation(Path(worker['output']), prompt_ids=config['prompt_subset']['sample_ids'],
            method='ours', mode=worker['mode'], skip_budget=worker['k'])
    current = process(args.coordinator_pid)
    if current is None or current['start'] != parent['start'] or current['state'] != 'T':
        raise RuntimeError('coordinator identity changed during handover')
    # Its workers have exited. Kill only this paused coordinator; never resume old VBench code.
    os.kill(args.coordinator_pid, signal.SIGKILL)
    for _ in range(50):
        current = process(args.coordinator_pid)
        if current is None or current['state'] == 'Z':
            break
        time.sleep(.1)
    else:
        raise RuntimeError('old coordinator did not exit')
    command = [str(Path(__file__).resolve().parent / 'run_pipeline.py') if x.endswith('run_pipeline.py') else x for x in command]
    if '--resume' not in command:
        command.append('--resume')
    command.append('--skip-vbench')
    record['status'] = 'restarting_without_vbench'
    dump(root / 'HANDOVER.json', record)
    completed = subprocess.run(command, cwd=PROJECT, env=env, check=False)
    record['status'] = 'complete' if completed.returncode == 0 else 'pipeline_failed'
    record['returncode'] = completed.returncode
    dump(root / 'HANDOVER.json', record)
    sys.exit(completed.returncode)


if __name__ == '__main__':
    main()
