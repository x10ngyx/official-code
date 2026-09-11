"""Adopt the four existing workers without restarting generation.

The old orchestrator is stopped while its workers complete. Replace it only
after their exit, so its in-memory VBench stage can never be reached.
"""
import os
from pathlib import Path
import signal
import time
import run


def identity(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:
        return ''


def active(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


if __name__ == '__main__':
    control=run.read(run.ROOT/'VBENCH_SKIPPED_BY_USER.json')
    parent=control['superseded_orchestrator_pid']
    workers=control['existing_worker_pids']
    expected=str(run.HERE/'run.py')
    while any(active(pid) and expected+' worker ' in identity(pid) for pid in workers):
        if expected+' run ' in identity(parent):
            os.kill(parent,signal.SIGSTOP)
        time.sleep(10)
    if expected+' run ' in identity(parent):
        os.kill(parent,signal.SIGTERM)
        os.kill(parent,signal.SIGCONT)
    # run() reacquires the shared lock and revalidates existing completions.
    # Complete workers exit without loading a model; normal audit/mix follows.
    try:
        run.run()
    except Exception as exc:
        run.atomic_json(run.ROOT/'LAST_ERROR.json',dict(error=repr(exc)))
        raise
