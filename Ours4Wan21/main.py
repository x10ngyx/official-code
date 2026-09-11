#!/usr/bin/env python3
"""Formal CNN+G1 entry point over the versioned, validated implementation.

Legacy train.py/generate.py remain available for reproducing MLP experiments.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent
TRAIN = PROJECT / 'experiments/cnn_mixed3500_v1'
INFERENCE = PROJECT / 'experiments/cnn_vbench20_v1'


def validate_jobs(jobs):
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('expected a nonempty list of frozen G1 candidate jobs')
    for job in jobs:
        if (job.get('kind') != 'candidate' or job.get('group') != 'G1'
                or job.get('state_mode') != 'cnn_G1'):
            raise ValueError('formal method requires group=G1 and state_mode=cnn_G1')
    return jobs


def verify_checkpoint(selection_path):
    # Reuse the same strict loader as the completed CNN video evaluation.
    sys.path[:0] = [str(INFERENCE), str(TRAIN), str(PROJECT)]
    import torch
    from adapter import Policy
    torch.set_num_threads(1)
    selection = json.loads(selection_path.read_text())
    checkpoint = Path(selection['checkpoint'])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != selection['sha256']:
        raise ValueError('selected checkpoint SHA256 mismatch')
    policy = Policy(checkpoint, device='cpu', state_mode='cnn_G1')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['epoch'] != selection['selected']['epoch']:
        raise ValueError('selected epoch mismatch')
    with torch.no_grad():
        logits = policy.net(torch.zeros(1, 18439))
    if logits.shape != (1, 2) or not torch.isfinite(logits).all():
        raise ValueError('invalid G1 actor output')
    print(json.dumps(dict(method='CNN+G1', state_mode='cnn_G1',
        checkpoint=str(checkpoint), sha256=digest, epoch=payload['epoch'],
        actor_parameters=sum(p.numel() for p in policy.net.parameters()),
        input_dimension=18439, status='verified'), indent=2))


def main():
    exp = Path(os.environ.get('OURS4WAN21_EXP_BASE',
                            os.environ.get('EXP_BASE', '/mnt/hdd/xiongyuxiang/tmp/exp')))
    parser = argparse.ArgumentParser(description='Ours4Wan21 formal method: CNN+G1 (8x8 raw three-state features).')
    commands = parser.add_subparsers(dest='command', required=True)
    verify = commands.add_parser('verify', help='verify selected G1 checkpoint and run a CPU actor forward')
    verify.add_argument('--selection', type=Path,
                        default=exp / 'ours21_cnn_mixed3500_v1/analysis/G1/checkpoint_selection.json')
    commands.add_parser('train', help='resume G1 only in the existing frozen mixed3500 training suite')
    generate = commands.add_parser('generate', help='run frozen G1 candidate jobs using the CNN resident worker')
    generate.add_argument('--jobs', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'verify':
        verify_checkpoint(args.selection)
    elif args.command == 'train':
        subprocess.run([sys.executable, str(TRAIN / 'pipeline.py'),
                        'train_group', '--index', '0'], check=True)
    else:
        validate_jobs(json.loads(args.jobs.read_text()))
        subprocess.run([sys.executable, str(INFERENCE / 'generate_worker.py'),
                        '--jobs', str(args.jobs.resolve())], check=True)


if __name__ == '__main__':
    main()
