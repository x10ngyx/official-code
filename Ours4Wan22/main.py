#!/usr/bin/env python3
"""Ours4Wan22 formal CNN+G1 method entry point."""
import os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    verify = commands.add_parser('verify',help='verify source locks and a Wan22 CNN checkpoint on CPU')
    verify.add_argument('--policy',type=Path,required=True)
    gen = commands.add_parser('generate',help='persistent batch with one warmup; native baseline or learned candidate')
    gen.add_argument('--wan22-root',type=Path,required=True)
    gen.add_argument('--checkpoint-dir',type=Path,required=True)
    gen.add_argument('--jobs',type=Path,required=True)
    gen.add_argument('--profile',type=Path,required=True)
    gen.add_argument('--policy',type=Path)
    gen.add_argument('--skip-budget',type=int)
    gen.add_argument('--target-speedup',type=float,help='target speedup; default full-generate calibration supports 1.5–3.5x')
    gen.add_argument('--calibration',type=Path,help='override the bundled 20260912 SeaCache full-generate speed-to-K calibration')
    gen.add_argument('--output',type=Path,required=True)
    evaluate = commands.add_parser('evaluate',help='paired VideoMetrics + VBench + full performance report')
    evaluate.add_argument('--baseline',type=Path,required=True)
    evaluate.add_argument('--candidate',type=Path,required=True)
    evaluate.add_argument('--output',type=Path,required=True)
    evaluate.add_argument('--metric-models',type=Path,required=True)
    evaluate.add_argument('--device',default='cuda:0')
    evaluate.add_argument('--vbench-mode',choices=('vbench200','custom'),default='vbench200')
    train = commands.add_parser('train',help='offline IQL from explicitly split raw CNN+G1 trajectories')
    train.add_argument('--dataset',type=Path,required=True)
    train.add_argument('--weights',type=Path,required=True)
    train.add_argument('--output',type=Path,required=True)
    train.add_argument('--device',default='cuda')
    train.add_argument('--epochs',type=int,default=200)
    train.add_argument('--batch-size',type=int,default=256)
    train.add_argument('--smoke-only',action='store_true',help='mark synthetic validation checkpoints as nondeployable')
    profile = commands.add_parser('profile',help='shared SeaCache DiT/T5/VAE Calflops; forward remaining arguments')
    args,extra = parser.parse_known_args()
    if extra and args.command != 'profile':
        parser.error('unrecognized arguments: '+' '.join(extra))
    if Path(sys.prefix).name != 'wan2.2':
        raise ValueError('run in conda environment wan2.2')
    import torch
    torch.set_num_threads(1)
    from ours4wan22.shared import OFFICIAL,verify_sources,sha256
    verify_sources()
    if args.command == 'verify':
        from ours4wan22.policy import Policy
        policy = Policy(args.policy,device='cpu')
        action,prob = policy.choose(torch.zeros(18439))
        print(dict(status='verified',method='Wan22 CNN+G1',actor_parameters=sum(p.numel() for p in policy.net.parameters()),
                   checkpoint_sha256=sha256(args.policy),action=action,p_skip=prob))
    elif args.command == 'profile':
        subprocess.run([sys.executable,str(OFFICIAL/'SeaCache4Wan22/experiments/performance_t2v_a14b/profile_calflops.py'),*extra],check=True)
    else:
        if args.command == 'generate':
            from ours4wan22.generation import run
        elif args.command == 'evaluate':
            from ours4wan22.evaluation import run
        else:
            if args.epochs < 1 or args.batch_size < 1:
                parser.error('positive epochs and batch size required')
            from ours4wan22.training import run
        print(run(args))


if __name__ == '__main__':
    main()
