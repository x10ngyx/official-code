#!/usr/bin/env python3
"""Wan21 single-skip causal dataset. All public step numbers are ONE-based."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import sys
from common import WORKSPACE, COARSE, REMAINING, external, prompts, steps, read_json, source_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','run','evaluate','vbench','summarize','audit','publish'])
    p.add_argument('--prompts',type=Path,help='JSONL: sample_id, prompt_en or prompt; exactly 40 rows by default')
    p.add_argument('--expected-prompts',type=int,default=40,help='use 1 only for remote smoke validation')
    p.add_argument('--stage',choices=['coarse','remaining','all'],default='coarse')
    p.add_argument('--output',type=Path)
    p.add_argument('--wan21-root',type=Path)
    p.add_argument('--checkpoint-dir',type=Path,default=WORKSPACE/'models/Wan2.1-T2V-1.3B')
    p.add_argument('--profile',type=Path,help='locked Wan21Benchmark Calflops component profile JSON')
    p.add_argument('--max-new-cells',type=int,help='graceful stop after N newly completed cells')
    p.add_argument('--device',default='cuda:0',help='quality evaluator device; run after releasing generation GPU')
    args = p.parse_args()
    if args.expected_prompts<1 or (args.max_new_cells is not None and args.max_new_cells<1):
        p.error('counts must be positive')
    os.environ['TORCH_HOME'] = str(WORKSPACE/'models/torch-cache')
    os.environ['HF_HOME'] = str(WORKSPACE/'models/huggingface')
    os.environ['XDG_CACHE_HOME'] = str(WORKSPACE/'models/xdg')
    if args.action in ['plan','run']:
        if args.prompts is None:
            p.error('--prompts required')
        rows = prompts(args.prompts,args.expected_prompts)
    if args.action=='plan':
        print(json.dumps(dict(numbering='1-based execution step',coarse=COARSE,remaining=REMAINING,
            baseline_videos=len(rows),coarse_candidates=len(rows)*len(COARSE),
            remaining_candidates=len(rows)*len(REMAINING),all_candidates=len(rows)*49,
            total_dataset_videos=len(rows)*50,selected_stage=args.stage,
            order='all baseline prompts; then each selected step across all prompts',
            first_jobs=[dict(step=s,prompt_id=r['sample_id']) for s in [0]+steps(args.stage)[:1] for r in rows],
            storage_rgb_f32_gib=len(rows)*50*81*3*480*832*4/1024**3,
            storage_latents_fp16_gib=len(rows)*50*50*16*21*60*104*2/1024**3),indent=2))
        return
    if args.output is None:
        p.error('--output required')
    args.output = external(args.output)
    if args.action=='run':
        if args.wan21_root is None or args.profile is None:
            p.error('--wan21-root and --profile required')
        from collect import collect
        collect(args,rows)
    else:
        if Path(sys.prefix).name != 'wan2.2':
            raise ValueError('activate wan2.2')
        if read_json(args.output/'run.json')['sources'] != source_hashes():
            raise ValueError('analysis source changed since generation; preserve original code for this run')
        from evaluate import quality, vbench, summarize, audit
        from dataset import publish
        {'evaluate':quality,'vbench':vbench,'summarize':lambda a:summarize(a.output),'audit':audit,'publish':publish}[args.action](args)


if __name__=='__main__':
    main()
