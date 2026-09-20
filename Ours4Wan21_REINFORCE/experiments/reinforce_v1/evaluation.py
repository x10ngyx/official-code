"""Held-out final-policy evaluation, with temporary videos and durable metrics."""
import fcntl
import os
from pathlib import Path
import random
import re
import subprocess
import sys

import torch

from artifacts import (EXP_ROOT, MODEL_ROOT, OFFICIAL, directory, dump, read, sha, under)
from core import validate_checkpoint


def evaluate_run(args):
    from run import checked_config, WorkerPool
    run = under(args.run, EXP_ROOT)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.name):
        raise ValueError('invalid evaluation name')
    if args.count < 1 or not args.budgets or len(set(args.budgets)) != len(args.budgets) or any(not 20 <= k <= 40 for k in args.budgets):
        raise ValueError('positive evaluation count and distinct K in 20..40 required')
    with (run / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = checked_config(run)
        if config['smoke_only']:
            raise ValueError('GPU smoke is not a production evaluation checkpoint')
        trained = read(run / 'TRAINING_COMPLETE.json')
        checkpoint = Path(trained['checkpoint'])
        if sha(checkpoint) != trained['sha256']:
            raise ValueError('final checkpoint changed')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        validate_checkpoint(payload)
        if payload['run_id'] != config['run_id'] or payload['batch_index'] != config['batches']:
            raise ValueError('evaluation requires final run checkpoint')
        candidates = [r for r in config['prompts'] if r['split'] == args.split]
        if args.count > len(candidates):
            raise ValueError('requested evaluation count exceeds held-out pool')
        prompts = random.Random(42).sample(candidates, args.count)
        out = directory(run / 'evaluation' / args.name, '# Final-policy held-out evaluation. Custom VBench score is a ten-dimension raw mean, not the official full-suite score.')
        plan = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                    split=args.split, prompts=prompts, budgets=args.budgets,
                    generation_seed=42, action='argmax', metrics=['psnr', 'ssim', 'lpips', 'vbench_custom_score'])
        if (out / 'plan.json').exists() and read(out / 'plan.json') != plan:
            raise ValueError('evaluation name already has a different frozen plan')
        dump(out / 'plan.json', plan)
        if (out / 'COMPLETE.json').exists():
            marker = read(out / 'COMPLETE.json')
            for path, value in marker['files'].items():
                if sha(out / path) != value:
                    raise ValueError('completed evaluation artifact changed')
            print('Evaluation already complete and verified.')
            return
        jobs = []
        for k in args.budgets:
            for index, row in enumerate(prompts):
                jid = f'eval_{args.name}_K{k:02d}_{index:04d}'
                jobs.append(dict(**row, id=jid, k=k, worker=index % len(config['gpus']), baseline=None,
                    sampling_seed=None, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                    policy_version=payload['version'], batch_index=payload['batch_index'], evaluation=True,
                    output=f'evaluation/{args.name}/{jid}'))
        pool = WorkerPool(run, config)
        try:
            pool.dispatch(jobs, f'eval_{args.name}')
        finally:
            pool.close()
        results = []
        env = os.environ.copy()
        env.update(PYTHON_BIN=sys.executable, CUDA_VISIBLE_DEVICES=config['gpus'][0]['device'],
                   VBENCH_CACHE_DIR=str(MODEL_ROOT / 'VBench'),
                   TORCH_HOME=str(MODEL_ROOT / 'VBench/torch'),
                   HF_HOME=str(MODEL_ROOT / 'VBench/huggingface'),
                   XDG_CACHE_HOME=str(MODEL_ROOT / 'VBench/xdg'))
        for k in args.budgets:
            group = [j for j in jobs if j['k'] == k]
            scores = {}
            for kind in ('reference', 'candidate'):
                stage = directory(out / f'staging_K{k}_{kind}', '# Temporary symlinks for VBench; removed when scores are complete.')
                prompt_map = {}
                for job in group:
                    suffix = '.reference.mp4' if kind == 'reference' else '.mp4'
                    source = run / '_temporary' / (job['id'] + suffix)
                    link = stage / (job['id'] + '.mp4')
                    if not link.is_symlink():
                        link.symlink_to(source)
                    elif link.resolve() != source.resolve():
                        raise ValueError('VBench staging link differs')
                    prompt_map[link.name] = job['prompt']
                map_path = out / f'prompts_K{k}_{kind}.json'
                dump(map_path, prompt_map)
                score_dir = out / f'vbench_K{k}_{kind}'
                aggregate = score_dir / 'vbench_custom_aggregate_scores.json'
                if not aggregate.exists():
                    if score_dir.exists():
                        # Preserve partial diagnostics; runner itself requires a new work directory.
                        import time
                        score_dir.rename(score_dir.with_name(score_dir.name + f'_incomplete_{time.time_ns()}'))
                    if any(not (stage / name).is_file() for name in prompt_map):
                        raise ValueError('temporary evaluation videos missing; cannot run VBench')
                    subprocess.run(['bash', str(OFFICIAL / 'VbenchEvaluation/run_custom_vbench.sh'),
                        str(stage), str(score_dir), str(map_path)], env=env, check=True)
                scores[kind] = read(aggregate)
            measurement = [read(run / j['output'] / 'measurement.json') for j in group]
            quality = [read(run / j['output'] / 'quality.json')['video'] for j in group]
            results.append(dict(k=k, prompts=len(group),
                psnr=sum(q['psnr_rgb_db_mean'] for q in quality)/len(group),
                ssim=sum(q['ssim_rgb_mean'] for q in quality)/len(group),
                lpips=sum(q['lpips_alex_v0_1_spatial_mean'] for q in quality)/len(group),
                generate_seconds=sum(m['generate_seconds'] for m in measurement)/len(group),
                speedup=sum(m['native_baseline']['generate_seconds'] for m in measurement)
                        / sum(m['generate_seconds'] for m in measurement),
                dit_tflops=sum(m['dit_tflops'] for m in measurement)/len(group),
                vbench=scores))
        dump(out / 'summary.json', dict(results=results, checkpoint_sha256=sha(checkpoint),
            scope='held-out prompt subset, final argmax actor; VBench custom ten-dimension raw mean',
            component_details='per-trajectory measurement.json and timing.json',
            training_normalizer_updated=False))
        # Only after all quality stages succeed; no retained evaluation video.
        for job in jobs:
            for suffix in ('.mp4', '.reference.mp4'):
                (run / '_temporary' / (job['id'] + suffix)).unlink(missing_ok=True)
        for stage in out.glob('staging_*'):
            for link in stage.glob('*.mp4'):
                link.unlink()
        files = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'COMPLETE.json'}
        dump(out / 'COMPLETE.json', dict(status='all_quality_complete', files=files,
                                       videos_retained=False, raw_latents_retained=False))
