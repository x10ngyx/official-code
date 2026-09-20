"""Prepare, train/resume, evaluate, and validate the independent REINFORCE run."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

import torch

from core import (Config, HERE, PROJECT, OFFICIAL, PROTOCOL, new_checkpoint,
                  reinforce_update, validate_checkpoint)
from artifacts import (DEFAULT_BASELINES, EXP_ROOT, MODEL_ROOT, WORKSPACE,
    baseline_references, complete, digest, directory, discard_temporary_videos, dump,
    gpu_info, identity, load_prompts, make_plan, read, save, sha, source_hashes,
    space, under, verify_sources, training_position)


def prepare(args):
    run = under(args.run, EXP_ROOT)
    if not run.name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('run directory needs a simple unique name')
    cfg = Config(seed=args.seed)
    cfg.validate()
    if args.batch_size < 1 or (args.batches is not None and args.batches < 1) or (args.epochs is not None and args.epochs < 1):
        raise ValueError('positive batches/batch size required')
    if args.smoke and (args.epochs is not None or args.batches != 1 or args.batch_size > 2):
        raise ValueError('GPU smoke is one update and at most two trajectories')
    if (run / 'config.json').exists():
        raise ValueError('run already prepared; use train --run to resume')
    prompts, provenance = load_prompts(args.prompts)
    train_count = sum(r['split'] == 'train' for r in prompts)
    if args.epochs is not None:
        args.batches = args.epochs * ((train_count + args.batch_size - 1) // args.batch_size)
    gpus = gpu_info(args.gpus.split(','))
    profile = Path(args.profile).resolve(strict=True)
    fp = read(profile)
    expected = dict(task='t2v-1.3B', video_shape_fhw=[81, 480, 832],
                    sampling_steps=50, solver='unipc', shift=5., cfg=5., seed=42,
                    parameter_dtype='bfloat16', transformer_blocks=30)
    if any(fp['input'].get(k) != v for k, v in expected.items()):
        raise ValueError('FLOPs profile protocol mismatch')
    sys.path.insert(0, str(OFFICIAL / 'ComponentMetrics'))
    from reporting import extract_component_tflops
    extract_component_tflops(fp)
    sys.path.insert(0, str(OFFICIAL / 'Wan21Benchmark'))
    from protocol import source_lock
    wan_root = Path(args.wan_root).resolve(strict=True)
    source_lock(wan_root)
    if fp['source']['wan21_generate_sha256'] != sha(wan_root / 'generate.py'):
        raise ValueError('FLOPs profile source mismatch')
    weights = under(args.wan_weights, MODEL_ROOT)
    if weights.name != 'Wan2.1-T2V-1.3B' or not weights.is_dir():
        raise ValueError('reuse existing Wan2.1-T2V-1.3B model directory')
    weights_hashes = {str(p): sha(p) for p in sorted(weights.rglob('*')) if p.is_file()}
    if not weights_hashes:
        raise ValueError('empty Wan checkpoint directory')
    baselines = baseline_references(None if args.no_baseline_reuse else args.baseline_source,
                                    prompts, gpus, profile,
                                    {str(Path(p).relative_to(weights)): value for p, value in weights_hashes.items()})
    model_dir = under(MODEL_ROOT / run.name, MODEL_ROOT)
    if model_dir.exists():
        raise ValueError('model output name already used')
    config = dict(schema='ours21_reinforce_run_v1', protocol=PROTOCOL,
        training=asdict(cfg), batches=args.batches, batch_size=args.batch_size,
        smoke_only=args.smoke, prompts=prompts, prompt_sources=provenance,
        gpus=gpus, baselines=baselines, profile=str(profile), profile_sha256=sha(profile),
        wan_root=str(wan_root), wan_weights=str(weights), weights_hashes=weights_hashes,
        model_dir=str(model_dir), sources=source_hashes(),
        wan_sources={str(p): sha(p) for p in sorted(wan_root.rglob('*.py'))},
        storage='pooled FP32 raw features + FP16 actor inputs, trace, PSNR, component timing; NO raw latent or retained training video',
        normalizer='identity first batch; cumulative population moments merged AFTER each single optimizer step',
        sampling='iid uniform integer K=20..40 inclusive; iid train prompts with replacement',
        checkpoint_selection='final checkpoint; no validation/test selection',
        training_reward='absolute VideoMetrics decoded RGB per-frame mean PSNR',
        torch_version=torch.__version__, cuda_version=torch.version.cuda)
    if args.epochs is not None:
        config.update(epochs=args.epochs, train_prompts_per_epoch=train_count,
                      sampling='each epoch independently shuffles all train prompts without replacement; iid integer K=20..40')
    config['run_id'] = digest(config)
    directory(run, '# REINFORCE experiment\n\nconfig.json freezes the run. batches/ contains only pooled features and small records; model_weights links to models/. _temporary/ videos are removed after metrics. TRAINING_COMPLETE is not quality-evaluation completion.')
    directory(model_dir, '# Random-initialized CNN+G1 REINFORCE checkpoints and optimizer/moment state.')
    for name, description in [('batches', 'One-policy-version batches.'), ('logs', 'Worker diagnostics.'),
                              ('requests', 'Immutable worker job lists and acknowledgements.'),
                              ('evaluation', 'Held-out evaluation records.')]:
        directory(run / name, '# ' + description)
    dump(run / 'config.json', config)
    save(model_dir / 'batch_0000.pt', new_checkpoint(cfg, config['run_id'], args.smoke))
    (run / 'model_weights').symlink_to(model_dir, target_is_directory=True)
    link = PROJECT / 'experiment_results' / run.name
    if link.exists() or link.is_symlink():
        raise ValueError('result symlink already exists')
    link.symlink_to(run, target_is_directory=True)
    dump(run / 'STATUS.json', dict(stage='prepared', batches=0,
        total_batches=args.batches, progress=training_position(config, 0)))
    print(json.dumps(dict(run=str(run), status='prepared_not_started',
                         baseline_videos_reused=len(baselines), training=asdict(cfg)), indent=2))


class WorkerPool:
    def __init__(self, run, config):
        self.run, self.config = Path(run), config
        self.workers = {}

    def dispatch(self, jobs, label):
        requests = []
        for rank in sorted({j['worker'] for j in jobs}):
            subset = [j for j in jobs if j['worker'] == rank]
            pending = [j for j in subset if not complete(self.run / j['output'], identity(self.config, j))]
            if not pending:
                continue
            request = self.run / 'requests' / f'{label}_gpu{rank}.json'
            ack = request.with_suffix('.ack.json')
            dump(request, pending)
            ack.unlink(missing_ok=True)
            if rank not in self.workers:
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = self.config['gpus'][rank]['device']
                env['TORCH_HOME'] = str(MODEL_ROOT / 'torch-cache')
                log = (self.run / 'logs' / f'worker_{rank}.log').open('a')
                process = subprocess.Popen([sys.executable, str(HERE / 'worker.py'),
                    '--run', str(self.run), '--rank', str(rank)], env=env,
                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
                self.workers[rank] = (process, log)
            process, _ = self.workers[rank]
            process.stdin.write(json.dumps(dict(jobs=str(request), ack=str(ack))) + '\n')
            process.stdin.flush()
            requests.append((process, request, ack))
        while requests:
            remaining = []
            for process, request, ack in requests:
                if ack.exists():
                    receipt = read(ack)
                    if receipt['request_sha256'] != sha(request) or receipt['status'] != 'complete':
                        raise ValueError('worker acknowledgement mismatch')
                elif process.poll() is not None:
                    raise RuntimeError(f'rollout worker failed with exit {process.returncode}; see {self.run / "logs"}')
                else:
                    remaining.append((process, request, ack))
            requests = remaining
            if requests:
                time.sleep(1)
        for job in jobs:
            if not complete(self.run / job['output'], identity(self.config, job)):
                raise ValueError('batch incomplete after worker barrier')

    def close(self):
        for process, log in self.workers.values():
            if process.poll() is None:
                try:
                    process.stdin.write('{"stop": true}\n'); process.stdin.flush()
                    process.wait(timeout=10)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
            if process.stdin:
                process.stdin.close()
            log.close()


def checked_config(run):
    config = read(run / 'config.json')
    if digest({k: v for k, v in config.items() if k != 'run_id'}) != config['run_id']:
        raise ValueError('run configuration changed')
    verify_sources(config)
    for path, value in {**config['weights_hashes'], **config['wan_sources'],
                        **config['prompt_sources'], config['profile']: config['profile_sha256']}.items():
        if sha(path) != value:
            raise ValueError('frozen input changed: ' + path)
    if config['torch_version'] != torch.__version__ or config['cuda_version'] != torch.version.cuda:
        raise ValueError('software versions differ from prepared run')
    return config


def batch_receipt(run, jobs):
    return {j['id']: sha(run / j['output'] / 'episode.pt') for j in jobs}


def commit_update(run, config, batch_index, jobs, device='cpu'):
    model_dir = Path(config['model_dir'])
    previous = model_dir / f'batch_{batch_index:04d}.pt'
    dest = model_dir / f'batch_{batch_index + 1:04d}.pt'
    checkpoint = torch.load(previous, map_location='cpu', weights_only=False)
    validate_checkpoint(checkpoint)
    if checkpoint['run_id'] != config['run_id'] or checkpoint['batch_index'] != batch_index:
        raise ValueError('checkpoint/run mismatch')
    for job in jobs:
        if not complete(run / job['output'], identity(config, job)):
            raise ValueError('cannot update from an incomplete batch')
    receipt = batch_receipt(run, jobs)
    parent_sha = sha(previous)
    if dest.exists():
        result = torch.load(dest, map_location='cpu', weights_only=False)
        validate_checkpoint(result)
        if (result['parent_version'] != checkpoint['version'] or result['parent_sha256'] != parent_sha
                or result['batch_receipt'] != receipt or result['run_id'] != config['run_id']
                or result['batch_index'] != batch_index + 1):
            raise ValueError('completed update does not match parent/batch')
    else:
        episodes = [torch.load(run / j['output'] / 'episode.pt', map_location='cpu', weights_only=False) for j in jobs]
        result = reinforce_update(checkpoint, episodes, device=device)
        result.update(parent_sha256=parent_sha, batch_receipt=receipt)
        save(dest, result)  # Checkpoint is the atomic commit; resume never applies it twice.
    batch_dir = run / 'batches' / f'{batch_index:04d}'
    dump(batch_dir / 'update.json', dict(checkpoint=str(dest), sha256=sha(dest),
         metrics=result['metrics'], parent_sha256=parent_sha, trajectories=receipt))
    return result


def train(args):
    run = under(args.run, EXP_ROOT)
    with (run / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = checked_config(run)
        torch.set_num_threads(1)
        pool = WorkerPool(run, config)
        try:
            for batch in range(config['batches']):
                path = Path(config['model_dir']) / f'batch_{batch:04d}.pt'
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                validate_checkpoint(checkpoint)
                jobs = make_plan(config, batch, path, sha(path), checkpoint['version'])
                for job in jobs:
                    job['output'] = f"batches/{batch:04d}/{job['id']}"
                batch_dir = directory(run / 'batches' / f'{batch:04d}', '# One frozen policy rollout batch and one REINFORCE update.')
                plan = batch_dir / 'plan.json'
                if plan.exists() and read(plan) != jobs:
                    raise ValueError('frozen batch plan changed')
                dump(plan, jobs)
                dump(run / 'STATUS.json', dict(stage='collecting', batch=batch, total=config['batches'],
                                              **training_position(config, batch)))
                pool.dispatch(jobs, f'batch_{batch:04d}')
                dump(run / 'STATUS.json', dict(stage='updating', batch=batch, total=config['batches'],
                                              **training_position(config, batch)))
                result = commit_update(run, config, batch, jobs)
                position = training_position(config, batch + 1)
                print(json.dumps(dict(batch=batch + 1, **position, **result['metrics'])), flush=True)
                if config.get('epochs') and position['completed_batches_in_epoch'] == 0:
                    epoch_dir = directory(run / 'epochs', '# Completed epoch checkpoint receipts; no duplicated feature data.')
                    dump(epoch_dir / f"epoch_{position['completed_epochs']:03d}.json",
                         dict(**position, checkpoint=str(Path(config['model_dir']) / f'batch_{batch + 1:04d}.pt')))
            final = Path(config['model_dir']) / f"batch_{config['batches']:04d}.pt"
            dump(run / 'TRAINING_COMPLETE.json', dict(run_id=config['run_id'], batches=config['batches'],
                trajectories=training_position(config, config['batches'])['completed_trajectories'], checkpoint=str(final),
                sha256=sha(final), evaluation='not_yet_run', smoke_only=config['smoke_only']))
            dump(run / 'STATUS.json', dict(stage='training_complete', batches=config['batches'],
                                          **training_position(config, config['batches'])))
        except BaseException as error:
            dump(run / 'STATUS.json', dict(stage='stopped', error=repr(error), completed_artifacts_preserved=True))
            raise
        finally:
            pool.close()
            discard_temporary_videos(run / '_temporary')


def evaluate(args):
    from evaluation import evaluate_run
    evaluate_run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='freeze inputs/create random actor; no GPU generation')
    p.add_argument('--run', type=Path, required=True)
    budget = p.add_mutually_exclusive_group(required=True)
    budget.add_argument('--batches', type=int, help='legacy iid prompt sampling budget')
    budget.add_argument('--epochs', type=int, help='full shuffled passes through train prompts, fresh trajectories each pass')
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpus', default='0', help='independent rollout GPUs, e.g. 0,1,2,3; no FSDP')
    p.add_argument('--prompts', type=Path, help='JSON/JSONL with sample_id,prompt,split; default original Ours 800/100/100')
    p.add_argument('--wan-root', type=Path, default=Path('/home/wangyue/work/Wan2.1'))
    p.add_argument('--wan-weights', type=Path, default=MODEL_ROOT / 'Wan2.1-T2V-1.3B')
    p.add_argument('--profile', type=Path, default=DEFAULT_BASELINES / 'component_profile.json')
    p.add_argument('--baseline-source', type=Path, default=DEFAULT_BASELINES)
    p.add_argument('--no-baseline-reuse', action='store_true')
    p.add_argument('--smoke', action='store_true', help='real GPU verification; marked non-production')
    p.set_defaults(func=prepare)
    p = sub.add_parser('train', help='on-policy collection/update, or sealed resume')
    p.add_argument('--run', type=Path, required=True)
    p.set_defaults(func=train)
    p = sub.add_parser('evaluate', help='final actor, held-out prompts, VideoMetrics + VBench custom score')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--split', choices=('val', 'test'), default='test')
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--budgets', type=int, nargs='+', default=[20, 30, 40])
    p.set_defaults(func=evaluate)
    p = sub.add_parser('test', help='CPU algorithm/gradient/resume tests; no Wan inference')
    p.set_defaults(func=lambda _: subprocess.run([sys.executable, str(HERE / 'test_reinforce.py')], check=True))
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
