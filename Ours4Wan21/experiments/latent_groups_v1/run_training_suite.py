#!/usr/bin/env python3
"""Build twelve caches, run four deterministic GPU queues, and analyze every 400-epoch run."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ours4wan21.contracts import EXP_ROOT, MODEL_ROOT, create_result, dump, sha256, under
from ours4wan21.feature_cache import load_feature_index

MODES = (
    'scalar5', 'sea7', 'sea7_dynamics_raw_sea128', 'sea7_cache_update192',
    'sea7_local_drift1024', 'sea7_spatial_gradient96', 'sea7_channel_geometry240',
    'sea7_distribution256', 'sea7_spectral_drift512', 'sea7_spectral_phase512',
    'sea7_spectral_shape576', 'sea7_spectral_dynamics1024',
)


def cache_complete(path: Path) -> bool:
    try:
        marker = json.loads((path / 'COMPLETE.json').read_text())
        return marker.get('status') == 'complete' and sha256(path / 'transitions.pt') == marker['sha256']
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def training_complete(result: Path, weights: Path) -> bool:
    try:
        marker = json.loads((result / 'TRAINING_COMPLETE.json').read_text())
        rows = [line for line in (result / 'epoch_metrics.jsonl').read_text().splitlines() if line.strip()]
        checkpoints = list((weights / 'checkpoints').glob('epoch_*.pt'))
        return (marker.get('status') == 'complete' and marker.get('epochs') == 400 and
                not marker.get('smoke_only') and len(rows) == 400 and len(checkpoints) == 400 and
                (result / 'model_weights').resolve() == weights.resolve())
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def analysis_complete(path: Path) -> bool:
    try:
        marker = json.loads((path / 'COMPLETE.json').read_text())
        selection = json.loads((path / 'checkpoint_selection.json').read_text())
        return (marker.get('status') == 'complete' and selection.get('status') == 'selected' and
                (path / 'selected_model.pt').is_file())
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-root', type=Path, required=True)
    parser.add_argument('--selection-dir', type=Path, required=True)
    parser.add_argument('--suite-name', default='ours21_random3000_12groups_v1')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        parser.error('--gpus must be distinct nonnegative physical GPU indices')
    if 'wan2.2' not in Path(sys.prefix).name.lower():
        raise ValueError('use the wan2.2 environment')

    collection = args.collection_root.resolve(strict=True)
    selection = (args.selection_dir / 'selection.json').resolve(strict=True)
    features = EXP_ROOT / f'{args.suite_name}_features'
    feature_index = load_feature_index(features)
    if feature_index['selection_sha256'] != sha256(selection) or len(feature_index['rows']) != 3000:
        raise ValueError('completed feature cache does not match the frozen 3000 selection')

    control_path = EXP_ROOT / f'{args.suite_name}_orchestration'
    config = dict(schema='ours21_12group_orchestration_v1', suite_name=args.suite_name,
                  modes=list(MODES), gpus=args.gpus, collection_root=str(collection),
                  selection=str(selection), selection_sha256=sha256(selection),
                  feature_index_sha256=sha256(features / 'index.json'), python=sys.executable,
                  training_epochs=400, started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    if args.resume:
        control = under(control_path, EXP_ROOT)
        existing = json.loads((control / 'config.json').read_text())
        if {key: existing[key] for key in config if key != 'started_at'} != {key: config[key] for key in config if key != 'started_at'}:
            raise ValueError('orchestration resume configuration changed')
        config = existing
    else:
        control = create_result(control_path, '# Twelve-group training orchestration\n\nconfig.json freezes the shared selection, feature cache and GPU queues. logs/ holds per-stage process output; status.json and COMPLETE.json summarize execution. Training data/results and model checkpoints remain in their own external result/model directories.')
        (control / 'logs').mkdir()
        dump(control / 'config.json', config)

    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    state_lock = threading.Lock()
    state = {'schema': 'ours21_12group_orchestration_status_v1', 'suite_name': args.suite_name,
             'status': 'running', 'modes': {mode: {} for mode in MODES}}

    def update(mode: str, **values) -> None:
        with state_lock:
            state['modes'][mode].update(values)
            dump(control / 'status.json', state)

    def execute(mode: str, stage: str, command: list[str], gpu: int | None = None) -> None:
        log_path = control / 'logs' / f'{mode}_{stage}.log'
        child_env = dict(env)
        if gpu is not None:
            child_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        update(mode, stage=stage, status='running', gpu=gpu, log=str(log_path), command=command)
        with log_path.open('ab') as log:
            completed = subprocess.run(command, cwd=PROJECT, env=child_env, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            update(mode, stage=stage, status='failed', returncode=completed.returncode)
            raise RuntimeError(f'{mode} {stage} failed; inspect {log_path}')
        update(mode, stage=stage, status='complete', returncode=0)

    for mode in MODES:
        cache = EXP_ROOT / f'{args.suite_name}_{mode}_cache'
        if cache.exists():
            if not cache_complete(cache):
                raise RuntimeError(f'incomplete or corrupt cache requires a fresh suite name: {cache}')
            update(mode, cache='complete')
            continue
        command = [sys.executable, 'prepare_data.py', '--collection-root', str(collection),
                   '--selection', str(selection), '--state-mode', mode, '--output-dir', str(cache)]
        if mode not in ('scalar5', 'sea7'):
            command.extend(('--feature-cache', str(features)))
        execute(mode, 'cache', command)
        if not cache_complete(cache):
            raise RuntimeError(f'{mode} cache did not pass completion/hash validation')
        update(mode, cache='complete')

    assignments = {gpu: list(MODES[index::len(args.gpus)]) for index, gpu in enumerate(args.gpus)}
    dump(control / 'gpu_assignments.json', assignments)

    def gpu_queue(gpu: int, modes: list[str]) -> None:
        for mode in modes:
            cache = EXP_ROOT / f'{args.suite_name}_{mode}_cache'
            training = EXP_ROOT / f'{args.suite_name}_{mode}_train'
            weights = MODEL_ROOT / f'{args.suite_name}_{mode}'
            analysis = EXP_ROOT / f'{args.suite_name}_{mode}_analysis'
            if training.exists() or weights.exists():
                if not training_complete(training, weights):
                    raise RuntimeError(f'incomplete training cannot resume; use a fresh suite name: {mode}')
                update(mode, training='complete', gpu=gpu)
            else:
                execute(mode, 'train', [sys.executable, 'train.py', '--dataset', str(cache),
                    '--state-mode', mode, '--output-dir', str(training), '--checkpoint-dir', str(weights),
                    '--device', 'cuda'], gpu)
                if not training_complete(training, weights):
                    raise RuntimeError(f'{mode} training failed completion checks')
                update(mode, training='complete', gpu=gpu)
            if analysis.exists():
                if not analysis_complete(analysis):
                    raise RuntimeError(f'incomplete analysis requires a fresh suite name: {mode}')
                update(mode, analysis='complete', gpu=gpu)
            else:
                execute(mode, 'analyze', [sys.executable, 'analyze_training.py',
                    '--training-result', str(training), '--dataset', str(cache),
                    '--checkpoint-dir', str(weights), '--output-dir', str(analysis),
                    '--device', 'cuda'], gpu)
                if not analysis_complete(analysis):
                    raise RuntimeError(f'{mode} analysis failed completion checks')
                update(mode, analysis='complete', gpu=gpu)

    failures = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = {pool.submit(gpu_queue, gpu, modes): gpu for gpu, modes in assignments.items()}
        for future, gpu in [(future, gpu) for future, gpu in futures.items()]:
            try:
                future.result()
            except BaseException as error:
                failures.append({'gpu': gpu, 'error': repr(error)})
    if failures:
        state['status'] = 'failed'
        state['failures'] = failures
        dump(control / 'status.json', state)
        raise RuntimeError(f'one or more GPU queues failed: {failures}')
    state['status'] = 'complete'
    dump(control / 'status.json', state)
    dump(control / 'COMPLETE.json', dict(status='complete', modes=len(MODES), epochs_per_mode=400,
        selection_sha256=sha256(selection), feature_index_sha256=sha256(features / 'index.json')))


if __name__ == '__main__':
    main()
