"""Small immutable records; all experiment outputs stay on the external disk."""
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time

import torch

from core import HERE, PROJECT, OFFICIAL, OURS, FORMAL, PROTOCOL, GENERATION

WORKSPACE = OFFICIAL.parents[1]
EXP_ROOT = Path('/all/yiran06-disk1/wangyue_home/exp').resolve()
MODEL_ROOT = Path('/all/yiran06-disk1/wangyue_home/models').resolve()
DEFAULT_BASELINES = EXP_ROOT / 'cacheimpact_wan21_compact1000_seed42_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temp, path)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(obj, temp)
    os.replace(temp, path)


def directory(path, description):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if not (path / 'README.md').exists():
        (path / 'README.md').write_text(description + '\n')
    return path


def under(path, root):
    p = Path(path).expanduser().resolve()
    if not p.is_relative_to(root) or p == root:
        raise ValueError(f'output must be below {root}')
    return p


def source_hashes():
    files = [*HERE.glob('*.py'), PROJECT / 'main.py', FORMAL / 'features.py', FORMAL / 'model.py',
             *sorted((OURS / 'ours4wan21').glob('*.py')),
             *sorted((OFFICIAL / 'SeaCache4Wan21').glob('*.py')),
             *sorted((OFFICIAL / 'ComponentMetrics').glob('*.py')),
             *sorted((OFFICIAL / 'VideoMetrics/video_metrics').glob('*.py')),
             OFFICIAL / 'Wan21Benchmark/protocol.py',
             OURS / 'data_collection/src/ours4wan21_data/manifest.py']
    return {str(p): sha(p) for p in sorted(set(files))}


def verify_sources(config):
    if config['sources'] != source_hashes():
        raise ValueError('source files changed since prepare; use a new run')


def load_prompts(path=None):
    if path:
        p = Path(path).resolve(strict=True)
        if p.suffix == '.jsonl':
            rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        else:
            obj = read(p)
            rows = obj if isinstance(obj, list) else obj['prompts']
        provenance = {str(p): sha(p)}
    else:
        src = OURS / 'data_collection'
        sys.path.insert(0, str(src / 'src'))
        from ours4wan21_data.manifest import build_seacache_manifest
        pool = src / 'resources/prompts/openvidhd_balanced_5000.upstream.jsonl'
        thresholds = src / 'configs/seacache_thresholds.wan22_v1.json'
        original, _ = build_seacache_manifest(pool, thresholds)
        rows = original[::3]
        provenance = {str(p): sha(p) for p in (pool, thresholds)}
    result = []
    ids, texts = set(), {}
    for row in rows:
        sid, prompt, split = row['sample_id'], row['prompt'], row['split']
        if (not re.fullmatch(r'[A-Za-z0-9_-]+', sid) or sid in ids
                or split not in ('train', 'val', 'test') or not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError('invalid prompt ID/text/split')
        normalized = ' '.join(prompt.split())
        if normalized in texts and texts[normalized] != split:
            raise ValueError('prompt text leaks across splits')
        ids.add(sid); texts[normalized] = split
        result.append(dict(sample_id=sid, prompt=prompt, split=split))
    if not result or not any(r['split'] == 'train' for r in result):
        raise ValueError('empty training prompt pool')
    return result, provenance


def gpu_info(gpus):
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError('GPU list must be nonempty and distinct')
    out = []
    for gpu in gpus:
        line = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=uuid,name', '--format=csv,noheader'], text=True).strip()
        uuid, name = line.split(', ', 1)
        if not uuid.startswith('GPU-') or '\n' in line:
            raise ValueError('expected one physical GPU')
        out.append(dict(device=str(gpu), uuid=uuid, name=name))
    if len({r['uuid'] for r in out}) != len(out):
        raise ValueError('duplicate physical GPU')
    return out


def baseline_references(root, prompts, gpus, profile, weights=None):
    """Read existing compact MP4s ONLY; never touch baseline latents/features."""
    refs = {}
    if root is None:
        return refs
    root = Path(root).resolve(strict=True)
    plan = read(root / 'plan.json')
    generation = {**GENERATION, 'size': [832, 480]}
    if (plan.get('generation') != generation or plan['t5_cpu']
            or plan.get('dit_dtype') != 'bfloat16' or plan['profile_sha'] != sha(profile)):
        raise ValueError('baseline generation/profile differs from current offload protocol')
    if weights is not None and plan['weights'] != weights:
        raise ValueError('baseline Wan model weights differ')
    rows = {r['sample_id']: r for r in plan['prompts']}
    expected_contract = digest(plan)
    gpu_map = {r['uuid']: i for i, r in enumerate(gpus)}
    for row in prompts:
        old = rows.get(row['sample_id'])
        if old is None:
            continue
        if old['prompt'] != row['prompt'] or old['split'] != row['split']:
            raise ValueError('baseline prompt identity mismatch')
        worker = root / f"worker_{old['worker']}"
        ident = read(worker / 'identity.json')
        if ident['contract'] != expected_contract:
            raise ValueError('baseline worker contract mismatch')
        if ident['uuid'] not in gpu_map:
            continue  # Unavailable baseline GPU: generate ephemeral native reference instead.
        cell = worker / 'cells' / (row['sample_id'] + '__skip_00')
        receipt = read(cell / 'COMPLETE.json')
        if receipt['contract'] != expected_contract or sha(cell / 'video.mp4') != receipt['files']['video.mp4']:
            raise ValueError('baseline video receipt mismatch')
        refs[row['sample_id']] = dict(path=str(cell / 'video.mp4'), sha256=receipt['files']['video.mp4'],
            worker=gpu_map[ident['uuid']], gpu_uuid=ident['uuid'], source='compact1000_native_baseline')
    return refs


def make_plan(config, batch_index, policy_path, policy_sha, version):
    # Separate deterministic planning RNG from action sampling and Wan noise RNG.
    rng = random.Random(f"reinforce:{config['training']['seed']}:{batch_index}")
    pool = [r for r in config['prompts'] if r['split'] == 'train']
    epoch_rows = None
    if config.get('epochs') is not None:
        per_epoch = (len(pool) + config['batch_size'] - 1) // config['batch_size']
        epoch, within = divmod(batch_index, per_epoch)
        if not 0 <= epoch < config['epochs']:
            raise ValueError('batch outside epoch schedule')
        random.Random(f"reinforce-epoch:{config['training']['seed']}:{epoch}").shuffle(pool)
        start = within * config['batch_size']
        epoch_rows = pool[start:start + config['batch_size']]
    jobs = []
    for i in range(len(epoch_rows) if epoch_rows is not None else config['batch_size']):
        row = epoch_rows[i] if epoch_rows is not None else rng.choice(pool)
        ref = config['baselines'].get(row['sample_id'])
        worker = ref['worker'] if ref else int(hashlib.sha256(row['sample_id'].encode()).hexdigest(), 16) % len(config['gpus'])
        jobs.append(dict(**row, id=f'b{batch_index:04d}_t{i:04d}', k=rng.randint(20, 40),
                         worker=worker, baseline=ref, sampling_seed=rng.randrange(2**63),
                         checkpoint=str(policy_path), checkpoint_sha256=policy_sha,
                         policy_version=version, batch_index=batch_index, evaluation=False))
    return jobs


def training_position(config, completed_batches):
    """Progress counts committed updates; partial last batches are retained."""
    if config.get('epochs') is None:
        return dict(completed_batches=completed_batches,
                    completed_trajectories=completed_batches * config['batch_size'])
    n = sum(r['split'] == 'train' for r in config['prompts'])
    per_epoch = (n + config['batch_size'] - 1) // config['batch_size']
    epochs, within = divmod(completed_batches, per_epoch)
    return dict(completed_batches=completed_batches, completed_epochs=epochs,
                completed_batches_in_epoch=within, batches_per_epoch=per_epoch,
                total_epochs=config['epochs'],
                completed_trajectories=epochs * n + min(within * config['batch_size'], n),
                total_training_trajectories=config['epochs'] * n)


def identity(config, job):
    return dict(run_id=config['run_id'], job=job)


def complete(path, expected):
    path = Path(path)
    marker = path / 'COMPLETE.json'
    if not marker.exists():
        return False
    obj = read(marker)
    if obj['identity'] != expected:
        raise ValueError('artifact identity differs')
    for name, value in obj['files'].items():
        if sha(path / name) != value:
            raise ValueError(f'corrupt completed artifact {path / name}')
    return True


def seal(path, expected, names):
    path = Path(path)
    dump(path / 'COMPLETE.json', dict(identity=expected,
        files={name: sha(path / name) for name in names}, status='complete'))


def space(path, workers=1):
    required = (30 + workers) * 2**30
    if shutil.disk_usage(path).free < required:
        raise RuntimeError(f'paused_low_disk: requires {(30 + workers)} GiB free; completed artifacts preserved')


def discard_temporary_videos(root):
    # Only this run's documented ephemeral video files, never source archives.
    root = Path(root)
    for path in root.rglob('*.mp4'):
        path.unlink()
