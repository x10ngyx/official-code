"""Frozen online protocol, prompt isolation, plans and resumable artifact contracts."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

from .contracts import EXP_ROOT, MODEL_ROOT, PROTOCOL, dump, sha256, under

THREAD_KEYS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


@dataclass(frozen=True)
class OnlineConfig:
    rounds: int = 8
    trajectories_per_round: int = 100
    prompt_pool_size: int = 3000
    batch_size: int = 256
    critic_warmup_epochs: int = 5
    joint_epochs: int = 20
    selection_start: int = 11
    selection_end: int = 20
    evaluation_every: int = 4
    evaluation_prompts: int = 20
    evaluation_targets: tuple = (1.8, 2.4, 3.0)
    vbench_enabled: bool = False
    target_min: float = 1.5
    target_max: float = 3.5
    actor_lr: float = 4e-5
    critic_lr: float = 1e-4
    weight_decay: float = .01
    tau: float = .7
    beta: float = 1.
    weight_max: float = 20.
    gamma: float = 1.
    target_rho: float = .999
    grad_clip_norm: float = 1.
    seed: int = 42
    plan_seed: int = 20260813
    fixed_state_limit: int = 4096

    def payload(self):
        return json.loads(json.dumps(asdict(self)))

    def evaluation_rounds(self):
        return sorted(set(range(self.evaluation_every, self.rounds+1, self.evaluation_every)) | {self.rounds})


ONLINE_IQL_PROFILES = {
    'default': {},
    'aggressive_a2_a3_v1': dict(tau=.85, beta=2.5, weight_max=75.),
}


def online_config(*, rounds=8, prompt_pool_size=3000, profile='default'):
    """Versioned online IQL choices; midpoint of the completed A2/A3 experiment."""
    if profile not in ONLINE_IQL_PROFILES:
        raise ValueError('unknown online IQL profile: '+str(profile))
    return OnlineConfig(rounds=rounds, prompt_pool_size=prompt_pool_size,
                        **ONLINE_IQL_PROFILES[profile])


def run_config(manifest):
    """Validate round count, training population and the named online IQL profile."""
    rounds=manifest['config']['rounds']
    if type(rounds) is not int or rounds < 1:
        raise ValueError('rounds must be a positive integer')
    size=OnlineConfig().prompt_pool_size
    if manifest.get('paths',{}).get('training_bundle'):
        size=len(manifest['pool'])
        if size<1:raise ValueError('empty offline training population')
    config=online_config(rounds=rounds,prompt_pool_size=size,profile=manifest.get('iql_profile','default'))
    if manifest['config'] != config.payload():
        raise ValueError('frozen online algorithm configuration changed')
    return config


def read(path):
    return json.loads(Path(path).read_text())


def require_environment():
    import sys
    explicit = os.environ.get('WAN22_PYTHON')
    if Path(sys.prefix).name != 'wan2.2' and not (
            explicit and Path(explicit).resolve() == Path(sys.executable).resolve()):
        raise ValueError('use conda environment wan2.2')
    if any(os.environ.get(k) != '1' for k in THREAD_KEYS):
        raise ValueError('all four BLAS/OpenMP thread variables must explicitly equal 1')


def text_key(text):
    return ' '.join(text.split()).casefold()


def prompts(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sid = r.get('sample_id', r.get('id'))
        prompt = r.get('prompt', r.get('prompt_en', r.get('text')))
        if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', sid):
            raise ValueError('prompt IDs must be safe ASCII filenames, at most 120 chars')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('empty prompt text')
        rows.append(dict(sample_id=sid, prompt=prompt.strip()))
    if (not rows or len({r['sample_id'] for r in rows}) != len(rows)
            or len({text_key(r['prompt']) for r in rows}) != len(rows)):
        raise ValueError('prompt IDs and normalized texts must be unique')
    return rows


def isolate_prompts(pool, evaluation, registry, prompt_splits, expected_evaluation=20):
    """Registry is required to detect renamed held-out prompts, not only IDs."""
    if len(evaluation) != expected_evaluation:
        raise ValueError(f'exactly {expected_evaluation} evaluation prompts required')
    by_id = {r['sample_id']: r['prompt'] for r in registry}
    if not set(prompt_splits).issubset(by_id):
        raise ValueError('registry must cover every offline train/validation/test prompt ID')
    offline_texts = {text_key(by_id[sid]) for sid in prompt_splits}
    forbidden_ids = {sid for sid, split in prompt_splits.items() if split != 'train'}
    forbidden_texts = {text_key(by_id[sid]) for sid in forbidden_ids}
    eval_ids = {r['sample_id'] for r in evaluation}
    eval_texts = {text_key(r['prompt']) for r in evaluation}
    if eval_ids & set(prompt_splits) or eval_texts & offline_texts:
        raise ValueError('evaluation20 prompts must be disjoint from the offline dataset')
    for r in pool:
        sid, key = r['sample_id'], text_key(r['prompt'])
        if sid in by_id and key != text_key(by_id[sid]):
            raise ValueError('prompt ID has changed text')
        if sid in forbidden_ids | eval_ids or key in forbidden_texts | eval_texts:
            raise ValueError('online pool leaks validation/test/evaluation20 prompts')


def gpu_slot(sid, count):
    return int(hashlib.sha256(sid.encode()).hexdigest()[:16], 16) % count


def make_plan(pool, round_index, gpu_count, budget_for_target, config=OnlineConfig(), *, gpu_slots=None):
    if round_index < 1 or not pool or gpu_count < 1:
        raise ValueError('invalid round/pool/GPU count')
    # Independent deterministic streams; prompts sampled WITH replacement each round.
    pr = random.Random(config.plan_seed + 2 * round_index)
    tr = random.Random(config.plan_seed + 2 * round_index + 1)
    n = config.trajectories_per_round
    targets = [config.target_min + (i + tr.random()) * (config.target_max-config.target_min)/n
               for i in range(n)]
    tr.shuffle(targets)
    return [dict(**(p := pr.choice(pool)), trajectory_id=f'r{round_index:03d}_{i:04d}',
                 target_speedup=t, skip_budget=budget_for_target(t),
                 sampling_seed=config.plan_seed + round_index * 100000 + i,
                 slot=(gpu_slots[p['sample_id']] if gpu_slots is not None else
                       gpu_slot(p['sample_id'], gpu_count))) for i, t in enumerate(targets)]


def balanced_training_slots(pool, gpu_count, config):
    """Freeze one GPU per sampled prompt, balancing each round's new baseline work."""
    assigned={}
    for r in range(1,config.rounds+1):
        rows=make_plan(pool,r,gpu_count,lambda target:0,config)
        counts={}
        for row in rows:counts[row['sample_id']]=counts.get(row['sample_id'],0)+1
        workload=[0.]*gpu_count
        for sid,n in counts.items():
            if sid in assigned:workload[assigned[sid]]+=n
        for sid,n in counts.items():
            if sid in assigned:continue
            slot=min(range(gpu_count),key=lambda g:(workload[g],g))
            assigned[sid]=slot
            workload[slot]+=2.25+n  # one native baseline plus sampled candidates
    return assigned


def freeze_json(path, payload):
    path = Path(path)
    if path.exists():
        if read(path) != payload:
            raise ValueError(f'frozen input changed: {path}')
    else:
        dump(path, payload)


def seal(folder, files, *, identity):
    folder = Path(folder)
    for f in files:
        if not (folder / f).is_file():
            raise ValueError(f'missing artifact: {folder / f}')
    dump(folder / 'COMPLETE.json', dict(identity=identity,
         files={str(f): sha256(folder / f) for f in files}))


def verified(folder, identity=None):
    folder = Path(folder)
    marker = folder / 'COMPLETE.json'
    if not marker.exists():
        return False
    p = read(marker)
    if identity is not None and p['identity'] != identity:
        raise ValueError(f'completed artifact identity mismatch: {folder}')
    if not p.get('files'):
        raise ValueError('empty completion manifest')
    for f, digest in p['files'].items():
        path = folder / f
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'completed artifact corrupt: {path}')
    return True


def prepare_directory(folder, identity):
    """Reuse sealed artifacts; retain incomplete work in place under an archive."""
    folder = Path(folder)
    if verified(folder, identity):
        return False
    if folder.exists():
        archived = folder.parent / 'incomplete'
        archived.mkdir(exist_ok=True)
        folder.rename(archived / f'{folder.name}_{time.time_ns()}')
    folder.mkdir(parents=True)
    (folder / 'README.md').write_text('Generated online pipeline artifact. COMPLETE.json seals identity and source hashes.\n')
    dump(folder / 'identity.json', identity)
    return True


def atomic_torch_save(payload, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_identity(path):
    path = under(path, MODEL_ROOT)
    return dict(path=str(path), sha256=sha256(path))


def inventory(root):
    """Cheap resume guard for resident model files; no duplicate multi-GB weight hashing."""
    root=Path(root)
    return {str(p.relative_to(root)):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(root.rglob('*')) if p.is_file()}
