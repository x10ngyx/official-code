"""Versioned state and fixed local training contracts shared by train/inference."""
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import hashlib
import json
import os

import torch

PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL = PROJECT.parent
WORKSPACE = Path(os.environ.get('OURS4WAN21_WORKSPACE', str(PROJECT.parents[2]))).expanduser().resolve()
EXP_ROOT = Path(os.environ.get('OURS4WAN21_EXP_BASE', os.environ.get('EXP_BASE',
                '/mnt/hdd/xiongyuxiang/tmp/exp'))).expanduser().resolve()
MODEL_ROOT = WORKSPACE / 'models'
STEPS = 50
FORCED = (0, 49)
SCHEMA = 'ours4wan21_exact_k_state_v1'
from .latent_features import GROUPS as LEGACY_GROUPS, contract as legacy_latent_contract
from .bloc_features import GROUPS as BLOC_GROUPS, contract as bloc_contract
GROUPS = {**LEGACY_GROUPS, **BLOC_GROUPS}


def latent_contract(group):
    return bloc_contract(group) if group in BLOC_GROUPS else legacy_latent_contract(group)


FEATURE_MODES = {f'sea7_{g}': g for g in LEGACY_GROUPS}
BLOC_MODES = {f'sea7_{g}': g for g in BLOC_GROUPS}
MODES = ('scalar5', 'sea7', 'sea7_latent', *FEATURE_MODES, *BLOC_MODES)

def latent_group(mode):
    return FEATURE_MODES.get(mode, BLOC_MODES.get(mode))
FIVE = ('cached_valid', 'step_fraction', 'skip_budget_fraction',
        'used_skips_fraction', 'consecutive_skips_fraction')
SEVEN = ('sea_adjacent_relative_l1', 'sea_accumulated_with_current', *FIVE)
PROTOCOL = dict(model='Wan2.1-T2V-1.3B', width=832, height=480, frames=81,
                fps=16, steps=50, solver='unipc', shift=5, cfg=5, seed=42,
                dit_compute_dtype='bfloat16', offload_model=False, t5_cpu=False,
                batch_size=1)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 400
    batch_size: int = 256
    num_workers: int = 0
    seed: int = 42
    lr: float = 1e-4
    weight_decay: float = .01
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.
    tau: float = .6
    beta: float = 1.
    gamma: float = 1.
    target_rho: float = .999
    weight_max: float = 20.
    grad_clip_norm: float = 1.
    reward_scale: float = 1.
    checkpoint_every: int = 1
    log_every: int = 0


IQL_PROFILE_PARAMETERS = {
    'aggressive_a1_v1': dict(tau=.70, beta=1.5, weight_max=30.),
    'aggressive_a2_v1': dict(tau=.80, beta=2., weight_max=50.),
    'aggressive_v1': dict(tau=.90, beta=3., weight_max=100.),
    'aggressive_a4_v1': dict(tau=.95, beta=5., weight_max=200.),
}
IQL_PROFILES = ('baseline', *IQL_PROFILE_PARAMETERS)


def training_config(profile='baseline', *, seed=42):
    """Explicit, versioned IQL experiment; the original defaults stay unchanged."""
    if profile not in IQL_PROFILES:
        raise ValueError(f'unknown IQL profile: {profile}')
    if type(seed) is not int or seed < 0:
        raise ValueError('training seed must be a nonnegative integer')
    config = replace(TrainingConfig(), seed=seed)
    if profile in IQL_PROFILE_PARAMETERS:
        config = replace(config, **IQL_PROFILE_PARAMETERS[profile])
    return config


def state_names(mode):
    if mode == 'sea7_latent':
        raise NotImplementedError('sea7_latent is reserved: latent feature is not confirmed; no zero padding or fallback is allowed')
    if mode not in MODES:
        raise ValueError(f'unknown state mode: {mode}')
    group = latent_group(mode)
    return tuple(f'{group}_{i}' for i in range(GROUPS[group][0])) + SEVEN if group else (FIVE if mode == 'scalar5' else SEVEN)


def observation(mode, *, step, budget, used, consecutive, cached_valid,
                adjacent=0., accumulated=0., latent=None):
    state_names(mode)
    if any(type(v) is not int for v in (step, budget, used, consecutive)):
        raise ValueError('state counters must be integers')
    if not (0 <= step < STEPS and 0 <= budget <= 48 and
            0 <= used <= min(step, budget) and 0 <= consecutive <= used):
        raise ValueError('invalid state counters')
    if type(cached_valid) is not bool or cached_valid != (step > 0):
        raise ValueError('Wan21 cache must be invalid only before step zero')
    if step in FORCED:
        adjacent = accumulated = 0.
    if not all(math.isfinite(v) and v >= 0 for v in (adjacent, accumulated)):
        raise ValueError('SEA distances must be finite and nonnegative')
    five = [float(cached_valid), step / 49., budget / 50., used / 50., consecutive / 50.]
    scalar = torch.tensor(five if mode == 'scalar5' else [adjacent, accumulated, *five], dtype=torch.float32)
    group = latent_group(mode)
    if group:
        if latent is None or latent.shape != (GROUPS[group][0],) or not torch.isfinite(latent).all():
            raise ValueError('selected feature mode requires its finite latent vector')
        return torch.cat((latent.detach().cpu().float(), scalar))
    if latent is not None:
        raise ValueError('scalar controls must not receive latent features')
    return scalar


def state_contract(mode):
    return dict(schema=SCHEMA, mode=mode, names=list(state_names(mode)),
                num_steps=STEPS, forced_steps=list(FORCED),
                latent_feature=latent_contract(latent_group(mode)) if latent_group(mode) else None, budget_divisor=50,
                action_unit='denoising_step_shared_cond_uncond',
                distance='SeaCache4Wan21 filtered adjacent relative L1; pre-action accumulated distance; boundaries zero')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def under(path, root):
    path, root = Path(path).expanduser().resolve(), Path(root).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f'path must be below {root}: {path}')
    return path


def create_result(path, description):
    path = under(path, EXP_ROOT)
    link = PROJECT / 'experiment_results' / path.name
    if path.exists() or link.exists() or link.is_symlink():
        raise FileExistsError(f'use a fresh result name: {path}')
    path.mkdir(parents=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(path, target_is_directory=True)
    (path / 'README.md').write_text(description + '\n')
    return path


def create_nested_result(path, parent, description):
    """Create a small child result below one registered external result root.

    Large suites may contain dozens of generation conditions.  Registering the
    suite root once keeps ``experiment_results/`` useful while preserving the
    external-results and no-overwrite contracts for every child.
    """
    parent = under(parent, EXP_ROOT)
    path = under(path, parent)
    link = PROJECT / 'experiment_results' / parent.name
    if (not parent.is_dir() or not link.is_symlink() or
            link.resolve() != parent.resolve()):
        raise ValueError('nested result parent must be a registered experiment result')
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'use a fresh nested result path: {path}')
    path.mkdir(parents=True)
    (path / 'README.md').write_text(description + '\n')
    return path
