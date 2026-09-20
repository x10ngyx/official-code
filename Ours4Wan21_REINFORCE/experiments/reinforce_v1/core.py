"""REINFORCE with batch-frozen observation statistics and one update per batch."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OFFICIAL = PROJECT.parent
OURS = OFFICIAL / 'Ours4Wan21'
FORMAL = OURS / 'experiments/cnn_mixed3500_v1'
for path in (OURS, FORMAL):
    sys.path.insert(0, str(path))
from model import CNN, ARCH, dimension, encode_normalized
from features import CONTRACT
from ours4wan21.local_iql import required_hard_budget_action

DIM = dimension('G1')
SCHEMA = 'ours21_reinforce_g1_v1'
PROTOCOL = dict(model='Wan2.1-T2V-1.3B', width=832, height=480, frames=81,
                fps=16, steps=50, solver='unipc', shift=5, cfg=5, seed=42,
                dit_compute_dtype='bfloat16', offload_model=True, t5_cpu=False,
                batch_size=1)
GENERATION = dict(size=(832, 480), frame_num=81, shift=5, sample_solver='unipc',
                  sampling_steps=50, guide_scale=5, seed=42, offload_model=True)


@dataclass(frozen=True)
class Config:
    seed: int = 42
    lr: float = 1e-4
    weight_decay: float = .01
    grad_clip: float = 1.
    reward_ema_decay: float = .9
    microbatch_steps: int = 16
    # GPU rollout vs CPU update, including changed convolution batch size.
    replay_atol: float = 3e-4

    def validate(self):
        if (self.seed < 0 or self.lr <= 0 or self.weight_decay < 0
                or self.grad_clip <= 0 or not 0 <= self.reward_ema_decay < 1
                or self.microbatch_steps < 1 or self.replay_atol <= 0
                or not all(math.isfinite(x) for x in (self.lr, self.weight_decay,
                    self.grad_clip, self.reward_ema_decay, self.replay_atol))):
            raise ValueError('invalid REINFORCE configuration')


class RunningMoments:
    """Population moments in float64; identity until the first complete batch."""
    def __init__(self, dim=DIM):
        self.count = 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.m2 = torch.zeros(dim, dtype=torch.float64)

    def update(self, rows):
        x = rows.detach().cpu().double()
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not len(x) or not torch.isfinite(x).all():
            raise ValueError('invalid raw observation matrix')
        n = len(x)
        mean = x.mean(0)
        m2 = ((x - mean) ** 2).sum(0)
        delta = mean - self.mean
        total = self.count + n
        self.m2 += m2 + delta.square() * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total

    def normalizer(self):
        if not self.count:
            return dict(mean=self.mean.float().clone(), std=torch.ones_like(self.mean).float())
        return dict(mean=self.mean.float().clone(),
                    std=(self.m2 / self.count).clamp_min(0).sqrt().clamp_min(1e-6).float())

    def state_dict(self):
        return dict(count=self.count, mean=self.mean.clone(), m2=self.m2.clone())

    @classmethod
    def load(cls, state):
        out = cls(len(state['mean']))
        out.count = state['count']
        out.mean, out.m2 = state['mean'].clone().double(), state['m2'].clone().double()
        if (type(out.count) is not int or out.count < 0 or out.mean.ndim != 1
                or out.m2.shape != out.mean.shape or not torch.isfinite(out.mean).all()
                or not torch.isfinite(out.m2).all() or (out.m2 < 0).any()):
            raise ValueError('invalid moments checkpoint')
        return out


def tensor_hash(tensor):
    x = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((x.dtype, tuple(x.shape))).encode() + x.view(torch.uint8).numpy().tobytes()).hexdigest()


def policy_version(actor_state, normalizer, batch_index):
    h = hashlib.sha256(str(batch_index).encode())
    for k, v in sorted({**actor_state, **{'normalizer.' + k: v for k, v in normalizer.items()}}.items()):
        h.update(k.encode()); h.update(tensor_hash(v).encode())
    return h.hexdigest()


def new_checkpoint(config: Config, run_id: str, smoke_only=False):
    config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        actor = CNN('G1', 2)
    opt = torch.optim.AdamW(actor.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    moments = RunningMoments()
    return pack(actor, opt, moments, {}, config, 0, run_id, smoke_only)


def pack(actor, optimizer, moments, reward_baselines, config, batch_index, run_id, smoke_only):
    state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    norm = moments.normalizer()
    return dict(schema=SCHEMA, architecture=ARCH, feature_contract=CONTRACT,
                protocol=PROTOCOL, group='G1', config=asdict(config),
                batch_index=batch_index, run_id=run_id, smoke_only=smoke_only,
                policy_net=state, normalizer=norm, moments=moments.state_dict(),
                reward_baselines=dict(reward_baselines), optimizer=optimizer.state_dict(),
                version=policy_version(state, norm, batch_index),
                normalization='batch_frozen_cumulative_population_fp16_then_fp32',
                reward='VideoMetrics mean per-frame decoded RGB PSNR; terminal only')


def validate_checkpoint(p):
    if (p['schema'] != SCHEMA or p['architecture'] != ARCH
            or p['feature_contract'] != CONTRACT or p['protocol'] != PROTOCOL
            or p['group'] != 'G1' or type(p['batch_index']) is not int or p['batch_index'] < 0):
        raise ValueError('not a matching REINFORCE CNN+G1 checkpoint')
    Config(**p['config']).validate()
    if p['version'] != policy_version(p['policy_net'], p['normalizer'], p['batch_index']):
        raise ValueError('policy/normalizer version mismatch')
    if any(not torch.isfinite(v).all() for v in p['policy_net'].values()):
        raise ValueError('nonfinite actor')
    norm = RunningMoments.load(p['moments']).normalizer()
    if set(p['normalizer']) != set(norm) or any(
            norm[k].shape != (DIM,) or not torch.equal(norm[k], p['normalizer'][k]) for k in norm):
        raise ValueError('normalizer is not the checkpoint cumulative moments')
    if any(not 20 <= int(k) <= 40 or not math.isfinite(v) for k, v in p['reward_baselines'].items()):
        raise ValueError('invalid historical reward baseline')


def validate_episode(ep, checkpoint):
    if (ep['policy_version'] != checkpoint['version'] or ep['run_id'] != checkpoint['run_id']
            or ep['batch_index'] != checkpoint['batch_index'] or ep['split'] != 'train'
            or ep.get('action_mode') != 'policy_categorical'):
        raise ValueError('stale, evaluation, or foreign trajectory cannot enter on-policy update')
    k = ep['k']
    if type(k) is not int or not 20 <= k <= 40:
        raise ValueError('training K outside 20..40')
    raw, x = ep['raw'], ep['inputs']
    if (raw.shape != (50, DIM) or raw.dtype != torch.float32
            or x.shape != raw.shape or x.dtype != torch.float16
            or not torch.isfinite(raw).all() or not torch.isfinite(x).all()):
        raise ValueError('invalid observation encoding')
    if not torch.equal(x, encode_normalized(raw, checkpoint['normalizer'])):
        raise ValueError('rollout normalized inputs differ from batch-frozen statistics')
    if (raw[0, :-7].count_nonzero() or raw[[0, 49], -7:-5].count_nonzero()
            or (raw[:, -7:-5] < 0).any()):
        raise ValueError('invalid first-step features/SEA sentinel')
    actions, free, logp = ep['actions'], ep['free'], ep['logp']
    if (actions.shape != (50,) or actions.dtype != torch.int64
            or free.shape != (50,) or free.dtype != torch.bool
            or logp.shape != (50,) or not torch.isfinite(logp).all()
            or not math.isfinite(ep['reward']) or not 0 <= ep['reward'] <= 100):
        raise ValueError('invalid trajectory actions/reward')
    used = consecutive = 0
    for t, a in enumerate(actions.tolist()):
        required, _ = required_hard_budget_action(step_index=t, used_skips=used,
                    skip_budget=k, num_steps=50)
        if a not in (0, 1) or bool(free[t]) != (required is None) or (required is not None and a != required):
            raise ValueError('invalid Exact-K action/mask')
        expected = torch.tensor([float(t > 0), t / 49, k / 50, used / 50, consecutive / 50])
        if not torch.equal(raw[t, -5:], expected):
            raise ValueError('invalid pre-action budget state')
        if not free[t] and logp[t] != 0:
            raise ValueError('forced action must have log-prob zero')
        if free[t] and logp[t] > 0:
            raise ValueError('positive log-prob')
        used += a
        consecutive = consecutive + 1 if a else 0
    if used != k:
        raise ValueError('not exactly K reuses')


def reinforce_update(checkpoint, episodes, device='cpu'):
    """One optimizer step. All validation/replay precedes any parameter update."""
    validate_checkpoint(checkpoint)
    if not episodes or len({e['trajectory_id'] for e in episodes}) != len(episodes):
        raise ValueError('empty or duplicated batch')
    for ep in episodes:
        validate_episode(ep, checkpoint)
    cfg = Config(**checkpoint['config'])
    actor = CNN('G1', 2).to(device).eval()
    actor.load_state_dict(checkpoint['policy_net'])
    opt = torch.optim.AdamW(actor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # On CPU load_state_dict can retain tensor aliases into its argument.
    # Updating Adam moments must never mutate the parent checkpoint in memory.
    opt.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    opt.zero_grad(set_to_none=True)
    baseline = checkpoint['reward_baselines']
    loss_value = entropy_sum = max_error = 0.
    free_steps = 0
    advantages = []
    for ep in episodes:
        adv = ep['reward'] - baseline.get(str(ep['k']), 0.)
        advantages.append(adv)
        idx = ep['free'].nonzero().flatten()
        for chunk in idx.split(cfg.microbatch_steps):
            x = ep['inputs'][chunk].float().to(device)
            a = ep['actions'][chunk].to(device)
            logits = actor(x)
            lp_all = F.log_softmax(logits, -1)
            lp = lp_all.gather(1, a[:, None]).squeeze(1)
            err = float((lp.detach().cpu() - ep['logp'][chunk]).abs().max())
            max_error = max(max_error, err)
            if not torch.isfinite(logits).all() or err > cfg.replay_atol:
                raise ValueError(f'behavior log-prob replay mismatch: {err}')
            # Sum over decisions, average over trajectories, NOT over free steps.
            loss = -adv * lp.sum() / len(episodes)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite policy loss')
            loss.backward()
            loss_value += float(loss.detach())
            entropy_sum += float(-(lp_all.detach().exp() * lp_all.detach()).sum())
            free_steps += len(chunk)
    norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip, error_if_nonfinite=True)
    opt.step()
    if not all(torch.isfinite(v).all() for v in actor.parameters()):
        raise ValueError('nonfinite parameters after update')
    # These changes happen AFTER the only actor update, for the NEXT rollout.
    moments = RunningMoments.load(checkpoint['moments'])
    for ep in episodes:
        moments.update(ep['raw'])
    next_baseline = dict(baseline)
    for k in sorted({e['k'] for e in episodes}):
        rewards = [e['reward'] for e in episodes if e['k'] == k]
        mean = sum(rewards) / len(rewards)
        key = str(k)
        next_baseline[key] = (cfg.reward_ema_decay * baseline[key] + (1 - cfg.reward_ema_decay) * mean
                              if key in baseline else mean)
    result = pack(actor, opt, moments, next_baseline, cfg, checkpoint['batch_index'] + 1,
                  checkpoint['run_id'], checkpoint['smoke_only'])
    result['parent_version'] = checkpoint['version']
    result['metrics'] = dict(loss=loss_value, reward_mean=sum(e['reward'] for e in episodes)/len(episodes),
        advantage_mean=sum(advantages)/len(advantages), entropy=entropy_sum/max(1, free_steps),
        gradient_norm_before_clip=float(norm), trajectories=len(episodes), free_steps=free_steps,
        behavior_logp_max_error=max_error, optimizer_steps=1, stats_count=moments.count,
        parent_normalizer_count=checkpoint['moments']['count'],
        k_histogram={str(k): sum(e['k'] == k for e in episodes) for k in range(20, 41)})
    validate_checkpoint(result)
    return result
