"""Causal compact BLOC features; versioned independently of v1 contracts."""
from collections import deque
import math

import torch
import torch.nn.functional as F

VERSION = 'bloc_compact_v2_prototype'
EPS = 1e-6
DIMS = {'a': 64, 'ab': 128, 'ac': 80, 'abc': 144}
GROUPS = {f'bloc_{g}': (d, f'Compact BLOC {g.upper()}') for g, d in DIMS.items()}


def contract(group):
    if group not in GROUPS:
        raise ValueError('unknown BLOC group')
    return dict(version=VERSION, group=group, dim=GROUPS[group][0], channels=16,
                steps=50, stage_boundaries=[0], input_quantization='float16_then_float32',
                state_layout='feature_then_sea7', epsilon=EPS,
                relative_scale='half_sum_current_cached_channel_rms_plus_eps',
                pooling=[1, 2, 2], trend='difference_to_previous_and_past3_mean',
                absent_cache='zero_feature_and_original_cached_valid',
                stage_change='reset_cache_and_summary_history_require_recompute')


class CompactHistory:
    """observe raw input before action; commit only the action actually executed.

    A fresh instance per video. Explicit stage resets invalidate the cache and
    require recompute. The production Wan21 policy remains single-stage.
    """

    def __init__(self, group='abc'):
        if group not in DIMS:
            raise ValueError('unknown compact feature group')
        self.group = group
        self.last_step = -1
        self.pending = False
        self.stage = None
        self.reset_cache()

    def reset_cache(self):
        if self.pending:
            raise ValueError('cannot reset during a pending action')
        self.cache = None
        self.cache_rms = None
        self.past = deque(maxlen=3)
        self.previous_sigma = None

    @torch.no_grad()
    def observe(self, latent, step, sigma, *, stage='single'):
        if self.pending or step != self.last_step + 1 or not 0 <= step < 50:
            raise ValueError('requires consecutive observe/commit pairs')
        if not isinstance(stage, str) or not stage:
            raise ValueError('explicit nonempty stage required')
        if not math.isfinite(float(sigma)) or not 0 <= float(sigma) <= 1:
            raise ValueError('invalid sigma')
        if self.stage != stage:
            self.reset_cache()
            self.stage = stage
        if self.previous_sigma is not None and float(sigma) >= self.previous_sigma:
            raise ValueError('sigma must decrease within a stage')
        x = latent.detach()
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if x.ndim != 5 or x.shape[:2] != (1, 16):
            raise ValueError('expected one 16-channel latent')
        x = x.to(torch.float16).to(torch.float32).clone()
        if not torch.isfinite(x).all():
            raise ValueError('nonfinite latent')
        self.current = x
        self.current_rms = x.square().mean((2, 3, 4)).sqrt()
        self.current_q = None
        self.current_step = step
        self.current_sigma = float(sigma)
        self.pending = True
        if self.cache is None:
            return x.new_zeros((1, DIMS[self.group]))
        if x.shape != self.cache.shape:
            raise ValueError('latent shape changed without cache reset')
        d = x - self.cache
        s = (self.current_rms + self.cache_rms) / 2 + EPS
        product = self.current_rms * self.cache_rms
        valid = (self.current_rms > EPS) & (self.cache_rms > EPS)
        cosine = ((x * self.cache).mean((2, 3, 4)) / product.clamp_min(EPS**2)).clamp(-1, 1)
        cosine = torch.where(valid, cosine, torch.zeros_like(cosine))
        fields = torch.stack((d.square().mean((2, 3, 4)).sqrt() / s,
                              cosine, d.mean((2, 3, 4)) / s,
                              (self.current_rms - self.cache_rms) / s), dim=2)
        parts = [fields.flatten(1)]
        if 'b' in self.group:
            pooled = F.adaptive_avg_pool3d(d, (1, 2, 2)).flatten(2) / s[:, :, None]
            parts.append(pooled.flatten(1))
        if 'c' in self.group:
            q = torch.cat((fields.mean(1), fields.std(1, unbiased=False)), dim=1)
            self.current_q = q
            parts.append(torch.cat((q - self.past[-1], q - torch.stack(tuple(self.past)).mean(0)), 1)
                         if self.past else q.new_zeros((1, 16)))
        value = torch.cat(parts, 1)
        if value.shape != (1, DIMS[self.group]) or not torch.isfinite(value).all():
            raise ValueError('invalid compact feature output')
        return value

    def commit(self, action):
        if not self.pending or type(action) is not int or action not in (0, 1):
            raise ValueError('commit one executed binary action')
        if (self.cache is None or self.current_step in (0, 49)) and action:
            raise ValueError('uncached or boundary step must recompute')
        if not action:
            self.cache = self.current
            self.cache_rms = self.current_rms
        if self.current_q is not None:
            self.past.append(self.current_q)
        self.previous_sigma = self.current_sigma
        self.last_step = self.current_step
        self.pending = False
