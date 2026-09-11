"""Causal G1 features shared by offline replay and online decisions."""
import torch
import torch.nn.functional as F
from .contracts import FEATURE, FORCED, STAGE_STARTS


def pool(z):
    # Same operations and layout as formal Wan21 features.py::pool.
    direct = F.adaptive_avg_pool3d(z, (4,8,8)).flatten()
    stats = torch.cat((F.adaptive_avg_pool2d(z.mean(2), (8,8)),
                       F.adaptive_avg_pool2d(z.var(2, unbiased=False), (8,8))), 1).flatten()
    return direct, stats


class History:
    def __init__(self):
        self.previous = self.cache = self.current = None
        self.step, self.sigma, self.pending = -1, None, False

    @torch.no_grad()
    def observe(self, latent, step, sigma):
        if self.pending or step != self.step + 1 or not 0 <= step < 50:
            raise ValueError('consecutive observe/commit required')
        if not 0 <= sigma <= 1 or (self.sigma is not None and sigma >= self.sigma):
            raise ValueError('scheduler sigmas must decrease')
        x = latent.detach().half().float()
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if tuple(x.shape) != (1, *FEATURE['input_shape']) or not torch.isfinite(x).all():
            raise ValueError('Wan22 latent shape/finite contract')
        if step in STAGE_STARTS:
            self.previous = self.cache = None
        self.current, self.step, self.sigma, self.pending = x, step, sigma, True
        if self.cache is None:
            return x.new_zeros(18432)
        parts = [pool(z) for z in (x, self.previous, self.cache)]
        value = torch.cat((torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])))
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite G1 features')
        return value

    def commit(self, action):
        if not self.pending or action not in (0,1) or (self.step in FORCED and action):
            raise ValueError('invalid history commit/action')
        if action == 0:
            self.cache = self.current
        self.previous, self.pending = self.current, False
