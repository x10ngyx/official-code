"""Single-skip controller on the locked, shared Wan21 sampler/forward.

No threshold decisions. Last-step skip is explicitly enabled for this causal
experiment, unlike the production SeaCache/Exact-K boundary policy.
"""
import hashlib
import sys
import types
from common import PACKAGES, validate_trace
sys.path.insert(0, str(PACKAGES/'SeaCache4Wan21'))
import torch
import torch.nn.functional as F
from seacache import SeaCacheConfig, SeaCacheController
from wan21_integration import seacache_forward, t2v_generate


def tensor_sha(tensor):
    x = tensor.detach().contiguous().cpu()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()


def pool_g1(z):
    # Exact raw G1 pooling/quantization from the formal CNN feature contract.
    z = z.detach().to(torch.float16).float().unsqueeze(0)
    direct = F.adaptive_avg_pool3d(z, (4,8,8)).flatten()
    stats = torch.cat((F.adaptive_avg_pool2d(z.mean(2), (8,8)),
                       F.adaptive_avg_pool2d(z.var(2, unbiased=False), (8,8))), 1).flatten()
    return direct.cpu(), stats.cpu()


class SingleSkipController(SeaCacheController):
    def __init__(self, skip_step, baseline_features=None):
        if type(skip_step) is not int or not 0 <= skip_step <= 50 or skip_step == 1:
            raise ValueError('skip_step is 0 (baseline) or one-based 2..50')
        self.skip_step = skip_step
        self.baseline_features = baseline_features
        super().__init__(SeaCacheConfig(threshold=1.0))

    def reset(self):
        super().reset()
        self.cache_steps = {}
        self.raw_previous = {}
        self.latent_hashes = []
        self.g1 = {}
        self.proxies = {}
        self.previous_pool = None
        self.input_latents = []
        self.step_metadata = []

    def observe_latent(self, latent, index, timestep=None):
        if index != len(self.input_latents):
            raise ValueError('capture all 50 ordered pre-action latents')
        # Retain on the original device; storage conversion/I/O is after generate.
        self.input_latents.append(latent.detach().clone())
        self.step_metadata.append(dict(step_index=index, step_fraction=index/49,
            timestep=float(timestep.detach().flatten()[0].cpu()) if timestep is not None else None,
            sigma=float(self.scheduler_sigmas[index].detach().cpu()) if self.scheduler_sigmas is not None else None,
            model_stage='single'))
        # Only pre-action states; candidate prefix must be bitwise identical.
        if self.skip_step and index >= self.skip_step:
            return
        value = tensor_sha(latent)
        self.latent_hashes.append(value)
        if self.baseline_features is not None:
            if value != self.baseline_features['latent_hashes'][index]:
                raise ValueError(f'baseline prefix mismatch at execution step {index+1}')
        if self.skip_step == 0 or index == self.skip_step-2 or index == self.skip_step-1:
            current = pool_g1(latent)
            if index > 0 and (not self.skip_step or index == self.skip_step-1):
                if self.previous_pool is None:
                    raise ValueError('missing preceding latent')
                # All earlier steps recompute: previous latent == cached-input latent.
                prev = self.previous_pool
                self.g1[index+1] = torch.cat((current[0], prev[0], prev[0],
                                              current[1], prev[1], prev[1]))
            self.previous_pool = current

    def plan_step(self, *, branch, step_index, num_steps, feature, grid_size):
        self._validate_step(branch, step_index, num_steps, feature, grid_size)
        if num_steps != 50:
            raise ValueError('fixed 50-step protocol')
        # Complete data-pipeline trace: measure every branch, including the suffix.
        observe = True
        raw_d = sea_d = None
        if observe:
            filtered = self._filter_feature(feature, grid_size, step_index, num_steps)
            if branch in self.raw_previous:
                raw_d = self._relative_l1(feature, self.raw_previous[branch])
                sea_d = self._relative_l1(filtered, self.previous_features[branch])
                self.proxies.setdefault(step_index+1, {})[branch] = dict(
                    modulated_input_relative_l1=raw_d, sea_relative_l1=sea_d)
            self.raw_previous[branch] = feature.detach().clone()
            self.previous_features[branch] = filtered.detach().clone()
        reuse = self.skip_step != 0 and step_index == self.skip_step-1
        if reuse and self.cache_steps.get(branch) != step_index-1:
            raise ValueError('the immediately previous step must refresh cache')
        before = self.accumulators[branch]
        total = before + (sea_d if sea_d is not None else 0.)
        self.accumulators[branch] = total if reuse else 0.
        row = dict(call_index=self._expected_call_index, step_index=step_index,
                   branch=branch, action='reuse' if reuse else 'recompute',
                   execution=None, reason='single_step_intervention' if reuse else 'all_other_steps_full',
                   cache_source_step_index=self.cache_steps.get(branch),
                   raw_proxy=raw_d, sea_proxy=sea_d,
                   filtered_relative_l1=sea_d,
                   accumulated_distance_before=before, accumulated_distance_with_current=total,
                   accumulated_distance_after=self.accumulators[branch],
                   requested_threshold=None, stored_feature='sea_filtered',
                   distance_reference='previous_step_same_cfg_branch',
                   distance_feature='sea_filtered_first_block_modulated_input',distance_metric='relative_l1_mean',
                   native_forced_recompute=step_index==0,
                   controller_kind='fixed_single_skip; accumulators diagnostic only')
        self.decisions.append(row)
        self._pending_decision = row
        self._expected_call_index += 1
        return reuse

    def record_recompute(self, branch, step_index, residual):
        super().record_recompute(branch, step_index, residual)
        self.cache_steps[branch] = step_index

    def write_trace(self, path=None, extra=None):
        # Disk writes are deliberately outside measured generate().
        validate_trace(self.summary(), self.skip_step)

    def summary(self):
        return dict(schema='single_skip_cfg_trace_v1', skip_step_1based=self.skip_step,
                    decisions=self.decisions)

    def features(self):
        return dict(schema='single_skip_features_v1', latent_hashes=self.latent_hashes,
                    g1=self.g1, proxies=self.proxies,
                    layout='current_previous_cached 3d pools, then each role mean/variance pools',
                    g1_dim=18432, quantization='float16_then_float32',
                    source='pre-action raw latent; no future features; no learned encoder')


def install(pipe, controller):
    pipe.model.seacache_controller = controller
    pipe.generate = types.MethodType(t2v_generate, pipe)

    def forward(model, x, *args, **kwargs):
        if kwargs['seacache_branch'] == 'cond':
            controller.observe_latent(x[0], kwargs['seacache_step_index'],kwargs['t'])
        return seacache_forward(model, x, *args, **kwargs)

    pipe.model.forward = types.MethodType(forward, pipe.model)
