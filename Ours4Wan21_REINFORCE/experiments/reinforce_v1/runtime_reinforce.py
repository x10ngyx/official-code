"""CNN+G1 categorical gate over the existing Exact-K/SeaCache implementation."""
import time
import types

import torch

from core import (CNN, DIM, GENERATION, PROTOCOL, CONTRACT, ARCH,
                  encode_normalized, validate_checkpoint)
from features import History, pool
from ours4wan21 import runtime as rt
from ours4wan21.policy import Policy as MeasuredPolicy


class G1History(History):
    """Same raw G1 formula, without evaluating unused G2/G3/G4 filters."""
    @torch.no_grad()
    def observe(self, latent, step, sigma):
        if self.pending or step != self.step + 1 or not 0 <= step < 50:
            raise ValueError('history observe/commit order')
        if not 0 <= sigma <= 1 or (self.sigma is not None and sigma >= self.sigma):
            raise ValueError('sigma order')
        x = latent.detach().half().float()
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if x.shape != (1, 16, 21, 60, 104) or not torch.isfinite(x).all():
            raise ValueError('G1 latent shape/finite contract')
        self.current, self.step, self.sigma, self.pending = x, step, sigma, True
        if self.cache is None:
            return x.new_zeros(DIM - 7)
        parts = [pool(z) for z in (x, self.previous, self.cache)]
        return torch.cat([p[0] for p in parts] + [p[1] for p in parts])


class Policy(MeasuredPolicy):
    def __init__(self, checkpoint, device='cuda', sampling_seed=None, allow_smoke=False):
        validate_checkpoint(checkpoint)
        if checkpoint['smoke_only'] and not allow_smoke:
            raise ValueError('smoke checkpoint rejected for production')
        self.device = torch.device(device)
        self.mode, self.group = 'sea7', 'G1'
        self.net = CNN('G1', 2).to(self.device).eval()
        self.net.load_state_dict(checkpoint['policy_net'])
        self.normalizer = {k: v.cpu().clone() for k, v in checkpoint['normalizer'].items()}
        self.version = checkpoint['version']
        self.current_feature = None
        self.flops_profile = dict(flops_per_call=2 * 12290304,
            convention='2_flops_per_mac', scope='CNN Conv/Linear only; excludes activation, normalization, pooling and SEA',
            breakdown={'conv_linear': 2 * 12290304})
        self._event_pairs = []
        if self.device.type == 'cuda':
            with torch.no_grad(), torch.autocast('cuda', enabled=False), torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
                self.net(torch.zeros(1, DIM, device=self.device))
            for _ in range(48):
                pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for e in pair:
                    e.record(torch.cuda.current_stream(self.device))
                self._event_pairs.append(pair)
            self._event_pairs[-1][1].synchronize()
        self.set_sampling(sampling_seed)
        self.reset_measurements()

    def reset_measurements(self):
        super().reset_measurements()
        self.log_probs = []

    @torch.no_grad()
    def choose(self, scalar):
        if self.current_feature is None:
            raise ValueError('missing pre-action G1')
        index = len(self._measurements)
        if index >= 48:
            raise ValueError('too many actor queries')
        start, end = self._event_pairs[index] if self.device.type == 'cuda' else (None, None)
        began = time.perf_counter()
        with torch.autocast(device_type=self.device.type, enabled=False), torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
            raw = torch.cat((self.current_feature, scalar.cpu()))
            # Identical CPU normalization arithmetic during rollout and replay.
            x = encode_normalized(raw, self.normalizer).float().unsqueeze(0).to(self.device)
            if start is not None:
                start.record(torch.cuda.current_stream(self.device))
            tick = time.perf_counter()
            logits = self.net(x)[0]
            host = time.perf_counter() - tick
            if end is not None:
                end.record(torch.cuda.current_stream(self.device))
            if not torch.isfinite(logits).all():
                raise ValueError('nonfinite logits')
            logp = logits.log_softmax(-1)
            probability = float(logits.softmax(-1)[1])
            if self.sampling_generator is None:
                action = int(logits.argmax())
            else:
                action = int(torch.rand((), generator=self.sampling_generator).item() < probability)
            self.log_probs.append(float(logp[action]))
        self._measurements.append(dict(network_host_span_seconds=host,
                                       decision_wall_seconds=time.perf_counter() - began))
        return action, probability


class Controller(rt.Controller):
    allow_model_offload = True

    def reset(self):
        super().reset()
        self.feature_history = G1History()
        self.policy.current_feature = None
        self.raw_states = []

    def observe_latent(self, latent, step):
        tick = time.perf_counter()
        with torch.autocast(device_type=latent.device.type, enabled=False):
            feature = self.feature_history.observe(latent, step, float(self.scheduler_sigmas[step])).cpu()
        if not torch.isfinite(feature).all():
            raise ValueError('nonfinite pooled G1')
        self.policy.current_feature = feature
        self.current_latent_feature = None
        self.feature_wall_seconds.append(time.perf_counter() - tick)

    def plan_step(self, **kwargs):
        reuse = super().plan_step(**kwargs)
        if kwargs['branch'] == 'cond':
            self.raw_states.append(torch.cat((self.policy.current_feature,
                                             torch.tensor(self.decisions[-1]['state']))))
        return reuse

    def feature_overhead(self):
        return dict(group='G1', calls=len(self.feature_wall_seconds),
                    wall_seconds=sum(self.feature_wall_seconds), tflops=None,
                    scope='pooling and CPU feature transfer, includes queue waits; nested in DiT/generate')

    def summary(self):
        out = super().summary()
        if len(self.raw_states) != 50:
            raise ValueError('incomplete state sequence')
        out.update(policy_family='REINFORCE_CNN_G1', policy_version=self.policy.version,
                   protocol=PROTOCOL, feature_contract=CONTRACT, architecture=ARCH)
        return out

    def episode(self, job, checkpoint, reward):
        rows = self.summary()['decisions'][::2]
        raw = torch.stack(self.raw_states).float()
        logp = torch.zeros(50)
        free = torch.tensor([r['policy_queried'] for r in rows])
        logp[free] = torch.tensor(self.policy.log_probs)
        return dict(trajectory_id=job['id'], batch_index=checkpoint['batch_index'],
                    policy_version=self.policy.version, run_id=checkpoint['run_id'],
                    sample_id=job['sample_id'], split=job['split'], k=job['k'],
                    sampling_seed=job.get('sampling_seed'), action_mode=self.policy.action_mode,
                    raw=raw, inputs=encode_normalized(raw, self.policy.normalizer),
                    actions=torch.tensor([int(r['action'] == 'reuse') for r in rows]),
                    free=free, logp=logp, reward=float(reward))


def apply_policy(pipeline, policy, budget):
    integration = rt.integration_functions()
    controller = Controller(policy, budget)
    pipeline.model.seacache_controller = controller

    def forward(model, x, *args, **kwargs):
        if kwargs.get('seacache_branch') == 'cond':
            if len(x) != 1:
                raise ValueError('requires batch=1')
            controller.observe_latent(x[0], kwargs['seacache_step_index'])
        return integration.seacache_forward(model, x, *args, **kwargs)

    pipeline.model.forward = types.MethodType(forward, pipeline.model)

    def generate(self, input_prompt, **kwargs):
        if kwargs != GENERATION:
            raise ValueError('requires fixed Wan21 offload protocol')
        if self.t5_cpu or self.rank != 0 or self.sp_size != 1 or self.param_dtype != torch.bfloat16:
            raise ValueError('requires single GPU BF16, T5 encoded on GPU')
        return integration.t2v_generate(self, input_prompt, **kwargs)

    pipeline.generate = types.MethodType(generate, pipeline)
    return controller
