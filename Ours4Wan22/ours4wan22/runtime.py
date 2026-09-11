"""Exact-K policy over the locked Wan22 SeaCache residual execution path."""
from contextlib import contextmanager
import math
import time
import torch
from .contracts import FORCED, budget, scalar_state
from .features import History
from .shared import load
from .model import reference as _cnn  # establishes the shared local_iql import
from ours4wan21.local_iql import required_hard_budget_action

sea = load('sea', 'SeaCache4Wan22/runtime/seacache.py')


class Controller(sea.SeaCacheController):
    def __init__(self, policy, k):
        super().__init__(sea.SeaCacheConfig(threshold=1.))
        self.policy, self.k = policy, budget(k)
        self.used = self.consecutive = 0
        self.history = History()
        self.latent_feature = None
        self.feature_times, self.saved_features = [], []
        self.policy.reset_measurements()

    def set_scheduler_sigmas(self, sigmas):
        if (not torch.is_tensor(sigmas) or sigmas.ndim != 1 or len(sigmas) < 50
                or not torch.isfinite(sigmas).all() or not ((sigmas[:50] >= 0) & (sigmas[:50] <= 1)).all()
                or not (sigmas[:49] > sigmas[1:50]).all()):
            raise ValueError('actual decreasing scheduler sigmas required')
        super().set_scheduler_sigmas(sigmas.detach().cpu())

    def observe_latent(self, latent, step):
        if self.scheduler_sigmas is None:
            raise ValueError('scheduler not initialized')
        began = time.perf_counter()
        with torch.autocast(latent.device.type, enabled=False):
            self.latent_feature = self.history.observe(latent, step, float(self.scheduler_sigmas[step])).cpu()
        self.saved_features.append(self.latent_feature)
        self.feature_times.append(time.perf_counter()-began)

    def plan_step(self, *, stage, step_index, num_steps, feature, grid_size):
        self._validate_step(stage, step_index, num_steps, feature, grid_size)
        if num_steps != 50 or stage != ('high' if step_index < 32 else 'low'):
            raise ValueError('requires actual 32/18 Wan22 stage split')
        if self._current_step == step_index:
            # cond has executed; the explicit uncond call receives its shared action.
            if self._current_stage != stage or set(self.decisions[-1]['branches']) != {'cond'}:
                raise RuntimeError('invalid CFG execution order')
            return self.decisions[-1]['action'] == 'reuse'
        if step_index != len(self.decisions) or (self.decisions and set(self.decisions[-1]['branches']) != {'cond','uncond'}):
            raise RuntimeError('incomplete or nonconsecutive CFG step')
        if self.history.step != step_index or not self.history.pending:
            raise RuntimeError('observe current raw latent before gating')
        state = self.states.setdefault(stage, sea._StageState())
        filtered = self._filter_feature(feature, grid_size, step_index, num_steps)
        cached = set(state.residuals) == {'cond','uncond'}
        d = total = 0.
        before = state.accumulator
        if step_index not in FORCED:
            if state.previous_feature is None or not cached:
                raise RuntimeError('unexpected cache loss in eligible step')
            d = self._relative_l1(filtered, state.previous_feature)
            total = before + d
        scalar = scalar_state(step_index, self.k, self.used, self.consecutive, cached, d, total)
        required, reason = required_hard_budget_action(step_index=step_index,
            used_skips=self.used, skip_budget=self.k, num_steps=50, forced_steps=FORCED,
            current_forced_recompute=not cached)
        if required is None:
            action, probability = self.policy.choose(torch.cat((self.latent_feature, scalar)))
            reason = 'policy_argmax'
            if action not in (0,1) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('invalid actor decision')
        else:
            action, probability = required, None
        self.history.commit(action)
        state.accumulator = total if action else 0.
        state.previous_feature = filtered.detach().clone()
        self._current_step, self._current_stage = step_index, stage
        self.decisions.append(dict(step_index=step_index, stage=stage,
            action='reuse' if action else 'recompute', reason=reason, branches={},
            state=scalar.tolist(), state_mode='cnn_G1', skip_budget=self.k,
            used_skips_before=self.used, consecutive_skips_before=self.consecutive,
            p_skip=probability, policy_queried=required is None, actor_mask=float(required is None),
            native_forced_recompute=step_index in FORCED, relative_l1=d,
            accumulator_before=before, accumulated_distance_with_current=total,
            accumulator_after=state.accumulator))
        self.used += action
        self.consecutive = self.consecutive + 1 if action else 0
        return bool(action)

    def summary(self):
        if len(self.decisions) != 50 or self.used != self.k or any(
                set(d['branches']) != {'cond','uncond'} for d in self.decisions):
            raise RuntimeError('incomplete trajectory or Exact-K violation')
        # The reference sampler logs this before decode. Keep it small and defer
        # full trace/feature serialization and CUDA-event reads until after generate.
        return dict(schema='ours4wan22_cnn_G1_trace_v1', total_steps=50,
                    reuse=self.used, recompute=50-self.used, skip_budget=self.k)

    def trace(self):
        from .contracts import FEATURE, SCALARS
        return dict(self.summary(), forced_steps=list(FORCED), decisions=self.decisions,
            scalar_names=list(SCALARS), feature_contract=FEATURE,
            latent_feature_file='latent_features.pt',
            policy_input='concat(latent_features[step], decisions[step].state); normalize then FP16->FP32',
            latent_feature_overhead=dict(calls=len(self.feature_times),
                wall_seconds=sum(self.feature_times), per_step_wall_seconds=self.feature_times,
                tflops=None, scope='raw quantization, statistics/pooling and CPU transfer; nested in DiT/generate; feature FLOPs uncounted'))

    def write_trace(self, path=None, extra=None):
        return None


@contextmanager
def apply_policy(pipeline, policy, k):
    """Scoped injection into the validated prepared SeaCache sampler.

    Both experts retain the reference model.forward. Their wrapper observes raw
    latents using explicit branch/step metadata; it never infers branch by parity.
    The sampler's factory is restored even if generation fails.
    """
    import wan.text2video as sampler
    if (pipeline.t5_cpu or pipeline.rank != 0 or pipeline.sp_size != 1
            or pipeline.t5_fsdp or pipeline.dit_fsdp or pipeline.use_sp
            or pipeline.param_dtype != torch.bfloat16 or pipeline.boundary != .875):
        raise ValueError('single GPU BF16 Wan22 protocol required')
    controller = Controller(policy, k)
    original_factory = sampler.SeaCacheController
    original_generate = pipeline.generate
    models = (pipeline.high_noise_model, pipeline.low_noise_model)
    originals = [model.forward for model in models]
    sampler.SeaCacheController = lambda config: controller
    try:
        def generate(input_prompt, size=(832,480), frame_num=45, shift=12.,
                     sample_solver='dpm++', sampling_steps=50, guide_scale=(3.,4.),
                     n_prompt='', seed=42, offload_model=True, seacache_config=None):
            if (tuple(size),frame_num,shift,sample_solver,sampling_steps,tuple(guide_scale),seed,offload_model) != (
                    (832,480),45,12.,'dpm++',50,(3.,4.),42,True) or seacache_config is None:
                raise ValueError('fixed Wan22 generation protocol required')
            return original_generate(input_prompt,size=size,frame_num=frame_num,shift=shift,
                sample_solver=sample_solver,sampling_steps=sampling_steps,guide_scale=guide_scale,
                n_prompt=n_prompt,seed=seed,offload_model=offload_model,seacache_config=seacache_config)
        pipeline.generate = generate
        for model, original in zip(models, originals):
            def forward(x, *args, _original=original, **kwargs):
                if kwargs.get('seacache') is not controller:
                    raise ValueError('missing policy controller')
                branch, step = kwargs.get('seacache_branch'), kwargs.get('seacache_step_index')
                if branch == 'cond':
                    if len(x) != 1 or step != len(controller.decisions):
                        raise ValueError('explicit batch1 cond order required')
                    controller.observe_latent(x[0], step)
                elif branch != 'uncond' or controller._current_step != step:
                    raise ValueError('explicit uncond must follow cond')
                return _original(x, *args, **kwargs)
            model.forward = forward
        yield controller
    finally:
        sampler.SeaCacheController = original_factory
        pipeline.generate = original_generate
        for model, original in zip(models, originals):
            model.forward = original
