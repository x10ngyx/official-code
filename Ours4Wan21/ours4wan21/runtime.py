"""RL gate over the unchanged local SeaCache4Wan21 residual/filter implementation."""
import importlib.util
from pathlib import Path
import sys
import types
import time

import torch

from .contracts import FORCED, OFFICIAL, STEPS, observation, state_contract, latent_group
from .local_iql import required_hard_budget_action
from .policy import resolve_budget


def _load_seacache():
    """Give the sibling's absolute import a private package namespace."""
    package = '_ours4wan21_seacache_reference'
    root = OFFICIAL / 'SeaCache4Wan21'
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(root)]
        sys.modules[package] = module
    name = package + '.seacache'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / 'seacache.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


reference = _load_seacache()


class Controller(reference.SeaCacheController):
    def __init__(self, policy, budget):
        self.policy = policy
        self.budget = resolve_budget(skip_budget=budget)
        super().__init__(reference.SeaCacheConfig(threshold=1.))

    def reset(self):
        super().reset()
        if hasattr(self.policy, 'reset_measurements'):
            self.policy.reset_measurements()
        group = latent_group(self.policy.mode)
        from .latent_features import LatentFeatureHistory
        self.feature_history = LatentFeatureHistory([group]) if group else None
        self.current_latent_feature = None
        self.feature_wall_seconds = []
        self.used = {'cond': 0, 'uncond': 0}
        self.consecutive = {'cond': 0, 'uncond': 0}

    def observe_latent(self, latent, step):
        if self.feature_history is None:
            return
        started = time.perf_counter()
        with torch.autocast(device_type=latent.device.type, enabled=False):
            values = self.feature_history.observe(latent, step, self.scheduler_sigmas[step].item())
            # CPU state serialization/normalization uses the same FP32 values as offline.
            self.current_latent_feature = values[latent_group(self.policy.mode)][0].cpu()
        self.feature_wall_seconds.append(time.perf_counter()-started)

    def feature_overhead(self):
        return dict(group=latent_group(self.policy.mode), calls=len(self.feature_wall_seconds),
                    wall_seconds=sum(self.feature_wall_seconds),
                    per_step_wall_seconds=self.feature_wall_seconds,
                    scope='observe and causal feature extraction through CPU state transfer, including GPU queue waits; nested in DiT/generate; excludes predictor network',
                    tflops=None, tflops_scope='feature FFT/quantile/reductions are not counted as predictor MLP FLOPs')

    def set_scheduler_sigmas(self, sigmas):
        if sigmas is None or len(sigmas) < STEPS or not torch.isfinite(sigmas).all():
            raise ValueError('RL SEA state requires actual finite scheduler sigmas')
        super().set_scheduler_sigmas(sigmas)

    def plan_step(self, *, branch, step_index, num_steps, feature, grid_size):
        if num_steps != STEPS:
            raise ValueError('Ours4Wan21 requires 50 steps')
        self._validate_step(branch, step_index, num_steps, feature, grid_size)
        if self.scheduler_sigmas is None:
            raise RuntimeError('set scheduler sigmas before inference')
        # Inherit the local corrected filter, including forced boundaries.
        filtered = self._filter_feature(feature, grid_size, step_index, num_steps)
        before = self.accumulators[branch]
        d = total = 0.
        if step_index not in FORCED:
            d = self._relative_l1(filtered, self.previous_features[branch])
            total = before + d
        state = observation(self.policy.mode, step=step_index, budget=self.budget,
            used=self.used[branch], consecutive=self.consecutive[branch],
            cached_valid=branch in self.residuals, adjacent=d, accumulated=total,
            latent=self.current_latent_feature)
        required, reason = required_hard_budget_action(step_index=step_index,
            used_skips=self.used[branch], skip_budget=self.budget, num_steps=STEPS,
            forced_steps=FORCED, current_forced_recompute=branch not in self.residuals)
        if branch == 'cond':
            if required is None:
                action, probability = self.policy.choose(state)
                reason = getattr(self.policy, 'action_mode', 'policy_argmax')
            else:
                action, probability = required, None
        else:
            cond = self.decisions[-1]
            # T2V first-block modulated input precedes text conditioning and
            # must agree across CFG branches. Never silently collapse mixed data.
            if cond['branch'] != 'cond' or cond['step_index'] != step_index:
                raise RuntimeError('uncond must follow its explicit cond step')
            if not torch.allclose(state, torch.tensor(cond['state']), atol=2e-6, rtol=1e-6):
                raise RuntimeError('CFG scalar states disagree; shared step policy is invalid')
            if abs(d - cond['filtered_relative_l1']) > 2e-6 or abs(total - cond['accumulated_distance_with_current']) > 2e-6:
                raise RuntimeError('CFG SEA signals disagree')
            action = int(cond['action'] == 'reuse')
            probability, reason = cond['p_skip'], cond['reason']
            if required is not None and action != required:
                raise RuntimeError('CFG exact-K constraints disagree')
        if action not in (0, 1):
            raise RuntimeError('policy action must be binary')
        if branch == 'cond' and self.feature_history is not None:
            self.feature_history.commit(action)
        self.accumulators[branch] = total if action else 0.
        self.previous_features[branch] = filtered.detach().clone()
        decision = dict(call_index=self._expected_call_index, branch=branch,
            step_index=step_index, action='reuse' if action else 'recompute', reason=reason,
            state=state.tolist(), state_mode=self.policy.mode, skip_budget=self.budget,
            used_skips_before=self.used[branch], consecutive_skips_before=self.consecutive[branch],
            p_skip=probability, policy_queried=branch == 'cond' and required is None,
            actor_mask=float(required is None), native_forced_recompute=step_index in FORCED,
            behavior_action_probability=(probability if action else 1-probability) if probability is not None else 1.,
            filtered_relative_l1=d, accumulated_distance_before=before,
            accumulated_distance_with_current=total, accumulated_distance_after=self.accumulators[branch],
            stored_feature='sea_filtered', execution=None)
        self.used[branch] += action
        self.consecutive[branch] = self.consecutive[branch] + 1 if action else 0
        self.decisions.append(decision)
        self._pending_decision = decision
        self._expected_call_index += 1
        return bool(action)

    def summary(self):
        payload = super().summary()
        if len(self.decisions) != 100 or any(v != self.budget for v in self.used.values()):
            raise RuntimeError('incomplete trajectory or exact-K violation')
        for key in ('threshold', 'use_ret_steps'):
            payload.pop(key)
        payload.update(schema='ours4wan21_policy_trace_v1',
            gate_mode='exact_K_shared_step_policy_independent_CFG_residuals',
            state_contract=state_contract(self.policy.mode),
            forced_steps=list(FORCED), skip_budget=self.budget,
            step_reuse=self.used['cond'], step_recompute=50-self.used['cond'],
            actor_queries=sum(d['policy_queried'] for d in self.decisions),
            action_mode=getattr(self.policy, 'action_mode', 'policy_argmax'),
            sampling_seed=getattr(self.policy, 'sampling_seed', None),
            latent_feature_overhead=self.feature_overhead())
        return payload

    def write_trace(self, path=None, extra=None):
        # The sibling sampler calls this before VAE decode. Defer disk output
        # to the runner so measured generate time excludes trace serialization.
        self.trace_extra = dict(extra or {})
        return None


def integration_functions():
    """Load the exact sibling forward/sampler, resolving its absolute import safely."""
    name = '_ours4wan21_seacache_reference.integration'
    if name not in sys.modules:
        original = sys.modules.get('seacache')
        sys.modules['seacache'] = reference
        try:
            path = OFFICIAL / 'SeaCache4Wan21/wan21_integration.py'
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        finally:
            if original is None:
                sys.modules.pop('seacache', None)
            else:
                sys.modules['seacache'] = original
    return sys.modules[name]


def apply_policy(pipeline, policy, budget):
    integration = integration_functions()
    controller = Controller(policy, budget)
    pipeline.model.seacache_controller = controller
    def forward(model, x, *args, **kwargs):
        if kwargs.get('seacache_branch') == 'cond' and controller.feature_history is not None:
            if len(x) != 1:
                raise ValueError('feature input requires one candidate latent')
            controller.observe_latent(x[0], kwargs['seacache_step_index'])
        return integration.seacache_forward(model, x, *args, **kwargs)
    pipeline.model.forward = types.MethodType(forward, pipeline.model)
    def generate(self, input_prompt, size=(832, 480), frame_num=81, shift=5.,
                 sample_solver='unipc', sampling_steps=50, guide_scale=5.,
                 n_prompt='', seed=42, offload_model=False):
        if (tuple(size), frame_num, shift, sample_solver, sampling_steps,
                guide_scale, seed, offload_model) != ((832, 480), 81, 5., 'unipc', 50, 5., 42, False):
            raise ValueError('Ours4Wan21 generation protocol is frozen')
        if self.t5_cpu or self.rank != 0 or self.sp_size != 1 or self.param_dtype != torch.bfloat16:
            raise ValueError('requires single-device BF16 pipeline with T5 on GPU')
        return integration.t2v_generate(self, input_prompt, size=size, frame_num=frame_num,
            shift=shift, sample_solver=sample_solver, sampling_steps=sampling_steps,
            guide_scale=guide_scale, n_prompt=n_prompt, seed=seed, offload_model=offload_model)
    pipeline.generate = types.MethodType(generate, pipeline)
    return controller
