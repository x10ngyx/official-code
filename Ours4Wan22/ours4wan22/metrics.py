"""Strict shared component schema and trace-weighted Wan22 Calflops."""
import math
from .shared import load, OFFICIAL
import sys

sys.path.insert(0, str(OFFICIAL / 'ComponentMetrics'))
from reporting import extract_component_latency, extract_component_tflops
aggregate = load('sea_aggregate', 'SeaCache4Wan22/experiments/performance_t2v_a14b/aggregate_performance.py')
load_profile = aggregate.load_profile


def performance(timing, profile, trace=None):
    if timing.get('status') != 'success':
        raise ValueError('inference did not succeed')
    calls = timing.get('calls', [])
    if len(calls) != 100:
        raise ValueError('50 steps require 100 CFG calls')
    block_count = profile['input']['transformer_blocks']
    if timing['transformer_block_count_by_stage'] != dict(high=block_count, low=block_count):
        raise ValueError('profile block count mismatch')
    if trace is not None:
        decisions = trace['decisions']
        if len(decisions) != 50 or sum(d['action']=='reuse' for d in decisions) != trace['skip_budget']:
            raise ValueError('invalid Exact-K trace')
    flops = cuda = 0.
    for i, call in enumerate(calls):
        step, stage, branch = aggregate.expected_call_identity(i)
        if (call['step_index'], call['model_stage'], call['cfg_branch']) != (step,stage,branch):
            raise ValueError('timing call identity mismatch')
        action = trace['decisions'][step]['action'] if trace else 'recompute'
        if action not in ('reuse','recompute'):
            raise ValueError('invalid action')
        expected = 0 if action == 'reuse' else block_count
        if call['blocks_executed'] != expected or call['full_compute'] != bool(expected) or call['reuse'] != (expected == 0):
            raise ValueError('actual block execution disagrees with action')
        if trace:
            d = trace['decisions'][step]
            if d['stage'] != stage or d['step_index'] != step or d['branches'].get(branch) != action:
                raise ValueError('CFG trace mismatch')
            if step in (0,32,49) and action != 'recompute':
                raise ValueError('native forced step reused')
        entry = profile['stages'][stage]['branches'][branch]
        flops += entry['estimated_full_flops' if expected else 'estimated_always_on_flops']
        cuda += aggregate.finite_nonnegative(call['cuda_seconds'], 'DiT CUDA')
    if not math.isclose(cuda, timing['model_forward_cuda_seconds'], rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError('DiT timing sum mismatch')
    seconds = aggregate.finite_nonnegative(timing['pipeline_generate_wall_seconds'], 'generate latency')
    if seconds == 0:
        raise ValueError('zero latency')
    return dict(pipeline_generate_wall_seconds=seconds, estimated_dit_tflops=flops/1e12,
                **extract_component_latency(timing), **extract_component_tflops(profile),
                predictor=timing.get('predictor'), latent_feature=timing.get('latent_feature'),
                scope='full generate including transfer/offload and controller; excludes model load/export/evaluation. DiT TFLOPs exclude separately reported predictor and uncounted SEA/filter/pooling/residual/scheduler operations.')
