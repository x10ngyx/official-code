"""Predictor-only operation estimates and per-video timing aggregation."""
import math
import torch


def network_flops(net):
    """Batch-one FP32 forward, matching local Calflops 0.3.2 conventions.

    Linear: 2 MAC operations, bias excluded; affine LayerNorm: 5/element;
    SiLU: 1/element. These are estimator conventions, not instruction counts.
    Input normalization, output softmax/argmax and SEA filtering are excluded.
    """
    counts = dict(linear=0, layer_norm=0, silu=0)
    width = None
    for layer in net.net:
        if isinstance(layer, torch.nn.Linear):
            if width is not None and layer.in_features != width:
                raise ValueError('inconsistent predictor dimensions')
            counts['linear'] += 2 * layer.in_features * layer.out_features
            width = layer.out_features
        elif isinstance(layer, torch.nn.LayerNorm):
            if tuple(layer.normalized_shape) != (width,):
                raise ValueError('unsupported predictor LayerNorm shape')
            counts['layer_norm'] += width * (5 if layer.elementwise_affine else 4)
        elif isinstance(layer, torch.nn.SiLU):
            counts['silu'] += width
        else:
            raise ValueError(f'unprofiled predictor operator: {type(layer).__name__}')
    return dict(flops_per_call=sum(counts.values()), breakdown=counts,
        convention='calflops_0.3.2_batch1_forward_estimate',
        scope='policy_net only: Linear (2 FLOPs/MAC, bias excluded), LayerNorm (affine 5/element), SiLU (1/element); excludes input normalization, softmax/argmax, transfers and SEA filter')


def summarize_calls(calls, profile, device_type):
    gpu = device_type == 'cuda'
    return dict(schema='ours4wan21_predictor_overhead_v1', device_type=device_type,
        call_count=len(calls), profile=profile,
        tflops=len(calls) * profile['flops_per_call'] / 1e12,
        network_cuda_seconds=sum(c['network_cuda_seconds'] for c in calls) if gpu else None,
        network_host_span_seconds=sum(c['network_host_span_seconds'] for c in calls),
        decision_wall_seconds=sum(c['decision_wall_seconds'] for c in calls),
        calls=calls,
        timing_scope='network CUDA events bracket policy_net.forward only; network host span is asynchronous dispatch on CUDA and execution on CPU; decision wall covers state transfer, normalization, forward, finite check, softmax/argmax and host scalar retrieval. SEA state construction/filtering and exact-K gate excluded.',
        accounting='Nested inside DiT/generate timings; do not add again. CUDA events resolved after generate, without additional per-query synchronization.')


def predictor_fields(timing, trace):
    """Strictly validate persisted measurements before including them in reports."""
    p = timing.get('predictor')
    if not isinstance(p, dict) or p.get('schema') != 'ours4wan21_predictor_overhead_v1':
        raise ValueError('predictor overhead was not measured; cannot infer it from old timings')
    n = p['call_count']
    decisions = trace['decisions']
    if type(n) is not int or not 0 <= n <= 48 or len(decisions) != 100:
        raise ValueError('invalid predictor count/trace')
    actual = sum(d.get('policy_queried') is True for d in decisions)
    if actual != n or trace['actor_queries'] != n or len(p['calls']) != n:
        raise ValueError('predictor call count disagrees with actual actor trace')
    if any(d.get('policy_queried') and (d['branch'] != 'cond' or not d['actor_mask']) for d in decisions):
        raise ValueError('forced/uncond call incorrectly counted as actor inference')
    if [c['call_index'] for c in p['calls']] != list(range(n)):
        raise ValueError('invalid predictor call ordering')
    if p['device_type'] not in ('cpu', 'cuda'):
        raise ValueError('unknown predictor timing device')
    for key in ('network_host_span_seconds', 'decision_wall_seconds', 'network_cuda_seconds'):
        if key == 'network_cuda_seconds' and p['device_type'] == 'cpu':
            if p[key] is not None or any(c[key] is not None for c in p['calls']):
                raise ValueError('CPU timing cannot report CUDA seconds')
            continue
        values = [p[key], *(c[key] for c in p['calls'])]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid predictor duration')
        if not math.isclose(p[key], sum(c[key] for c in p['calls']), abs_tol=1e-12, rel_tol=1e-9):
            raise ValueError('predictor duration sum mismatch')
    f = p['profile']['flops_per_call']
    if not isinstance(f, int) or f <= 0 or not math.isclose(p['tflops'], n*f/1e12, abs_tol=1e-18, rel_tol=1e-9):
        raise ValueError('predictor TFLOPs mismatch')
    generate = timing['pipeline_generate_wall_seconds']
    if not math.isfinite(generate) or generate <= 0:
        raise ValueError('invalid generate time')
    return dict(predictor_call_count=n, predictor_tflops=p['tflops'],
        predictor_network_cuda_seconds=p['network_cuda_seconds'],
        predictor_network_host_span_seconds=p['network_host_span_seconds'],
        predictor_decision_wall_seconds=p['decision_wall_seconds'],
        predictor_network_cuda_pct_of_generate=(100*p['network_cuda_seconds']/generate
            if p['network_cuda_seconds'] is not None else None),
        predictor_decision_wall_pct_of_generate=100*p['decision_wall_seconds']/generate)


def aggregate(rows):
    if not rows:
        raise ValueError('no overhead rows')
    result = dict(videos=len(rows))
    for key in ('predictor_call_count', 'predictor_tflops', 'predictor_network_cuda_seconds',
                'predictor_network_host_span_seconds', 'predictor_decision_wall_seconds'):
        values = [r[key] for r in rows]
        if any(v is None for v in values) and not all(v is None for v in values):
            raise ValueError('mixed CPU/CUDA timing cannot be aggregated')
        total = None if values[0] is None else sum(values)
        result[key + '_total'] = total
        result[key + '_mean_per_video'] = total / len(rows) if total is not None else None
    generate = sum(r['generate_seconds'] for r in rows)
    for kind in ('network_cuda', 'decision_wall'):
        seconds = result[f'predictor_{kind}_seconds_total']
        result[f'predictor_{kind}_pct_of_generate'] = 100*seconds/generate if seconds is not None else None
    result['scope'] = 'Predictor component share, not total cache-method overhead; nested in DiT/generate latency, not additive. Network TFLOPs exclude SEA filtering and other controller work.'
    return result
