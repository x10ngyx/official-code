"""Pure reporting normalization; no inference or model-source changes."""
import copy
import hashlib
import json
import math
from pathlib import Path

POLICY='magcache_shared_corrected_baseline_same_prompt_v1'
FIELD_MAP={'generate_seconds':'pipeline_generate_wall_seconds',
           't5_cuda_seconds':'t5_cuda_seconds','t5_host_span_seconds':'t5_host_span_seconds',
           'vae_decode_cuda_seconds':'vae_decode_cuda_seconds','vae_decode_host_span_seconds':'vae_decode_host_span_seconds',
           'dit_cuda_seconds':'dit_cuda_seconds'}


def load_reference(meta):
    assert meta['policy']==POLICY
    for source in meta['sources'].values():
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256'],source['path']
    refs={r['sample_id']:r for r in map(json.loads,Path(meta['sources']['reported_timings']['path']).read_text().splitlines())}
    reuse=json.loads(Path(meta['sources']['reuse_index']['path']).read_text())
    identities={r['sample_id']:r for r in reuse['rows']}
    assert len(refs)==len(identities)==200 and set(refs)==set(identities)
    assert reuse['timing_correction']['policy']=='gpu0_direct_same_condition_healthy_gpu123_mean_v1'
    for sid,row in refs.items():
        assert row['condition']=='baseline' and row['full_compute_forward_calls']==100
        assert identities[sid]['generate_seconds']==row['pipeline_generate_wall_seconds']
    return refs,identities


def normalize(raw,refs,meta):
    assert 'raw_same_gpu_performance' not in raw and 'timing_correction' not in raw
    result=copy.deepcopy(raw)
    for pair in result['per_video']:
        b=pair['baseline'];source=refs[b['sample_id']]
        for target,key in FIELD_MAP.items():b[target]=source[key]
        for key,field in [('full_calls','full_compute_forward_calls'),('reuse_calls','reuse_forward_calls'),('estimated_dit_tflops','estimated_dit_tflops'),('estimated_t5_tflops_per_video','estimated_t5_tflops_per_video'),('estimated_vae_decode_tflops_per_video','estimated_vae_decode_tflops_per_video')]:
            assert math.isclose(b[key],source[field],rel_tol=1e-10,abs_tol=1e-10),(key,b[key],source[field])
    result['sums']['baseline']={k:sum(p['baseline'][k] for p in result['per_video']) for k in raw['sums']['baseline']}
    result['speedup']=result['sums']['baseline']['generate_seconds']/result['sums']['magcache']['generate_seconds']
    assert result['sums']['magcache']==raw['sums']['magcache']
    assert result['dit_flops_speedup']==raw['dit_flops_speedup']
    result['raw_same_gpu_performance']=copy.deepcopy(raw)
    result['timing_correction']=meta
    return result
