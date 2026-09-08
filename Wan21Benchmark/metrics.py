"""Component-aware performance summaries and unchanged shared quality tools."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from protocol import ROOT, PACKAGES, external
from generation import PROTOCOL
sys.path.insert(0,str(PACKAGES/'ComponentMetrics'))
from reporting import extract_component_latency, extract_component_tflops


def flops_for_calls(calls, profile, method):
    f=profile['per_model_forward']['estimated_full_flops']
    a=profile['per_model_forward']['estimated_always_on_flops']
    layers=profile['input']['transformer_blocks']
    if layers!=30 or len(calls)!=100 or not 0<=a<=f:
        raise ValueError('invalid 30-block / 100-call Wan21 profile')
    total=0.
    full_seen=[0,0]
    for i,call in enumerate(calls):
        blocks=call['blocks_executed']
        allowed={30} if method=='baseline' else {1,30} if method=='dicache' else {0,30}
        if blocks not in allowed:
            raise ValueError('invalid method execution path')
        total += a+(f-a)*blocks/layers
        if method=='taylorseer':
            d=profile['input']['hidden_dim']
            n=profile['input']['seq_len']*d
            branch=i%2
            if blocks==layers:
                # Three module derivatives: subtraction and division per element.
                total += layers*6*n if full_seen[branch] else 0
                full_seen[branch] += 1
            else:
                # SA/FFN gating+add, cross add, modulation; three predictions.
                total += layers*((11 if full_seen[branch]>=2 else 5)*n+6*d)
    return total/1e12


def collect(directory,profile):
    manifest=json.loads((directory/'run.json').read_text())
    if manifest['protocol']!=PROTOCOL:
        raise ValueError('foreign/offload protocol is not accepted')
    complete=json.loads((directory/'COMPLETE.json').read_text())
    if complete['videos']!=len(manifest['prompts']):
        raise ValueError('incomplete generation')
    rows=[]
    for prompt in manifest['prompts']:
        path=directory/'timings'/f"{prompt['sample_id']}.json"
        timing=json.loads(path.read_text())
        expected='wan21' if manifest['method']=='baseline' else manifest['method']
        if timing['status']!='success' or timing['implementation']!=expected:
            raise ValueError('failed or wrong method timing')
        latency=timing['pipeline_generate_wall_seconds']
        if not math.isfinite(latency) or latency<=0:
            raise ValueError('invalid latency')
        rows.append(dict(sample_id=prompt['sample_id'],generate_seconds=latency,
                         dit_tflops=flops_for_calls(timing['calls'],profile,manifest['method']),
                         **extract_component_latency(timing),**extract_component_tflops(profile)))
    return manifest,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['summarize','evaluate'])
    p.add_argument('--baseline-dir',type=Path,required=True)
    p.add_argument('--candidate-dir',type=Path,required=True)
    p.add_argument('--profile',type=Path)
    args=p.parse_args()
    if Path(sys.prefix).name!="wan2.2":
        raise ValueError("use the wan2.2 conda environment")
    base=external(args.baseline_dir);cand=external(args.candidate_dir)
    bm=json.loads((base/'run.json').read_text());cm=json.loads((cand/'run.json').read_text())
    for key in ['protocol','prompts','checkpoint_dir','gpu']:
        if bm[key]!=cm[key]:
            raise ValueError('baseline/candidate mismatch: '+key)
    if bm['protocol']!=PROTOCOL or bm['method']!='baseline' or cm['method']=='baseline':
        raise ValueError('requires matching resident baseline and candidate')
    if not (base/'COMPLETE.json').is_file() or not (cand/'COMPLETE.json').is_file():
        raise ValueError('both generations must be complete')
    if args.action=='summarize':
        if args.profile is None:
            p.error('--profile is required')
        profile=json.loads(args.profile.read_text())
        if profile['input']['video_shape_fhw']!=[81,480,832]:
            raise ValueError('wrong FLOPs shape')
        _,br=collect(base,profile);_,cr=collect(cand,profile)
        payload=dict(protocol=PROTOCOL,method=cm['method'],baseline=br,candidate=cr,
                     latency_speedup_ratio_of_sums=sum(r['generate_seconds'] for r in br)/sum(r['generate_seconds'] for r in cr),
                     dit_tflops_speedup_ratio_of_sums=sum(r['dit_tflops'] for r in br)/sum(r['dit_tflops'] for r in cr),
                     scope='DiT Calflops plus dense attention; cache gate/DCTA excluded. Taylor factor updates and cached block arithmetic included analytically. T5/VAE separate.',
                     taylor_formula='N=seq_len*hidden_dim; each layer: full-update +6N, order0 reuse 5N+6D, order1 reuse 11N+6D; memory copies are not FLOPs')
        target=cand/'performance.json'
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(payload,indent=2)+'\n')
    else:
        output=cand/'evaluation'
        if output.exists():
            raise FileExistsError(output)
        env={**os.environ,'TORCH_HOME':str(ROOT/'models/torch-cache'),'PYTHON_BIN':sys.executable}
        subprocess.run([sys.executable,str(PACKAGES/'VideoMetrics/evaluate.py'),
                        '--reference-dir',str(base/'videos'),'--candidate-dir',str(cand/'videos'),
                        '--expected-frames','81','--device','cuda:0','--output-dir',str(output/'video_metrics')],env=env,check=True)
        for label,folder in [('reference',base),('candidate',cand)]:
            subprocess.run(['bash',str(PACKAGES/'VbenchEvaluation/run_vbench200.sh'),str(folder/'videos'),str(output/('vbench_'+label)),'1'],env=env,check=True)
        (output/'COMPLETE.json').write_text(json.dumps(dict(status='quality_complete',metrics=['psnr','ssim','lpips','vbench200']))+'\n')


if __name__=='__main__':
    main()
