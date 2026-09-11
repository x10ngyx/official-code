"""Independent arithmetic audit of the completed 40 trajectories and matched comparisons."""
import csv
import json
import math
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_experiment as exp

def close(a,b,label):
    if not math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-8):raise ValueError(f'{label}: {a} != {b}')

def main():
    root=exp.ROOT;cfg=exp.read(root/'config.json');done=exp.read(root/'COMPLETE.json')
    exp.validate_sources(cfg)
    if done['results_sha256']!=exp.sha(root/'analysis/results.csv'):raise ValueError('final table SHA mismatch')
    for p,h in exp.read(root/'support_sources.json').items():
        if exp.sha(Path(p))!=h:raise ValueError(f'support source changed: {p}')
    profile=exp.read(Path(cfg['flops_profile']));full=profile['per_model_forward']['estimated_full_flops'];always=profile['per_model_forward']['estimated_always_on_flops']
    results=exp.pair.read_csv(root/'analysis/results.csv');detail=exp.pair.read_csv(root/'analysis/per_video.csv')
    if len(results)!=10 or len(detail)!=40:raise ValueError('coverage')
    checked=0
    for result in results:
        i=int(result['strategy'])-1;phase=result['phase'];latency=[];flops=[]
        for g in range(4):
            r,tr=exp.condition(cfg,phase,i,g);sid=r['sample_id'];d=root/'conditions'/f'{phase}_{i+1}'/f'gpu{g}'
            t=exp.read(d/'timings'/f'{sid}.json');n=sum(c['blocks_executed']==30 for c in t['calls'])
            expected=(n*full+(100-n)*always)/1e12
            close(r['dit_tflops'],expected,'DiT TFLOPs')
            close(r['dit_cuda_seconds'],sum(c['cuda_seconds'] for c in t['calls']),'DiT time')
            for k,prefix in [('estimated_t5_tflops_per_video','t5'),('estimated_vae_decode_tflops_per_video','vae_decode')]:close(r[k],profile['component_profiles'][prefix]['estimated_tflops_per_video'],k)
            for prefix in ('t5','vae_decode'):close(r[prefix+'_cuda_seconds'],t['component_latency'][prefix]['cuda_seconds'],prefix+' time')
            if sum(r[k+'_cuda_seconds'] for k in ('t5','dit','vae_decode'))>r['generate_seconds']+.01:raise ValueError('component span exceeds inference')
            latency.append(r['generate_seconds']);flops.append(expected);checked+=100
        close(result['latency_speedup'],sum(r['generate_seconds'] for r in cfg['baselines'].values())/sum(latency),'aggregate latency ratio')
        close(result['dit_tflops'],st.fmean(flops),'aggregate flops')
        rows=[r for r in detail if r['strategy']==str(i+1) and r['phase']==phase]
        for m in exp.METRICS:close(result[m],st.fmean(float(r[m]) for r in rows),m+' mean')
    matching=exp.read(root/'matching.json')
    for i,row in enumerate(matching['pairs']):
        threshold,_=exp.pair.interpolate_threshold(exp.pair.read_csv(Path(cfg['calibration_csv'])),row['increase_speedup'])
        close(row['threshold'],threshold,'fixed threshold calibration')
    exp.dump(root/'analysis/INDEPENDENT_AUDIT.json',dict(status='pass',formal_trajectories=40,branch_calls=checked,checks=['input and support source SHA','same physical GPU and prompt identity','per-step threshold and actual blocks','closed-form actual DiT TFLOPs','component time and FLOPs','ratio of summed latency','quality means','calibration interpolation'],limitations=['four prompts and one seed','historical baseline latency may drift','fixed thresholds are calibration-matched; actual mismatch remains visible','VBench custom raw mean is not the standard official score','DiT FLOPs exclude SEA gate/filter overhead, matching existing calibration accounting']))
    print(json.dumps(dict(status='pass',trajectories=40,branch_calls=checked)))
if __name__=='__main__':main()
