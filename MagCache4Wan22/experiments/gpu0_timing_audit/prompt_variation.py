#!/usr/bin/env python3
"""Bounded follow-up: measure healthy-card per-prompt runtime dispersion."""
import argparse
import csv
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def describe(values):
    values=np.asarray(values,dtype=float)
    mean=float(values.mean())
    return dict(n=len(values),mean_seconds=mean,std_seconds=float(values.std(ddof=1)),
                cv_pct=float(100*values.std(ddof=1)/mean),
                p05_seconds=float(np.quantile(values,.05)),p95_seconds=float(np.quantile(values,.95)),
                min_seconds=float(values.min()),max_seconds=float(values.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir',type=Path,required=True)
    root=parser.parse_args().result_dir.resolve(strict=True)
    assert root.is_relative_to(Path('/all/yiran07-disk3/huteng_data/exp'))
    assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    source=root/'videos.csv'
    rows=list(csv.DictReader(source.open()))
    rows=[dict(r,gpu=int(r['gpu']),wall_seconds=float(r['wall_seconds']),full_calls=int(r['full_calls'])) for r in rows if int(r['gpu']) in (1,2,3)]
    assert len(rows)==1050
    aggregate=[]
    by_gpu=[]
    path_details=[]
    for cohort in dict.fromkeys(r['cohort'] for r in rows):
        rr=[r for r in rows if r['cohort']==cohort]
        assert len(rr)==150
        stats=describe([r['wall_seconds'] for r in rr])
        # An independent stdlib calculation checks the two headline statistics.
        assert abs(stats['mean_seconds']-statistics.mean(r['wall_seconds'] for r in rr))<1e-9
        assert abs(stats['std_seconds']-statistics.stdev(r['wall_seconds'] for r in rr))<1e-9
        pools=defaultdict(list)
        for r in rr:
            pools[(r['gpu'],r['schedule'])].append(r['wall_seconds'])
        normalized=[]
        for (gpu,schedule),values in pools.items():
            # Avoid singleton/small strata appearing spuriously stable. Each
            # focal sample is compared with the other same-card/path samples.
            if len(values)<5:
                continue
            total=sum(values)
            normalized.extend([100*(v/((total-v)/(len(values)-1))-1) for v in values])
            path_details.append(dict(cohort=cohort,gpu=gpu,schedule=schedule,**describe(values)))
        stats.update(cohort=cohort,label=rr[0]['label'],full_call_counts=dict(Counter(r['full_calls'] for r in rr)),
            controlled_n=len(normalized),controlled_excluded_n=150-len(normalized),
            same_gpu_path_loo_p05_pct=float(np.quantile(normalized,.05)),same_gpu_path_loo_p95_pct=float(np.quantile(normalized,.95)))
        aggregate.append(stats)
        for gpu in (1,2,3):
            values=[r['wall_seconds'] for r in rr if r['gpu']==gpu]
            assert len(values)==50
            by_gpu.append(dict(cohort=cohort,gpu=gpu,**describe(values)))
    payload=dict(status='complete',source={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
        script={'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        healthy_video_count=len(rows),cohorts=aggregate,per_gpu=by_gpu,
        metric='CUDA-synchronized complete generate wall time; excludes initialization, warmup, video export and evaluation',
        interpretation='Per-prompt observations include runtime noise; not a causal estimate of prompt-content cost. P05–P95 is the central 90% of observed samples, not a confidence interval.',
        path_control='Same physical GPU and full 100-call F/R sequence; only strata n>=5, each ratio uses the leave-one-out reference mean.')
    (root/'healthy_prompt_variation.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    for filename,data in [('healthy_prompt_variation_by_gpu.csv',by_gpu),('healthy_prompt_variation_by_path.csv',path_details)]:
        with (root/filename).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)
    print(json.dumps({'cohorts':aggregate,'baseline_per_gpu':[r for r in by_gpu if r['cohort']=='baseline']},ensure_ascii=False))


if __name__=='__main__':
    main()
