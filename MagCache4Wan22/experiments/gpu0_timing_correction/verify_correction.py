#!/usr/bin/env python3
"""Independently verify installed reports and per-video corrected records."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def near(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-9),(a,b)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--result-dir',type=Path,required=True)
    out=parser.parse_args().result_dir.resolve(strict=True)
    assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    policy=read(out/'policy.json');summary=read(out/'summary.json')
    assert read(out/'VALIDATION.json')['status']=='applied_and_verified'
    audit=Path(policy['raw_sources']).parent
    owner={(r['cohort'],r['sample_id']):int(r['gpu']) for r in csv.DictReader((audit/'videos.csv').open())}
    backups={r['path']:r for r in read(out/'backup_manifest.json')}
    for entry in backups.values():
        assert digest(Path(entry['backup']))==entry['before_sha256']
        assert digest(Path(entry['path']))==entry['after_sha256']
    unique={};healthy_count=0;gpu0_count=0
    for item in summary['rows']:
        root=Path(item['result_directory']);method=item['method'].replace('4Wan22','').lower()
        cohort=method+'_'+str(item['threshold']).replace('.','p')
        file=root/'performance/per_video.jsonl'
        original=[json.loads(s) for s in Path(backups[str(file)]['backup']).read_text().splitlines()]
        current=[json.loads(s) for s in file.read_text().splitlines()]
        assert len(original)==len(current)==400
        stats=read(root/'performance/summary.json');report=read(root/'benchmark_report.json')
        report_csv={r['condition']:r for r in csv.DictReader((root/'benchmark_report.csv').open())}
        for label in ['baseline',method]:
            key='baseline' if label=='baseline' else cohort
            old=[r for r in original if r['condition']==label];new=[r for r in current if r['condition']==label]
            peers=[r for r in old if owner[(key,r['sample_id'])]!=0]
            assert len(peers)==150
            ref={k:float(np.mean([r[k] for r in peers])) for k in policy['fields']}
            for a,b in zip(old,new):
                assert a['sample_id']==b['sample_id']
                gpu=owner[(key,a['sample_id'])]
                if gpu!=0:
                    assert a==b
                    healthy_count+=1
                else:
                    assert b['raw_timing']=={k:a[k] for k in policy['fields']}
                    for k in policy['fields']:near(b[k],ref[k])
                    restored=dict(b)
                    for k,v in restored.pop('raw_timing').items():restored[k]=v
                    restored['estimated_achieved_dit_tflops_per_second']=restored.pop('raw_estimated_achieved_dit_tflops_per_second')
                    restored.pop('timing_correction')
                    assert restored==a
                    gpu0_count+=1
                identity=(key,b['sample_id'])
                if identity in unique:assert unique[identity]==b
                unique[identity]=b
            for field in ['pipeline_generate_wall_seconds','model_forward_cuda_seconds','t5_cuda_seconds','vae_decode_cuda_seconds']:
                values=np.array([r[field] for r in new]);v=stats['conditions'][label][field]
                for k,x in [('mean',values.mean()),('total',values.sum()),('std_population',values.std()),('min',values.min()),('max',values.max()),('p50',np.quantile(values,.5)),('p90',np.quantile(values,.9))]:near(v[k],x)
            near(report['performance'][label]['inference_time_seconds_mean'],ref['pipeline_generate_wall_seconds'])
            near(report_csv[label]['inference_time_seconds_mean'],ref['pipeline_generate_wall_seconds'])
        independent=sum(r['pipeline_generate_wall_seconds'] for r in current if r['condition']=='baseline')/sum(r['pipeline_generate_wall_seconds'] for r in current if r['condition']==method)
        near(item['speedup'],independent);near(report['performance']['comparison']['latency_speedup_ratio_of_sums'],independent)
        assert f'{independent:.6f}x' in (root/'benchmark_report.md').read_text()
        assert report['source_files']['performance']==str(root/'performance/summary.json')
    assert (len(unique),gpu0_count,healthy_count)==(1400,600,1800)
    for entry in read(out/'links.json'):
        link=Path(entry['path']);assert link.is_symlink() and link.resolve()==Path(entry['target'])
    protected=read(out/'protected_sources.json')
    for path,value in protected.items():assert digest(Path(path))==value
    # The prior audit is historical: changed derived inputs are recovered via backups.
    historical_sources=read(audit/'source_manifest.json')['sources']
    historical_via_backup=0
    for source in historical_sources:
        path=Path(source['path'])
        if str(path) in backups:path=Path(backups[str(path)]['backup']);historical_via_backup+=1
        assert digest(path)==source['sha256'],path
    reuse_path=Path('/all/yiran07-disk3/huteng_data/exp/magcache4wan22_baseline_reuse_audit_20260906/baseline_reuse_manifest.json')
    reuse=read(reuse_path)
    for row in reuse['rows']:
        effective=unique[('baseline',row['sample_id'])]
        near(row['generate_seconds'],effective['pipeline_generate_wall_seconds'])
        raw=read(Path(row['timing']['path']))
        near(row['raw_generate_seconds'],raw['pipeline_generate_wall_seconds'])
        assert digest(Path(row['timing']['path']))==row['timing']['sha256']
    subset_ids={r['sample_id'] for r in csv.DictReader((audit/'same_prompt_baselines.csv').open())}
    subset_mean=float(np.mean([r['generate_seconds'] for r in reuse['rows'] if r['sample_id'] in subset_ids]))
    result=dict(status='passed',formal_conditions=6,unique_videos=1400,unique_gpu0_corrected=350,
        formal_gpu0_reporting_rows=gpu0_count,healthy_reporting_rows_exactly_unchanged=healthy_count,
        all_component_summary_statistics_match_independent_numpy=True,
        json_csv_markdown_speedups_match=True,raw_call_flops_fields_restored_exactly=True,
        protected_source_hashes_unchanged=len(protected),historical_audit_sources_verified=len(historical_sources),
        historical_derived_sources_verified_via_backup=historical_via_backup,
        baseline_reuse_index_rows_verified=200,calibration11_corrected_baseline_mean=subset_mean)
    (out/'INDEPENDENT_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
