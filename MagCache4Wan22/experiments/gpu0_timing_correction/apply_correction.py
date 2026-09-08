#!/usr/bin/env python3
"""Apply cohort-specific healthy mean substitution, preserving measured sources."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import sqlite3
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT=Path(__file__).resolve().parents[2]
REPO=PROJECT.parent
EXP=Path('/all/yiran07-disk3/huteng_data/exp')
PYTHON=Path('/home/huteng/yes/envs/wan2.2/bin/python')
TIME_FIELDS=('pipeline_generate_wall_seconds','model_forward_cuda_seconds','dit_cuda_seconds',
             't5_cuda_seconds','t5_host_span_seconds','vae_decode_cuda_seconds','vae_decode_host_span_seconds')
POLICY='gpu0_direct_same_condition_healthy_gpu123_mean_v1'


def encode(value):
    return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(encode(value))


def read_json(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def csv_bytes(rows):
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode()


def atomic_write(path,data):
    temp=path.with_name('.'+path.name+'.gpu0_correction_tmp')
    assert not temp.exists()
    temp.write_bytes(data)
    os.replace(temp,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    assert Path(sys.prefix).name=='wan2.2'
    assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    audit=args.audit_dir.resolve(strict=True);out=args.output_dir.resolve()
    assert out.is_relative_to(EXP) and out!=EXP and not out.exists()
    source_manifest=read_json(audit/'source_manifest.json')
    for source in source_manifest['sources']:
        assert sha(Path(source['path']).read_bytes())==source['sha256'],source['path']
    lookup={(r['cohort'],r['sample_id']):r for r in csv.DictReader((audit/'videos.csv').open())}
    assert len(lookup)==1400
    out.mkdir()
    now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    note=('Timing correction applied: each physical GPU0 video uses the arithmetic mean of the 150 '
          'measured GPU1/2/3 videos in the same method/threshold condition. Complete generate and '
          'component times are corrected estimates, not new measurements. GPUs1/2/3, TFLOPs, '
          'call counts, videos and quality scores are unchanged. Original per-video timing JSON '
          'remains raw; corrected performance rows retain raw_timing. Corrected quantiles and '
          'variance include mean imputation and do not represent measured runtime variability.')
    meta=dict(policy=POLICY,status='applied' if args.apply else 'staged',created_at=now,
        user_instruction='直接用健康卡的平均耗时修正GPU0上所有任务的耗时；修正后重新报告TeaCache、SeaCache加速比',
        scope='One shared 200-video baseline and six 200-video formal Wan2.2 candidate conditions',
        physical_gpu=0,reference_gpus=[1,2,3],reference_videos_per_condition=150,
        corrected_unique_videos=350,healthy_unique_videos_unchanged=1050,
        fields=list(TIME_FIELDS),matching='same method and threshold; no action-path matching',
        note=note,package=str(out),raw_sources=str(audit/'source_manifest.json'))
    write_json(out/'policy.json',meta)
    changes={};links={};before={};protected={};cohort_rows={};means={};summary_rows=[]

    def plan(path,data):
        path=path.resolve()
        assert path.is_relative_to(EXP)
        assert path not in changes
        assert path.is_file(),path
        before[path]=path.read_bytes();changes[path]=data

    def new_link(path,target):
        assert not path.exists() and not path.is_symlink(),path
        links[path]=target

    def pin(path):
        path=path.resolve(strict=True)
        protected[str(path)]=sha(path.read_bytes())

    spec=importlib.util.spec_from_file_location('sea_performance',REPO/'SeaCache4Wan22/experiments/vbench200_t2v/aggregate_performance.py')
    agg=importlib.util.module_from_spec(spec);spec.loader.exec_module(agg)
    source_roots=[];stage_roots=[]
    for cohort,definition in source_manifest['cohorts'].items():
        if cohort=='baseline':continue
        condition=cohort.split('_')[0]
        source=Path(definition['directory']).parent
        assert read_json(source/'status.json')['status']=='complete'
        original_rows=[json.loads(line) for line in (source/'performance/per_video.jsonl').read_text().splitlines()]
        assert len(original_rows)==400 and len({(r['condition'],r['sample_id']) for r in original_rows})==400
        original_summary=read_json(source/'performance/summary.json')
        updated=[]
        for label in ['baseline',condition]:
            name='baseline' if label=='baseline' else cohort
            rows=[r for r in original_rows if r['condition']==label]
            assert len(rows)==200
            for r in rows:
                audit_row=lookup[(name,r['sample_id'])]
                assert math.isclose(r['pipeline_generate_wall_seconds'],float(audit_row['wall_seconds']),rel_tol=1e-12)
                pin(Path(r['timing_path']))
                if r.get('trace_path'):pin(Path(r['trace_path']))
            healthy=[r for r in rows if int(lookup[(name,r['sample_id'])]['gpu'])!=0]
            assert len(healthy)==150
            avg={k:statistics.mean(r[k] for r in healthy) for k in TIME_FIELDS}
            if name in means:assert means[name]==avg
            means[name]=avg
            corrected=[]
            for original in rows:
                r=copy.deepcopy(original)
                if int(lookup[(name,r['sample_id'])]['gpu'])==0:
                    assert 'raw_timing' not in r
                    r['raw_timing']={k:r[k] for k in TIME_FIELDS}
                    r['raw_estimated_achieved_dit_tflops_per_second']=r['estimated_achieved_dit_tflops_per_second']
                    r.update(avg)
                    r['estimated_achieved_dit_tflops_per_second']=r['estimated_dit_tflops']/r['model_forward_cuda_seconds']
                    r['timing_correction']={'policy':POLICY,'physical_gpu':0,'cohort':name,'reference_count':150,'policy_path':str(out/'policy.json')}
                else:assert r==original
                corrected.append(r)
            if name in cohort_rows:assert cohort_rows[name]==corrected
            cohort_rows[name]=corrected;updated.extend(corrected)
        corrected_summary=copy.deepcopy(original_summary)
        corrected_summary['conditions']={label:agg.summarize(label,[r for r in updated if r['condition']==label]) for label in ['baseline',condition]}
        b,c=corrected_summary['conditions'].values()
        for label,new in corrected_summary['conditions'].items():
            old=original_summary['conditions'][label]
            for k in old:
                if 'flops' in k and 'per_second' not in k or 'calls' in k or k=='video_count':assert new[k]==old[k],(cohort,label,k)
        total_b=b['pipeline_generate_wall_seconds']['total'];total_c=c['pipeline_generate_wall_seconds']['total']
        corrected_summary['comparison']['latency_speedup_ratio_of_sums']=total_b/total_c
        corrected_summary['comparison']['latency_reduction_fraction']=1-total_c/total_b
        corrected_summary['timing_correction']=meta
        corrected_summary['raw_conditions']=original_summary['conditions']
        corrected_summary['raw_comparison']=original_summary['comparison']
        corrected_summary['latency_definition']['timing_basis']=POLICY
        corrected_summary['latency_definition']['correction_note']=note
        corrected_summary['warnings'].append(note)
        stage=out/'staged'/cohort;stage.mkdir(parents=True)
        (stage/'performance').mkdir()
        write_json(stage/'performance/summary.json',corrected_summary)
        data=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in updated).encode()
        (stage/'performance/per_video.jsonl').write_bytes(data)
        for item in ['run_config.json','evaluation','baseline','status.json']:
            (stage/item).symlink_to(source/item,target_is_directory=(source/item).is_dir())
        builder=REPO/(condition.title().replace('cache','Cache')+'4Wan22')/'experiments/vbench200_t2v/build_final_report.py'
        # Use the existing method-specific report renderer and unchanged quality sources.
        result=subprocess.run([str(PYTHON),str(builder),'--result-dir',str(stage)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        old_report=read_json(source/'benchmark_report.json')
        report=read_json(stage/'benchmark_report.json')
        for key in old_report:
            if key not in ('performance','source_files'):assert old_report[key]==report[key],(cohort,key)
        for path in old_report['source_files'].values():
            if Path(path)!=source/'performance/summary.json':pin(Path(path))
        report['source_files']=old_report['source_files']
        report['raw_performance']=old_report['performance']
        report['timing_correction']=meta
        write_json(stage/'benchmark_report.json',report)
        markdown=(stage/'benchmark_report.md').read_text()
        title,body=markdown.split('\n',1)
        (stage/'benchmark_report.md').write_text(title+'\n\n'+note+'\n\nCorrection package: `'+str(out)+'`.\n'+body)
        csv_rows=list(csv.DictReader((stage/'benchmark_report.csv').open()))
        for row in csv_rows:
            row['timing_basis']=POLICY
            row['raw_inference_time_seconds_mean']=old_report['performance'][row['condition']]['inference_time_seconds_mean']
        (stage/'benchmark_report.csv').write_bytes(csv_bytes(csv_rows))
        for item in ['performance/summary.json','performance/per_video.jsonl','benchmark_report.json','benchmark_report.md','benchmark_report.csv']:
            plan(source/item,(stage/item).read_bytes())
        for item in ['README.md','performance/README.md']:
            plan(source/item,(source/item).read_bytes()+('\n\n'+note+'\nCorrection package: `'+str(out)+'`.\n').encode())
        status=read_json(source/'status.json');status['phases']['timing_correction']=dict(status='complete',policy=POLICY,package=str(out),completed_at=now)
        plan(source/'status.json',encode(status))
        summary_rows.append(dict(method=report['method'],threshold=report['threshold'],baseline_seconds=b['pipeline_generate_wall_seconds']['mean'],candidate_seconds=c['pipeline_generate_wall_seconds']['mean'],speedup=total_b/total_c,raw_speedup=original_summary['comparison']['latency_speedup_ratio_of_sums'],candidate_dit_tflops=c['estimated_dit_tflops_per_video']['mean'],result_directory=str(source)))
        source_roots.append(source);stage_roots.append(stage)

    # One effective 200-row table per condition, including a canonical shared baseline.
    for cohort,rows in cohort_rows.items():
        target=out/'effective_timings'/f'{cohort}.jsonl';target.parent.mkdir(exist_ok=True)
        target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        cm=dict(meta,cohort=cohort,healthy_means=means[cohort],effective_timings=str(target))
        sidecar=out/'effective_timings'/f'{cohort}.correction.json';write_json(sidecar,cm)
        directory=Path(source_manifest['cohorts'][cohort]['directory'])
        new_link(directory/'reported_timings.jsonl',target)
        new_link(directory/'timing_correction.json',sidecar)

    # Rebuild the TeaCache aggregate through its existing renderer.
    tea_sources=source_roots[3:];tea_stages=stage_roots[3:]
    tea_suite=tea_sources[0].parent;tea_stage=out/'staged/teacache_suite';tea_stage.mkdir()
    command=[str(PYTHON),str(REPO/'TeaCache4Wan22/experiments/vbench200_t2v/build_suite_report.py'),'--suite-root',str(tea_stage)]
    for stage in tea_stages:command+=['--result-dir',str(stage)]
    result=subprocess.run(command,capture_output=True,text=True);assert result.returncode==0,result.stderr
    suite=read_json(tea_stage/'suite_report.json');old_suite=read_json(tea_suite/'suite_report.json')
    suite['timing_correction']=meta;suite['raw_baseline_performance']=old_suite['baseline_performance'];suite['raw_rows']=old_suite['rows']
    for row,source in zip(suite['rows'],tea_sources):row['result_dir']=str(source)
    write_json(tea_stage/'suite_report.json',suite)
    csv_rows=list(csv.DictReader((tea_stage/'suite_report.csv').open()))
    for row,source in zip(csv_rows,tea_sources):row['result_dir']=str(source);row['timing_basis']=POLICY
    (tea_stage/'suite_report.csv').write_bytes(csv_bytes(csv_rows))
    (tea_stage/'suite_report.md').write_text((tea_stage/'suite_report.md').read_text()+'\n'+note+'\n')
    for item in ['suite_report.json','suite_report.csv','suite_report.md']:plan(tea_suite/item,(tea_stage/item).read_bytes())
    for suite_root in [source_roots[0].parent,tea_suite]:
        plan(suite_root/'README.md',(suite_root/'README.md').read_bytes()+('\n\n'+note+'\nCorrection package: `'+str(out)+'`.\n').encode())

    # The validated video reuse index now exposes corrected latency explicitly.
    reuse_path=EXP/'magcache4wan22_baseline_reuse_audit_20260906/baseline_reuse_manifest.json'
    reuse=read_json(reuse_path);baseline={r['sample_id']:r for r in cohort_rows['baseline']}
    for row in reuse['rows']:
        r=baseline[row['sample_id']]
        row['raw_generate_seconds']=row['generate_seconds']
        row['generate_seconds']=r['pipeline_generate_wall_seconds']
        row['timing_basis']=POLICY if 'timing_correction' in r else 'measured_gpu123'
    reuse['raw_timing_policy']=reuse['timing_policy']
    reuse['timing_policy']={'policy':POLICY,'reported_timings':str(out/'effective_timings/baseline.jsonl'),'raw_timing_files':'Unchanged; timing.path and timing.sha256 still describe raw instrumentation','note':note}
    reuse['timing_correction']=meta
    plan(reuse_path,encode(reuse))
    plan(reuse_path.parent/'README.md',(reuse_path.parent/'README.md').read_bytes()+('\n\nTiming correction is now applied in baseline_reuse_manifest.json: GPU0 generate_seconds uses the healthy mean, original raw_generate_seconds and raw timing file hashes remain available. See `'+str(out)+'`.\n').encode())

    # Independent SQL verifies imputation, all final means and speedups.
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE effective (cohort TEXT,gpu INTEGER,raw REAL,effective REAL)')
    for cohort,rows in cohort_rows.items():
        for r in rows:db.execute('INSERT INTO effective VALUES (?,?,?,?)',(cohort,int(lookup[(cohort,r['sample_id'])]['gpu']),float(lookup[(cohort,r['sample_id'])]['wall_seconds']),r['pipeline_generate_wall_seconds']))
    sql_rows=db.execute('SELECT cohort, AVG(effective), AVG(CASE WHEN gpu!=0 THEN raw END), SUM(CASE WHEN gpu=0 THEN 1 ELSE 0 END), SUM(CASE WHEN gpu!=0 AND raw!=effective THEN 1 ELSE 0 END) FROM effective GROUP BY cohort').fetchall()
    for cohort,avg,healthy,n,changed in sql_rows:
        assert n==50 and changed==0 and math.isclose(avg,healthy,abs_tol=1e-9)
        assert math.isclose(avg,means[cohort]['pipeline_generate_wall_seconds'],abs_tol=1e-9)
    for row in summary_rows:
        cohort=('seacache' if row['method']=='SeaCache4Wan22' else 'teacache')+'_'+str(row['threshold']).replace('.','p')
        assert math.isclose(row['speedup'],means['baseline']['pipeline_generate_wall_seconds']/means[cohort]['pipeline_generate_wall_seconds'],abs_tol=1e-12)
    assert sum('timing_correction' in r for rows in cohort_rows.values() for r in rows)==350
    write_json(out/'summary.json',dict(policy=meta,conditions=means,rows=summary_rows))
    (out/'comparison.csv').write_bytes(csv_bytes(summary_rows))
    write_json(out/'protected_sources.json',protected)
    backup_manifest=[]
    for path,data in changes.items():
        backup=out/'backups'/path.relative_to(EXP);backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(before[path])
        assert sha(backup.read_bytes())==sha(before[path])
        backup_manifest.append(dict(path=str(path),backup=str(backup),before_sha256=sha(before[path]),after_sha256=sha(data)))
    write_json(out/'backup_manifest.json',backup_manifest)
    write_json(out/'links.json',[dict(path=str(p),target=str(t)) for p,t in links.items()])
    write_json(out/'source_code.json',[dict(path=str(p),sha256=sha(p.read_bytes())) for p in [Path(__file__),Path(__file__).with_name('README.md')]])
    validation=dict(status='staged',unique_videos=1400,unique_gpu0_corrected=350,unique_healthy_unchanged=1050,
        formal_performance_rows=2400,formal_gpu0_rows_corrected=600,formal_reports=6,shared_baseline_identical_across_reports=True,
        healthy_mean_sql_verification=True,flops_and_call_counts_unchanged=True,quality_report_fields_unchanged=True,
        protected_raw_timing_trace_and_quality_files=len(protected),backed_up_files=len(changes),installed_links=len(links))
    write_json(out/'VALIDATION.json',validation)
    installed=[];written=[]
    if args.apply:
        try:
            for path in changes:assert path.read_bytes()==before[path],path
            for path,data in changes.items():atomic_write(path,data);written.append(path)
            for path,target in links.items():path.symlink_to(target);installed.append(path)
            for source,digest in protected.items():assert sha(Path(source).read_bytes())==digest,source
            for path,data in changes.items():assert path.read_bytes()==data,path
            for source,stage in zip(source_roots,stage_roots):
                report=read_json(source/'benchmark_report.json');perf=read_json(source/'performance/summary.json')
                assert report['performance']['comparison']==perf['comparison']
                assert report['timing_correction']['policy']==POLICY
            validation['status']='applied_and_verified';write_json(out/'VALIDATION.json',validation)
        except BaseException:
            for path in reversed(installed):path.unlink()
            for path in reversed(written):atomic_write(path,before[path])
            validation['status']='rolled_back';write_json(out/'VALIDATION.json',validation)
            raise
    table='\n'.join(f"| {r['method']} | {r['threshold']:.2f} | {r['candidate_seconds']:.6f} | {r['raw_speedup']:.6f} | {r['speedup']:.6f} |" for r in summary_rows)
    (out/'REPORT.md').write_text('# Wan2.2 GPU0 timing correction\n\n'+note+'\n\nStatus: '+validation['status']+'. Shared baseline mean: '+f"{means['baseline']['pipeline_generate_wall_seconds']:.6f}"+' seconds/video.\n\n| Method | Threshold | Corrected candidate seconds | Original speedup | Corrected speedup |\n| --- | ---: | ---: | ---: | ---: |\n'+table+'\n\nThe seven effective per-video datasets contain 1400 unique videos, 350 corrected GPU0 rows and 1050 unchanged healthy rows. Six formal result directories repeat the shared baseline; their 2400 reporting rows therefore contain 600 GPU0 substitutions. The 1400 original per-video timing files and all referenced cache traces and quality summaries remain unchanged.\n\nThis correction uses one arithmetic mean per condition, including SeaCache conditions whose full-call counts vary by prompt, as explicitly requested. It does not use the earlier path-matched multiplicative preview. MagCache GPU1/2/3 scan measurements are unaffected.\n\n`backup_manifest.json` locates every original derived file; `effective_timings/` provides reusable corrected rows; `VALIDATION.json` records application and verification.\n')
    (out/'README.md').write_text('# GPU0 healthy-mean timing correction\n\nREPORT.md / comparison.csv / summary.json contain the corrected formal results. policy.json records the authorized method and scope. effective_timings/ has seven canonical corrected tables with raw timing retained on modified rows. backups/ and backup_manifest.json retain every replaced reporting artifact. staged/ holds method-rendered report outputs. protected_sources.json and VALIDATION.json verify unchanged raw timing, trace and quality inputs.\n')
    for project in [PROJECT,REPO/'SeaCache4Wan22',REPO/'TeaCache4Wan22']:
        link=project/'experiment_results'/out.name
        assert not link.exists() and not link.is_symlink();link.symlink_to(out,target_is_directory=True)
    print(json.dumps({'status':validation['status'],'output':str(out),'rows':summary_rows,'validation':validation},ensure_ascii=False))


if __name__=='__main__':main()
