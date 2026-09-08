#!/usr/bin/env python3
"""Update a completed MagCache scan to the corrected, prompt-matched shared baseline."""
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT=Path(__file__).resolve().parents[2]
EXP=Path('/all/yiran07-disk3/huteng_data/exp')
sys.path.insert(0,str(PROJECT/'runtime'))
from scan import select_preset_targets
from normalization import POLICY, FIELD_MAP, load_reference, normalize


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def save(path,value):path.write_bytes(encoded(value))
def record(path):return dict(path=str(path.resolve()),sha256=sha(path))
def csv_data(rows):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode()
def atomic(path,data):
    temp=path.with_name('.'+path.name+'.shared_baseline_tmp');assert not temp.exists();temp.write_bytes(data);os.replace(temp,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['scan-dir','correction-dir','reuse-index','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    assert Path(sys.prefix).name=='wan2.2'
    assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    root=args.scan_dir.resolve(strict=True);correction=args.correction_dir.resolve(strict=True)
    reuse_path=args.reuse_index.resolve(strict=True);out=args.output_dir.resolve()
    assert all(p.is_relative_to(EXP) for p in [root,correction,reuse_path,out]) and not out.exists()
    assert read(correction/'INDEPENDENT_VALIDATION.json')['status']=='passed'
    old_audit=read(root/'final_validation.json');assert old_audit['status']=='pass'
    for source,digest in old_audit['artifact_sha256'].items():assert sha(Path(source))==digest,source
    groups=sorted(p for p in root.glob('target_*_gpu*') if p.is_dir());assert len(groups)==3
    source=read(root/'shared_profile/calflops.json')['source']['prepared_manifest']
    for name,digest in source['package_file_sha256'].items():assert sha(PROJECT/name)==digest,name
    prompt_sets=[{p['sample_id'] for p in read(g/'plan.json')['prompts']} for g in groups]
    assert all(s==prompt_sets[0] for s in prompt_sets) and len(prompt_sets[0])==11
    out.mkdir()
    now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    meta=dict(policy=POLICY,status='applied',created_at=now,package=str(out),scan_directory=str(root),
        baseline_prompt_count=11,baseline_policy='Shared corrected baseline restricted to the same 11 sample IDs; raw GPU0 baseline imputed using 150 healthy GPU1/2/3 means; healthy baseline values unchanged',
        candidate_policy='All GPU1/2/3 candidate timings, calls, FLOPs and video/quality artifacts unchanged',
        sources={'reported_timings':record(correction/'effective_timings/baseline.jsonl'),
                 'reuse_index':record(reuse_path),'healthy_mean_policy':record(correction/'policy.json')})
    refs,identities=load_reference(meta)
    meta['baseline_mean_seconds']=statistics.mean(refs[sid]['pipeline_generate_wall_seconds'] for sid in prompt_sets[0])
    save(out/'policy.json',meta)
    notice=f"报告已统一使用共享baseline中相同11条prompt的修正计时，均值{meta['baseline_mean_seconds']:.6f}秒/视频；候选GPU1/2/3实测计时不变。当前速度为修正基准估算，原同卡结果保存在raw_same_gpu_performance和修正包备份中。修正包：{out}。"
    affected=[]
    for g in groups:
        affected += [g/n for n in ['performance.json','target_selection.json','GENERATION_COMPLETE.json','report.json','COMPLETE.json','README.md','status.json']]
    affected += [root/n for n in ['COMPLETE.json','status.json','README.md','scan_summary.json','final_validation.json','scan_candidates.csv','RESULTS.md']]
    affected += [groups[0]/n for n in ['scan_1p8_plateaus.csv','scan_1p8_E_grid.csv','scan_1p8_per_prompt.csv','SCAN_READOUT.md','scan_1p8_validation.json']]
    before={p:p.read_bytes() for p in affected}
    backups={}
    for path,data in before.items():
        dest=out/'backups'/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);backups[path]=dest
    protected=dict(old_audit['artifact_sha256'])
    for p in affected:protected.pop(str(p),None)
    video_sources={};new_perfs={};changes={};selections={};old_selections={}
    for g in groups:
        plan=read(g/'plan.json');assert plan['gpu_ids'] in (['1'],['2'],['3'])
        raw=read(g/'performance.json');current={}
        for c in plan['conditions']:
            for prompt in plan['prompts']:
                path=g/'runs'/c['id']/prompt['sample_id']/'manifest.json';manifest=read(path)
                protected[str(path)]=sha(path)
                for key in ['video','run','timing','trace']:
                    if manifest.get(key):
                        item=manifest[key];assert sha(Path(item['path']))==item['sha256']
                        protected[item['path']]=item['sha256']
                        if key=='video':video_sources[item['path']]=item['sha256']
                if c['id']=='baseline':
                    identity=identities[prompt['sample_id']]
                    assert manifest['video']['sha256']==identity['video']['sha256']
                    assert manifest['prompt']==identity['prompt_en'] and manifest['protocol']['seed']==identity['seed']
        for cid,value in raw.items():
            assert len(value['per_video'])==11
            assert {p['baseline']['sample_id'] for p in value['per_video']}==prompt_sets[0]
            current[cid]=normalize(value,refs,meta)
        selected=select_preset_targets(current,plan['target_groups'])
        assert selected['targets'][0]['status']=='hit'
        report=read(g/'report.json');assert report['performance']==raw
        old_selections[g.name]=copy.deepcopy(report['target_selection'])
        report['performance']=current;report['target_selection']=selected;report['timing_correction']=meta
        report['raw_same_gpu_target_selection']=old_selections[g.name]
        changes[g/'performance.json']=encoded(current)
        changes[g/'target_selection.json']=encoded(selected)
        changes[g/'report.json']=encoded(report)
        for name,key in [('GENERATION_COMPLETE.json','performance'),('COMPLETE.json','report')]:
            marker=read(g/name);target=g/('performance.json' if key=='performance' else 'report.json')
            marker['raw_'+key]=record(backups[target])
            marker[key]=dict(path=str(target),sha256=hashlib.sha256(changes[target]).hexdigest())
            marker['timing_correction']=meta;changes[g/name]=encoded(marker)
        status=read(g/'status.json');status['timing_correction']=meta;changes[g/'status.json']=encoded(status)
        changes[g/'README.md']=before[g/'README.md']+('\n\n'+notice+'\n').encode()
        selections[g.name]=selected;new_perfs[g.name]=current
    complete=read(root/'COMPLETE.json');complete['raw_same_gpu_targets']=complete['targets'];complete['targets']=selections;complete['timing_correction']=meta
    changes[root/'COMPLETE.json']=encoded(complete)
    status=read(root/'status.json');status['timing_correction']=meta;changes[root/'status.json']=encoded(status)
    changes[root/'README.md']=before[root/'README.md']+('\n\n'+notice+'\n').encode()
    save(out/'protected_sources.json',protected)
    save(out/'backup_manifest.json',[dict(path=str(path),backup=str(backups[path]),before_sha256=hashlib.sha256(data).hexdigest()) for path,data in before.items()])
    marker=root/'timing_correction.json';assert not marker.exists() and not marker.is_symlink()
    try:
        for p in affected:assert p.read_bytes()==before[p],p
        for path,data in changes.items():atomic(path,data)
        marker.symlink_to(out/'policy.json')
        log=out/'scan_audit.log'
        cmd=[sys.executable,str(PROJECT/'experiments/scan_readout/audit_scan.py'),'--result-dir',str(root),'--baseline-videos',str(Path(identities[next(iter(prompt_sets[0]))]['video']['path']).parent)]
        with log.open('w') as stream:
            result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
        assert result.returncode==0,log.read_text()[-5000:]
        g=groups[0];old=json.loads(before[g/'performance.json']);new=new_perfs[g.name]
        # Refresh the complete 1.8x lookup tables without altering quality values.
        for name in ['scan_1p8_plateaus.csv','scan_1p8_E_grid.csv','scan_1p8_per_prompt.csv']:
            rows=list(csv.DictReader(io.StringIO(before[g/name].decode())))
            for row in rows:
                cid=row['condition'];v=new[cid]
                if name.endswith('plateaus.csv'):
                    row['raw_same_gpu_speedup']=row['speedup'];row['speedup']=v['speedup']
                    row['relative_error_percent']=(v['speedup']/1.8-1)*100
                    row['within_target_tolerance']=abs(v['speedup']-1.8)<=selections[g.name]['targets'][0]['tolerance']
                    row['baseline_seconds']=v['sums']['baseline']['generate_seconds']/11
                elif name.endswith('E_grid.csv'):
                    row['raw_same_gpu_speedup']=row['representative_speedup'];row['representative_speedup']=v['speedup']
                else:
                    row['raw_same_gpu_baseline_seconds']=row['baseline_seconds'];row['raw_same_gpu_speedup']=row['speedup']
                    row['baseline_seconds']=refs[row['sample_id']]['pipeline_generate_wall_seconds']
                    row['speedup']=row['baseline_seconds']/float(row['candidate_seconds'])
                row['timing_basis']=POLICY
            atomic(g/name,csv_data(rows))
        readout=before[g/'SCAN_READOUT.md'].decode()
        for cid,v in old.items():
            readout=readout.replace(f"{v['speedup']:.6f}",f"{new[cid]['speedup']:.6f}")
            readout=readout.replace(f"{(v['speedup']/1.8-1)*100:+.3f}%",f"{(new[cid]['speedup']/1.8-1)*100:+.3f}%")
        old_b=next(iter(old.values()))['sums']['baseline'];new_b=next(iter(new.values()))['sums']['baseline']
        readout=readout.replace(f"{old_b['generate_seconds']/11:.6f}",f"{new_b['generate_seconds']/11:.6f}")
        for field in ['t5_cuda_seconds','dit_cuda_seconds','vae_decode_cuda_seconds']:
            readout=readout.replace(f"{old_b[field]/11:.6f}",f"{new_b[field]/11:.6f}")
        readout=readout.replace('加速比 = 11条baseline的完整generate时间之和','加速比 = 共享baseline中对应11条prompt的修正完整generate时间之和')
        readout=readout.replace('实测加速','修正加速').replace('原始结果位于同目录`report.json`、`runs/`、`evaluation/`','当前修正报告位于`report.json`，原始生成和质量证据位于`runs/`、`evaluation/`')
        title,body=readout.split('\n',1);atomic(g/'SCAN_READOUT.md',(title+'\n\n'+notice+'\n'+body).encode())
        one_validation=read(g/'scan_1p8_validation.json')
        one_validation['checked_at']=now;one_validation['timing_correction']=meta
        one_validation['aggregation_checks']='Corrected shared-baseline ratio-of-sums; per-prompt ratios and all three quality metrics match the current report. Original gate checks remain valid because plans, official gate code and schedules are unchanged.'
        for item in one_validation['source_artifacts']:item['sha256']=sha(Path(item['path']))
        atomic(g/'scan_1p8_validation.json',encoded(one_validation))
        # Independently reconstruct ratios from source rows and original candidate records.
        summary=read(root/'scan_summary.json');assert len(summary['candidates'])==14
        expected_sum=sum(refs[sid]['pipeline_generate_wall_seconds'] for sid in prompt_sets[0])
        comparisons=[]
        for row in summary['candidates']:
            perf=new_perfs[row['group']][row['condition']];raw=perf['raw_same_gpu_performance']
            candidate_sum=math.fsum(pair['magcache']['generate_seconds'] for pair in raw['per_video'])
            expected=expected_sum/candidate_sum
            assert math.isclose(row['speedup'],expected,rel_tol=1e-12)
            assert [r['magcache'] for r in perf['per_video']]==[r['magcache'] for r in raw['per_video']]
            assert perf['sums']['magcache']==raw['sums']['magcache']
            comparisons.append(dict(group=row['group'],condition=row['condition'],E=row['E'],raw_same_gpu_speedup=raw['speedup'],corrected_speedup=expected,selected=row['selected']))
        for source,digest in protected.items():assert sha(Path(source))==digest,source
        for g in groups:
            for name,key in [('GENERATION_COMPLETE.json','performance'),('COMPLETE.json','report')]:
                item=read(g/name)[key];assert sha(Path(item['path']))==item['sha256']
        for row in csv.DictReader((groups[0]/'scan_1p8_per_prompt.csv').open()):
            assert math.isclose(float(row['speedup']),refs[row['sample_id']]['pipeline_generate_wall_seconds']/float(row['candidate_seconds']),rel_tol=1e-12)
        validation=dict(status='applied_and_verified',baseline_prompt_count=11,baseline_mean_seconds=meta['baseline_mean_seconds'],
            conditions_recomputed=14,threshold_grid_points=313,candidate_videos_unchanged=154,original_same_gpu_baseline_videos_unchanged=33,
            protected_sources_verified=len(protected),video_sha_verified=len(video_sources),same_prompt_baseline_video_identity=True,
            official_source_lock_unchanged=True,full_scan_audit_passed=read(root/'final_validation.json')['status']=='pass',
            independent_source_ratio_verification=True,selected_parameters_unchanged=all(selections[g]['targets'][0]['nearest_condition']==old_selections[g]['targets'][0]['nearest_condition'] for g in selections),
            speedups={g:selections[g]['targets'][0]['measured_speedup'] for g in selections},backed_up_files=len(before))
        save(out/'VALIDATION.json',validation)
        (out/'comparison.csv').write_bytes(csv_data(comparisons));save(out/'selected.json',summary['selected'])
        manifest=read(out/'backup_manifest.json')
        for item in manifest:item['after_sha256']=sha(Path(item['path']))
        save(out/'backup_manifest.json',manifest)
    except BaseException:
        for path,data in before.items():atomic(path,data)
        if marker.is_symlink():marker.unlink()
        save(out/'VALIDATION.json',dict(status='rolled_back'))
        raise
    table='\n'.join(f"| {r['target']:.1f}× | {r['R']} | {r['E']:.3f} | {r['K']} | {r['generate_seconds']:.6f} | {r['speedup']:.6f}× | {r['relative_error_percent']:+.3f}% |" for r in summary['selected'])
    (out/'REPORT.md').write_text('# MagCache扫描共享baseline修正\n\n'+notice+'\n\n| 目标 | R | E | K | 候选秒/视频 | 修正加速比 | 目标偏差 |\n|---|---|---|---|---|---|---|\n'+table+'\n\n三档配置均保持不变，均命中相对±2%。14组代表/313个网格点的加速比及1.8×完整31点明细均已刷新；全部154条候选的实测耗时、运算量和质量不变。这里只覆盖11条标定prompt，不是完整VBench200。\n\n当前完整结果入口：`'+str(root/'RESULTS.md')+'`；原同卡报告备份见backup_manifest.json，独立核验见VALIDATION.json。\n')
    (out/'README.md').write_text('# MagCache scan normalization package\n\nREPORT.md and selected.json hold the normalized selections. comparison.csv retains old and corrected speeds for all 14 representatives. policy.json pins the corrected shared baseline and reused video identities. backups/ preserves replaced derived files, protected_sources.json records unchanged original evidence, scan_audit.log captures the complete CPU audit, and VALIDATION.json verifies application.\n')
    save(out/'source_code.json',[record(p) for p in sorted(Path(__file__).parent.glob('*.py'))]+[record(PROJECT/'experiments/scan_readout/audit_scan.py')])
    link=PROJECT/'experiment_results'/out.name;assert not link.exists() and not link.is_symlink();link.symlink_to(out,target_is_directory=True)
    print(json.dumps(validation,ensure_ascii=False))


if __name__=='__main__':main()
