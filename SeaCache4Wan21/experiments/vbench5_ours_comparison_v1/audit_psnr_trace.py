"""Recompute PSNR and export exact 50-step SeaCache/SEA7 action matrices."""
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys
import numpy as np
import run_4gpu as suite

OUT = suite.ROOT/'analysis/trace_psnr_audit'
sys.path.insert(0,str(suite.common.OFFICIAL/'VideoMetrics'))
from video_metrics.video import decode_video_rgb
from video_metrics.core import psnr_per_frame


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def actions(trace, timing, branch):
    selected=[r for r in trace['decisions'] if r['branch']==branch]
    assert [r['step_index'] for r in selected]==list(range(50))
    reuse=[int(r['action']=='reuse') for r in selected]
    assert [i for i,x in enumerate(reuse) if x]==trace['per_branch'][branch]['reuse_path']
    assert sum(reuse)==trace['per_branch'][branch]['reuse']
    offset=0 if branch=='cond' else 1
    for step, value in enumerate(reuse):
        assert timing['calls'][step*2+offset]['blocks_executed']==(0 if value else 30)
    return reuse,selected


def longest(values):
    best=run=0
    for value in values:
        run=run+1 if value else 0;best=max(best,run)
    return best


def main():
    if OUT.exists():raise FileExistsError('preserve previous audit')
    OUT.mkdir()
    (OUT/'README.md').write_text('# PSNR and exact trace audit\n\nPSNR_RECHECK.csv contains freshly decoded per-video PSNR checks. trace_steps.csv / traces.sqlite contain each observed branch action, not averages. trace_summary.csv compares action placement. AUDIT.json contains provenance and validation. Widget payloads are split by target, with all 50 steps and five prompts retained.\n')
    ids=[f'vbench200_{s}' for s in ('001','016','056','135','159')]
    reference_hashes={sid:[suite.sha256(suite.common.PREVIOUS/f'baselines/gpu{g}/videos/{sid}.mp4') for g in range(4)] for sid in ids}
    assert all(len(set(v))==1 for v in reference_hashes.values())
    baseline_metadata=suite.read(suite.common.PREVIOUS/'candidates/sea7/K23/quality/video_metrics/summary.json')
    summaries=list((suite.common.PREVIOUS/'candidates').glob('*/K*/quality/video_metrics/summary.json'))
    summaries += [suite.ROOT/f'target_{t}/quality/summary.json' for t in range(3)]
    assert len(summaries)==39
    for path in summaries:
        summary=suite.read(path)
        for key in ('protocol_id','upstream_lock','selected_metrics','aggregation','lpips_batch_size'):
            assert summary[key]==baseline_metadata[key]
        for key in ('python','platform','packages','thread_environment'):
            assert summary['software'][key]==baseline_metadata['software'][key]
        a=Path(summary['software']['ffmpeg']['executable']);b=Path(baseline_metadata['software']['ffmpeg']['executable'])
        assert a.resolve()==b.resolve() and summary['software']['ffmpeg']['version']==baseline_metadata['software']['ffmpeg']['version']
    psnr_checks=[];steps=[];stats=[];sources={}
    for t,(target,k) in enumerate(zip(suite.common.TARGETS,(23,29,35))):
        directories={'SeaCache':suite.ROOT/f'target_{t}', 'SEA7':suite.common.PREVIOUS/f'candidates/sea7/K{k}'}
        cached={}
        for method,d in directories.items():
            qd=d/'quality' if method=='SeaCache' else d/'quality/video_metrics'
            qs={r['video_id']:r for r in csv_rows(qd/'per_video.csv')}
            pf=csv_rows(qd/'per_frame.csv')
            saved_frames={sid:[r for r in pf if r['video_id']==sid] for sid in ids}
            source_paths=[qd/'summary.json',qd/'per_video.csv',qd/'per_frame.csv']
            for sid in ids:
                q=qs[sid];refpath=Path(q['reference']);candpath=Path(q['candidate'])
                assert suite.sha256(refpath)==q['reference_sha256']==reference_hashes[sid][0]
                assert suite.sha256(candpath)==q['candidate_sha256']
                ref=decode_video_rgb(refpath);candidate=decode_video_rgb(candpath)
                measured=psnr_per_frame(ref,candidate)
                del ref,candidate
                assert len(measured)==81
                assert [int(r['frame_index']) for r in saved_frames[sid]]==list(range(81))
                saved=np.array([float(r['psnr_rgb_db']) for r in saved_frames[sid]])
                error=float(np.max(np.abs(measured-saved)))
                assert error<1e-10 and abs(float(measured.mean())-float(q['psnr_rgb_db_mean']))<1e-10
                psnr_checks.append(dict(target=target,method=method,sample_id=sid,frames=81,
                    recorded_psnr=float(q['psnr_rgb_db_mean']),recomputed_psnr=float(measured.mean()),max_frame_error=error,
                    reference_sha256=q['reference_sha256'],candidate_sha256=q['candidate_sha256']))
                tracepath=d/'traces'/f'{sid}.json';timingpath=d/'timings'/f'{sid}.json'
                source_paths.extend([tracepath,timingpath])
                trace=suite.read(tracepath);timing=suite.read(timingpath)
                assert trace['total_steps']==50 and trace['total_branch_calls']==len(timing['calls'])==100
                for b,branch in enumerate(('cond','uncond')):
                    path,decisions=actions(trace,timing,branch)
                    cached[method,sid,branch]=path
                    order=ids.index(sid)*4+(0 if method=='SeaCache' else 2)+b
                    trace_label=f'{order+1:02d} | {sid[-3:]} | {method} {branch}'
                    for step,value in enumerate(path):
                        decision=decisions[step]
                        steps.append(dict(target=target,sample_id=sid,method=method,branch=branch,trace=trace_label,
                            step=step,step_label=f'{step:02d}',reuse=value,action='reuse' if value else 'recompute',
                            executed_blocks=0 if value else 30,skip_count=sum(path),psnr=float(q['psnr_rgb_db_mean']),
                            reason=decision['reason'],threshold=trace.get('threshold'),skip_budget=trace.get('skip_budget'),
                            source_trace=str(tracepath)))
                print(f'PSNR and trace checked: {target} {method} {sid}',flush=True)
            sources.update({str(path):suite.sha256(path) for path in source_paths})
        for sid in ids:
            for branch in ('cond','uncond'):
                sc=cached['SeaCache',sid,branch];ours=cached['SEA7',sid,branch]
                stats.append(dict(target=target,sample_id=sid,branch=branch,sea_reuse=sum(sc),ours_reuse=sum(ours),
                    disagree_steps=sum(a!=b for a,b in zip(sc,ours)),sea_first_reuse=sc.index(1),ours_first_reuse=ours.index(1),
                    sea_longest_reuse=longest(sc),ours_longest_reuse=longest(ours),
                    sea_reuse_last10=sum(sc[40:]),ours_reuse_last10=sum(ours[40:]),
                    seacache_cond_uncond_disagree=sum(a!=b for a,b in zip(cached['SeaCache',sid,'cond'],cached['SeaCache',sid,'uncond']))))
    assert len(steps)==3000 and len(psnr_checks)==30
    write_csv(OUT/'trace_steps.csv',steps);write_csv(OUT/'trace_summary.csv',stats);write_csv(OUT/'PSNR_RECHECK.csv',psnr_checks)
    with sqlite3.connect(OUT/'traces.sqlite') as db:
        db.execute('CREATE TABLE trace_steps (target REAL, sample_id TEXT, method TEXT, branch TEXT, trace TEXT, step INTEGER, step_label TEXT, reuse INTEGER, action TEXT, executed_blocks INTEGER, skip_count INTEGER, psnr REAL, reason TEXT, threshold REAL, skip_budget INTEGER, source_trace TEXT)')
        db.executemany('INSERT INTO trace_steps VALUES ('+','.join('?' for _ in steps[0])+')',[list(r.values()) for r in steps])
        db.commit()
        for target in suite.common.TARGETS:
            sql=f'SELECT * FROM trace_steps WHERE target={target} ORDER BY trace, step'
            cursor=db.execute(sql);cols=[d[0] for d in cursor.description]
            data=[dict(zip(cols,row)) for row in cursor]
            suite.dump(OUT/f'widget_{target:.1f}.json',dict(title=f'50-step action traces — nominal {target:.1f}×',
                subtitle='0 = recompute; 1 = reuse. Each prompt retains both CFG branches; steps 00–49.',
                source=dict(label='SeaCache / SEA7 measured branch traces',path=str(OUT/'traces.sqlite'),
                    query=dict(engine='SQLite',language='sql',sql=sql,tables_used=['trace_steps'],description='Exact trace actions cross-checked against executed Transformer blocks; no action averaging.')),
                table=dict(rows=data,row_count=len(data),truncated=False),
                chart=dict(type='heatmap',fields=dict(x={'field':'step_label'},y={'field':'reuse','aggregate':'max'},color={'field':'trace'})),
                display=dict(controls=True,x_axis_title='Denoising step (0-based)',y_axis_title='Prompt / method / CFG branch',unit='reuse (0/1)')))
    audit=dict(status='pass',metric_summaries_compared=39,baseline_bytes_identical_across_4gpus=True,
        psnr_recomputed_videos=30,psnr_recomputed_frames=2430,max_frame_psnr_error=max(r['max_frame_error'] for r in psnr_checks),
        metric_protocol='rgb_full_reference_v1',ffmpeg_paths_are_same_resolved_binary=True,
        raw_trace_branch_step_rows=3000,source_sha256=sources,
        metric_code_sha256={str(p):suite.sha256(p) for p in (suite.common.OFFICIAL/'VideoMetrics/evaluate.py',suite.common.OFFICIAL/'VideoMetrics/video_metrics/core.py',suite.common.OFFICIAL/'VideoMetrics/video_metrics/video.py')},
        script_sha256=suite.sha256(Path(__file__)),checked_at=datetime.now(timezone.utc).isoformat())
    suite.dump(OUT/'AUDIT.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('source_sha256','metric_code_sha256')},indent=2))


if __name__=='__main__':main()
