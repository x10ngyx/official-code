"""Import existing e391 results into the live run's sealed evaluation cache, CPU only."""
from pathlib import Path
import sys,csv,math,json,statistics
from unittest.mock import patch
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import torch
from ours4wan21.contracts import EXP_ROOT,PROTOCOL,dump,sha256,state_contract
from ours4wan21.online_common import read,seal,verified,run_config
from ours4wan21.online_pipeline import check_run,candidate_job,paired_quality,aggregate_target,dispatch
from ours4wan21.online_generation import job_identity
from ours4wan21.policy import Policy

RUN=EXP_ROOT/'ours21_dynamics128_e391_online_offline800_8rounds_a25_v1'
SOURCE=EXP_ROOT/'ours21_dynamics128_e391_vbench50_random42_4gpu_v1'
METRICS=('psnr_rgb_db','ssim_rgb','lpips_alex_v0_1_spatial')


def csv_rows(path):
    with Path(path).open() as f:
        reader=csv.DictReader(f);return reader.fieldnames,list(reader)


def write_rows(path,fields,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    torch.set_num_threads(1)
    m=check_run(RUN);cfg=run_config(m);source=read(SOURCE/'config.json');complete=read(SOURCE/'COMPLETE.json')
    ckpt=m['paths']['start_checkpoint'];checkpoint_sha=m['inputs']['start_checkpoint']['sha256']
    if (source['checkpoint_sha256']!=checkpoint_sha or source['mode']!=m['mode'] or
        source['protocol']!=PROTOCOL or complete['status']!='complete' or complete['candidate_videos']!=150 or
        source['skip_budgets']!=m['evaluation_budgets']):raise ValueError('offline evaluation/checkpoint mismatch')
    out=RUN/'evaluation_reference';out.mkdir(exist_ok=True)
    (out/'README.md').write_text('Reused original Dynamics128 e391 evaluation on the selected 20 VBench50 prompts. Candidate video/timing/trace and quality scores originate from the completed source archive, not from this online run. Sealed adapters satisfy existing resumable dispatch/quality contracts. No generation or metric computation performed by the importer.\n')
    sources={};all_jobs=[];report=[];policy=Policy(ckpt,device='cpu',state_mode=m['mode'])
    def record(p):
        p=Path(p).resolve(strict=True);sources[str(p)]=sha256(p);return p
    for p in (SOURCE/'config.json',SOURCE/'COMPLETE.json',Path(ckpt)):record(p)
    for target,k in zip(cfg.evaluation_targets,m['evaluation_budgets']):
        target_dir=out/f'target_{target:g}x';target_dir.mkdir(exist_ok=True)
        (target_dir/'README.md').write_text(f'Original e391 K{k} results for this run\'s frozen 20 prompts; baseline and candidate GPU/video hashes verified. Nominal target {target:g}x, actual speeds preserved.\n')
        candidates=target_dir/'candidates';candidates.mkdir(exist_ok=True)
        (candidates/'README.md').write_text('Sealed per-prompt reuse adapters with original provenance; video/timing/trace are symlinks.\n')
        pairs=[];video_rows=[];frame_rows=[];q_sources=[];vf=ff=None
        for prompt in m['evaluation']:
            sid=prompt['sample_id'];shards=[g for g,ids in source['shard_ids'].items() if sid in ids]
            if len(shards)!=1:raise ValueError('ambiguous source shard')
            cell=SOURCE/'shards'/('gpu'+shards[0])/f'K{k}'
            cm=read(record(cell/'run.json'));cc=read(record(cell/'COMPLETE.json'));components=read(record(cell/'components.json'))
            matched=[r for r in cm['prompts'] if r['sample_id']==sid]
            if (cm['protocol']!=PROTOCOL or cm['policy_sha256']!=checkpoint_sha or cm['state_mode']!=m['mode'] or
                cm['skip_budget']!=k or cm['gpu_uuid']!=prompt['baseline_gpu_uuid'] or len(matched)!=1 or matched[0]['prompt']!=prompt['prompt'] or
                cm['method']!='ours' or cm['flops_profile_sha256']!=m['inputs']['flops_profile']['sha256'] or
                Path(cm['checkpoint_dir']).resolve()!=Path(m['paths']['wan_checkpoint']).resolve() or
                cc['status']!='generation_complete' or cc['videos']!=len(cm['prompts'])):raise ValueError('source candidate identity differs')
            links=dict(video=record(cell/'videos'/(sid+'.mp4')),trace=record(cell/'traces'/(sid+'.json')),timing=record(cell/'timings'/(sid+'.json')))
            trace,timing=read(links['trace']),read(links['timing'])
            if trace['state_contract']!=state_contract(m['mode']) or trace['step_reuse']!=k or trace['step_recompute']!=50-k or len(trace['decisions'])!=100:
                raise ValueError('wrong source trace/state/budget')
            if timing['status']!='success' or timing['reuse_forward_calls']!=2*k or len(timing['calls'])!=100:raise ValueError('source timing incomplete')
            decisions=trace['decisions']
            free=[d for d in decisions[::2] if d['policy_queried']]
            # Check source actions against the same e391 deterministic evaluation policy.
            if free:
                x=torch.tensor([d['state'] for d in free],dtype=torch.float32)
                with torch.no_grad():pred=policy.net((x-policy.normalizer['mean'])/policy.normalizer['std']).argmax(-1).tolist()
                if pred!=[int(d['action']=='reuse') for d in free]:raise ValueError('source is not e391 argmax evaluation')
            for i,(d,call) in enumerate(zip(decisions,timing['calls'])):
                if (d['action']!=decisions[i//2*2]['action'] or call['blocks_executed']!=(0 if d['action']=='reuse' else 30)):
                    raise ValueError('source CFG/actions/blocks mismatch')
            measured=[x for x in components['rows'] if x['sample_id']==sid]
            if len(measured)!=1:raise ValueError('source measurement missing')
            measurement={a:b for a,b in measured[0].items() if a!='sample_id'}
            if not math.isclose(measurement['generate_seconds'],timing['pipeline_generate_wall_seconds'],rel_tol=1e-10):raise ValueError('source latency differs')
            for key in ('generate_seconds','dit_tflops','t5_cuda_seconds','dit_cuda_seconds','vae_decode_cuda_seconds','estimated_t5_tflops_per_video','estimated_vae_decode_tflops_per_video'):
                if not math.isfinite(measurement[key]) or measurement[key]<=0:raise ValueError('invalid source components')
            summary=read(record(cell/'quality/summary.json'))
            if summary['protocol_id']!='rgb_full_reference_v1' or summary['selected_metrics']!=['psnr','ssim','lpips']:
                raise ValueError('wrong quality protocol')
            vp=record(cell/'quality/per_video.csv');fp=record(cell/'quality/per_frame.csv');q_sources.append(str(vp))
            fields,vr=csv_rows(vp);f_fields,fr=csv_rows(fp)
            if vf is not None and (vf!=fields or ff!=f_fields):raise ValueError('source CSV schemas differ')
            vf,ff=fields,f_fields;selected=[x for x in vr if x['video_id']==sid];frames=[x for x in fr if x['video_id']==sid]
            base=Path(m['paths']['evaluation_bundle'])/'baselines'/sid
            if (len(selected)!=1 or len(frames)!=81 or sorted(int(x['frame_index']) for x in frames)!=list(range(81))):raise ValueError('quality coverage differs')
            quality=selected[0]
            if (quality['reference_sha256']!=sha256(base/'video.mp4') or quality['candidate_sha256']!=sha256(links['video']) or
                (int(quality['frames']),int(quality['height']),int(quality['width']))!=(81,480,832)):
                raise ValueError('quality matched video/geometry differs')
            for metric in METRICS:
                if not math.isclose(statistics.mean(float(x[metric]) for x in frames),float(quality[metric+'_mean']),rel_tol=1e-9,abs_tol=1e-9):
                    raise ValueError('quality frame/video means differ')
            folder=candidates/sid;job=candidate_job(RUN,m,dict(**prompt,target_speedup=target,skip_budget=k),ckpt,folder)
            identity=job_identity(job,m)
            if not verified(folder,identity):
                if folder.exists():raise ValueError('preexisting incomplete reference; inspect before reimport')
                folder.mkdir()
                (folder/'README.md').write_text('Reused original e391 evaluation; identity describes the requested cache key, not a new execution. generation.json and source_provenance.json retain origin and original code hashes.\n')
                dump(folder/'identity.json',identity)
                dump(folder/'generation.json',dict(identity=identity,protocol=PROTOCOL,gpu_uuid=cm['gpu_uuid'],gpu_name=cm['gpu'],
                    artifact_origin='reused_existing_offline_evaluation',source_run=str(SOURCE),source_cell=str(cell),
                    measurement_scope='original complete generate timing, excluding load/warmup/export; no new execution',
                    wan_checkpoint=cm['checkpoint_dir'],flops_profile_sha256=cm['flops_profile_sha256']))
                dump(folder/'source_provenance.json',dict(source_run=str(SOURCE),source_cell=str(cell),
                    original_run=cm,source_files={str(p):sources[str(p)] for p in [cell/'run.json',cell/'COMPLETE.json',cell/'components.json',vp,fp,*links.values()]},
                    target_speedup_metadata='Source measured explicit K; nominal target is the current label only',quality_reference=quality))
                dump(folder/'measurement.json',measurement)
                for label,path in links.items():(folder/('video.mp4' if label=='video' else label+'.json')).symlink_to(path)
                # Geometry is from archived full-frame quality, FPS from the frozen source protocol.
                dump(folder/'ffprobe.json',dict(origin='archived video quality geometry and frozen source fps; not a new probe',streams=[dict(width=832,height=480,nb_read_frames='81',r_frame_rate='16/1')]))
                seal(folder,['identity.json','generation.json','source_provenance.json','measurement.json','video.mp4','trace.json','timing.json','ffprobe.json'],identity=identity)
            all_jobs.append(job);pairs.append((sid,str(base),str(folder)));video_rows.append(quality);frame_rows.extend(frames)
        quality_dir=target_dir/'quality'
        identity=dict(pairs=[dict(id=sid,baseline=sha256(Path(b)/'video.mp4'),candidate=sha256(Path(c)/'video.mp4')) for sid,b,c in pairs],protocol='rgb_full_reference_v1')
        if not verified(quality_dir,identity):
            if quality_dir.exists():raise ValueError('incomplete existing reference quality')
            quality_dir.mkdir();(quality_dir/'README.md').write_text('Imported original VideoMetrics per-frame and per-video rows for the selected20; metric values and original video paths/hashes retained. No metric re-evaluation.\n')
            dump(quality_dir/'identity.json',identity);metrics=quality_dir/'metrics';metrics.mkdir()
            (metrics/'README.md').write_text('Selected original CSV rows, 20 videos and 1620 frames; no newly measured scores.\n')
            write_rows(metrics/'per_frame.csv',ff,frame_rows);write_rows(metrics/'per_video.csv',vf,video_rows)
            dump(metrics/'summary.json',dict(schema='ours21_reused_offline_quality_v1',protocol_id='rgb_full_reference_v1',
                videos=20,frames=1620,source_run=str(SOURCE),source_quality_files=sorted(set(q_sources)),
                origin='existing per-frame/per-video results, selected subset only; no quality computation',
                mean_metrics={key+'_mean':statistics.mean(float(x[key+'_mean']) for x in video_rows) for key in METRICS}))
            seal(quality_dir,['identity.json','metrics/per_frame.csv','metrics/per_video.csv','metrics/summary.json'],identity=identity)
        with patch('ours4wan21.online_pipeline.invoke',side_effect=AssertionError('must not rerun quality')):
            quality=paired_quality(RUN,m,pairs,quality_dir)
        report.append(dict(target_speedup=target,skip_budget=k,**aggregate_target(pairs,quality)))
    # Exercise the real resume path: not one subprocess may be launched for the 60 offline jobs.
    with patch('ours4wan21.online_pipeline.subprocess.Popen',side_effect=AssertionError('must not regenerate offline checkpoint')):
        dispatch(RUN,m,all_jobs,'offline_reuse_validation')
    check_run(RUN)  # live source/manifest hashes were left unchanged
    dump(out/'REUSE.json',dict(status='complete',source_run=str(SOURCE),source_files=sources,checkpoint_sha256=checkpoint_sha,
        prompts=20,reused_candidates=60,reused_frame_pairs=4860,generated_videos=0,quality_evaluations=0,
        validation='Actual dispatch launches zero workers for all60 cached jobs; actual paired_quality returns all3 existing caches with invoke forbidden; source/checkpoint/prompt/GPU/K/video SHA and CSV means verified',
        per_target=report))
    seal(out,['REUSE.json',*[f'target_{t:g}x/quality/COMPLETE.json' for t in cfg.evaluation_targets],
        *[str(Path(j['output']).relative_to(out)/'COMPLETE.json') for j in all_jobs]],identity=dict(source_run=str(SOURCE),checkpoint_sha256=checkpoint_sha))
    print(json.dumps(dict(reused_candidates=60,reused_frame_pairs=4860,new_inference=0,new_quality=0,per_target=[dict(target_speedup=x['target_speedup'],quality=x['mean_quality'],speedup=x['generate_speedup']) for x in report])),flush=True)

if __name__=='__main__':main()
