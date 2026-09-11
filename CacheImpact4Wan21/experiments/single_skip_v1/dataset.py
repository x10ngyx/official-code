"""Data-pipeline layout, complete raw states, paired tables and atomic releases.

Uses the existing collection pipeline's three-table organization and tensor
format, but a distinct schema: this is a single-skip causal dataset, not a
threshold/Exact-K training corpus.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import time
import math
from common import (PROTOCOL, cell_path, completed, job_plan, read_json,
                    sha, steps, trajectory_id, write_json, digest)

LATENT_SHAPE = (16,21,60,104)


def write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,path)


def records(rows):
    result=[]
    rank={r['sample_id']:i+1 for i,r in enumerate(rows)}
    release=0
    for step,row in job_plan(rows,'all'):
        if step:release+=1
        result.append(dict(**{k:v for k,v in row.items() if k!='split'},trajectory_id=trajectory_id(step,row['sample_id']),
            release_index=release if step else None,prompt_rank=rank[row['sample_id']],
            split=row.get('split','unassigned'),shard_index=0,
            candidate_index_for_prompt=steps('all').index(step)+1 if step else 0,
            policy_family='single_skip' if step else 'full_compute',
            skip_step_1based=step,skip_step_index_0based=step-1 if step else None,
            other_steps='recompute',target_speedup=None,q=None,fixed_threshold=None,
            expected_reuse=1 if step else 0,expected_recompute=49 if step else 50))
    return result


def initialize_manifest(output, rows):
    output=Path(output)
    all_records=records(rows)
    for name,part in [('baselines',[r for r in all_records if not r['skip_step_1based']]),
                      ('candidates',[r for r in all_records if r['skip_step_1based']])]:
        path=output/'manifests'/f'{name}.jsonl'
        if path.exists():
            if [json.loads(l) for l in path.read_text().splitlines()]!=part:
                raise ValueError('dataset manifest changed')
        else:write_jsonl(path,part)
    return {(r['skip_step_1based'],r['sample_id']):r for r in all_records}


def save_capture(controller, root, baseline, record):
    import torch
    root=Path(root); baseline=Path(baseline)
    if len(controller.input_latents)!=50 or len(controller.step_metadata)!=50:
        raise ValueError('dataset requires all 50 raw inputs, not only target features')
    latent_dir=root/'latents'
    latent_dir.mkdir(exist_ok=False)
    trace=controller.summary()
    step_rows=[]
    for i,(latent,metadata) in enumerate(zip(controller.input_latents,controller.step_metadata)):
        if metadata['step_index']!=i or metadata['sigma'] is None or metadata['timestep'] is None:
            raise ValueError('incomplete solver step metadata')
        cpu=latent.detach().to(device='cpu',dtype=torch.float16).contiguous()
        if tuple(cpu.shape)!=LATENT_SHAPE or not torch.isfinite(cpu).all():
            raise ValueError('invalid saved Wan21 raw latent')
        path=latent_dir/f'step_{i:03d}_input.pt'
        torch.save(cpu,path)
        pair=trace['decisions'][2*i:2*i+2]
        if len(pair)!=2 or pair[0]['action']!=pair[1]['action']:
            raise ValueError('CFG action mismatch')
        row=dict(**metadata,action=pair[0]['action'],cond_action=pair[0]['action'],
            uncond_action=pair[1]['action'],reason={r['branch']:r['reason'] for r in pair},
            branches={r['branch']:r['action'] for r in pair},branch_decisions={r['branch']:r for r in pair},
            requested_threshold=None,native_forced_recompute=i==0,
            latent_path=str(path.resolve()),baseline_latent_path=str((baseline/'latents'/path.name).resolve()),
            latent_sha256=sha(path),latent_shape=list(cpu.shape),latent_dtype=str(cpu.dtype),
            latent_mean=float(cpu.float().mean()),latent_std=float(cpu.float().std(unbiased=False)),
            latent_min=float(cpu.min()),latent_max=float(cpu.max()),
            distance_reference='previous_step_same_cfg_branch',
            distance_feature='sea_filtered_first_block_modulated_input',distance_metric='relative_l1_mean')
        for field in ['filtered_relative_l1','accumulated_distance_before',
                      'accumulated_distance_with_current','accumulated_distance_after']:
            row[field]={r['branch']:r[field] for r in pair}
            for r in pair:row[r['branch']+'_'+field]=r[field]
        for r in pair:
            r.update(metadata)
        step_rows.append(row)
    controller.input_latents.clear()
    trace.update(trajectory_id=record['trajectory_id'],manifest_record=record,
        schema='cacheimpact_wan21_trace_v2',step_records=step_rows,latent_save_dtype='torch.float16',
        artifact_io_outside_pipeline_generate_timing=True,task='t2v-1.3B',sampling_steps=50,
        sample_solver='unipc',shift=5.,guide_scale=5.,frame_num=81,size_wh=[832,480])
    write_json(root/'trace.json',trace)
    (root/'README.md').write_text('# Trajectory bundle\n\nlatents/ contains 50 raw pre-action FP16 tensors; trace.json contains 50 step and 100 CFG records. video.mp4 has a baseline.mp4/candidate.mp4 alias; rgb_f32.npy is the pre-encode terminal output. metrics.json retains the raw terminal MSE, performance.json the component measurements, ffprobe.json the exported video metadata. Generation COMPLETE.json and finalized dataset completion are separate.\n')
    return trace


def probe_video(path):
    command=['ffprobe','-v','error','-select_streams','v:0','-count_frames',
             '-show_streams','-show_format','-of','json',str(path)]
    result=json.loads(subprocess.check_output(command,text=True))
    streams=result.get('streams',[])
    if len(streams)!=1:raise ValueError('expected one video stream')
    s=streams[0]
    from fractions import Fraction
    if (int(s['width']),int(s['height']),int(s.get('nb_read_frames',s.get('nb_frames',0))))!=(832,480,81):
        raise ValueError('exported video shape/frame mismatch')
    if Fraction(s['avg_frame_rate'])!=16:raise ValueError('exported FPS mismatch')
    return result


def finalize_cell(output, step, row):
    """Publish completion only after canonical VideoMetrics and raw labels exist."""
    output=Path(output); root=cell_path(output,step,row['sample_id'])
    baseline=cell_path(output,0,row['sample_id'])
    manifest=read_json(output/'run.json')
    if not completed(root,manifest['contract_hash']):raise ValueError('incomplete generation')
    trace=read_json(root/'trace.json'); identity=trace['manifest_record']
    if trace['trajectory_id']!=trajectory_id(step,row['sample_id']):raise ValueError('wrong trajectory identity')
    if len(trace['step_records'])!=50:raise ValueError('missing step rows')
    canonical=root/'video_metrics'
    if not (canonical/'summary.json').exists() or not (root/'quality.json').exists():
        raise ValueError('both raw RGB and canonical MP4 quality required')
    summary=read_json(canonical/'summary.json')
    if (summary['protocol_id']!='rgb_full_reference_v1' or summary['frame_count_total']!=81
        or summary.get('video_count')!=1 or summary.get('selected_metrics')!=['psnr','ssim','lpips']):
        raise ValueError('canonical metric protocol mismatch')
    with (canonical/'per_video.csv').open() as f:video_rows=list(csv.DictReader(f))
    if len(video_rows)!=1:raise ValueError('one canonical video row required')
    video=video_rows[0]; primary=read_json(root/'metrics.json'); bp=read_json(baseline/'metrics.json')
    names={'psnr':'psnr_rgb_db','ssim':'ssim_rgb','lpips':'lpips_alex_v0_1_spatial'}
    with (canonical/'per_frame.csv').open() as f:frames=list(csv.DictReader(f))
    if [int(r['frame_index']) for r in frames]!=list(range(81)) or int(video['frames'])!=81:
        raise ValueError('canonical frame table must have 81 ordered rows')
    for name in names.values():
        values=[float(r[name]) for r in frames]
        if not all(math.isfinite(v) for v in values) or not math.isclose(sum(values)/81,float(video[name+'_mean']),rel_tol=1e-8,abs_tol=1e-8):
            raise ValueError('canonical frame/video metric mismatch')
    compact=dict(schema='cacheimpact_video_metrics_v1',protocol_id='rgb_full_reference_v1',
        selected_metrics=['psnr','ssim','lpips'],metric_names=names,video_id=identity['trajectory_id'],
        reference=video['reference'],candidate=video['candidate'],
        reference_sha256=video['reference_sha256'],candidate_sha256=video['candidate_sha256'],
        frames=int(video['frames']),evaluation_elapsed_seconds=summary['evaluation_elapsed_seconds'],
        metrics={name:{stat:float(video[f'{name}_{stat}']) for stat in ['mean','std_population','min','max']} for name in names.values()})
    write_json(canonical/'metrics.json',compact)
    trajectory=dict(**identity,baseline_video=str((baseline/'baseline.mp4').resolve()),
        candidate_video=str((root/('candidate.mp4' if step else 'baseline.mp4')).resolve()),
        trace_json=str((root/'trace.json').resolve()),baseline_trace_json=str((baseline/'trace.json').resolve()),
        latent_dir=str((root/'latents').resolve()),baseline_latent_dir=str((baseline/'latents').resolve()),
        timing_json=str((root/'timing.json').resolve()),performance_json=str((root/'performance.json').resolve()),
        video_metrics_json=str((canonical/'metrics.json').resolve()),
        video_metrics_per_frame_csv=str((canonical/'per_frame.csv').resolve()),
        video_metrics_per_video_csv=str((canonical/'per_video.csv').resolve()),
        video_metrics_summary_json=str((canonical/'summary.json').resolve()),
        video_metrics_protocol='rgb_full_reference_v1',metric_frames=81,action_count_unit='denoising_step',
        actual_reuse=int(step>0),actual_recompute=50-int(step>0),
        actual_reuse_branch_calls=2*int(step>0),actual_recompute_branch_calls=100-2*int(step>0),
        actual_mixed_steps=0,baseline_inference_seconds=bp['generate_seconds'],
        candidate_inference_seconds=primary['generate_seconds'],
        inference_latency_speedup=bp['generate_seconds']/primary['generate_seconds'],
        dit_flops_speedup=bp['estimated_dit_tflops']/primary['estimated_dit_tflops'],
        terminal_rgb_mse=primary['terminal_rgb_mse'],terminal_impact_scope='single_skip_with_all_other_steps_recompute')
    for prefix,m in [('baseline',bp),('candidate',primary)]:
        for k in ['estimated_dit_tflops','t5_cuda_seconds','dit_cuda_seconds','vae_decode_cuda_seconds',
                  'estimated_t5_tflops_per_video','estimated_vae_decode_tflops_per_video']:
            trajectory[prefix+'_'+k]=m[k]
    for short,long in names.items():
        for field,stat in [('mean','mean'),('std','std_population'),('min','min'),('max','max')]:
            trajectory[field+'_'+short]=float(video[f'{long}_{stat}'])
    terminal=dict(final_mean_psnr=trajectory['mean_psnr'],final_mean_ssim=trajectory['mean_ssim'],
                  final_mean_lpips=trajectory['mean_lpips'],final_terminal_rgb_mse=primary['terminal_rgb_mse'],
                  final_inference_latency_speedup=trajectory['inference_latency_speedup'],
                  final_dit_flops_speedup=trajectory['dit_flops_speedup'],
                  baseline_inference_seconds=bp['generate_seconds'],candidate_inference_seconds=primary['generate_seconds'])
    step_rows=[dict(**identity,**r,**terminal,is_intervention_step=bool(step and r['step_index']==step-1)) for r in trace['step_records']]
    branches=[]
    for r in trace['decisions']:
        state=step_rows[r['step_index']]
        branches.append(dict(**identity,**r,**terminal,latent_path=state['latent_path'],
            baseline_latent_path=state['baseline_latent_path'],latent_shape=state['latent_shape'],latent_dtype=state['latent_dtype']))
    hashes={str(p.relative_to(root)):sha(p) for p in [root/'quality.json',root/'quality_per_frame.csv',*canonical.iterdir()] if p.is_file()}
    payload=dict(schema='cacheimpact_wan21_candidate_complete_v2' if step else 'cacheimpact_wan21_baseline_complete_v2',
        trajectory_id=identity['trajectory_id'],release_index=identity['release_index'],
        contract_hash=manifest['contract_hash'],trajectory_row=trajectory,step_rows=step_rows,branch_rows=branches,
        generation_complete_sha256=sha(root/'COMPLETE.json'),quality_sha256=hashes)
    marker=output/'completed'/f'{identity["trajectory_id"]}.json' if step else root/'BASELINE_COMPLETE.json'
    write_json(marker,payload)
    return payload


def load_finalized(output, step, row, verify=True):
    output=Path(output);root=cell_path(output,step,row['sample_id'])
    marker=output/'completed'/f'{trajectory_id(step,row["sample_id"])}.json' if step else root/'BASELINE_COMPLETE.json'
    if not marker.exists():return None
    p=read_json(marker)
    if p['trajectory_id']!=trajectory_id(step,row['sample_id']):raise ValueError('completion identity mismatch')
    if p['generation_complete_sha256']!=sha(root/'COMPLETE.json'):raise ValueError('generation marker changed')
    if not completed(root,read_json(output/'run.json')['contract_hash'],verify=verify):raise ValueError('generation missing')
    for name,digest in p['quality_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('finalized quality changed')
    if len(p['step_rows'])!=50 or len(p['branch_rows'])!=100:raise ValueError('table counts invalid')
    if {f.name for f in (root/'latents').glob('*.pt')}!={f'step_{i:03d}_input.pt' for i in range(50)}:
        raise ValueError('50 raw latent files required')
    if [r['step_index'] for r in p['step_rows']]!=list(range(50)):
        raise ValueError('step table order invalid')
    if [(r['call_index'],r['step_index'],r['branch']) for r in p['branch_rows']] != [(i,i//2,('cond','uncond')[i%2]) for i in range(100)]:
        raise ValueError('branch table order invalid')
    return p


def publish(args):
    output=Path(args.output); manifest=read_json(output/'run.json')
    selected=[]
    # Longest contiguous completed candidate prefix in the frozen step-major order.
    for step,row in job_plan(manifest['prompts'],'all'):
        if not step:continue
        item=load_finalized(output,step,row)
        if item is None:break
        selected.append(item)
    if not selected:
        print('No fully evaluated contiguous candidate prefix to publish');return
    group_scores={}
    for path in (output/'vbench').glob('*/vbench_custom_aggregate_scores.json'):
        if (path.parent/'COMPLETE.json').exists():
            group_scores[path.parent.name]=dict(vbench_score=read_json(path)['vbench_score'],
                score_scope='group-level custom raw mean; not per-video or official Total Score',
                path=str(path.resolve()),sha256=sha(path))
    signature=digest(dict(contract_hash=manifest['contract_hash'],
        completions=[sha(output/'completed'/f'{p["trajectory_id"]}.json') for p in selected],
        vbench_group_scores=group_scores))
    current=output/'published/current/summary.json'
    if current.exists() and read_json(current).get('publication_signature')==signature:
        print('Published snapshot already matches completed dataset');return
    published=output/'published'; snapshot=published/'snapshots'/str(time.time_ns())
    tables=snapshot/'tables';tables.mkdir(parents=True)
    from evaluate import csv_file
    for name,key in [('trajectory_summary','trajectory_row'),('step_transitions','step_rows'),('branch_transitions','branch_rows')]:
        rows=[p[key] for p in selected] if key=='trajectory_row' else [r for p in selected for r in p[key]]
        write_jsonl(tables/f'{name}.jsonl',rows)
        csv_file(tables/f'{name}.csv',[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
    write_json(snapshot/'summary.json',dict(schema='cacheimpact_published_snapshot_v2',
        published_candidate_count=len(selected),published_step_count=len(selected)*50,
        published_branch_transition_count=len(selected)*100,expected_candidate_count=len(manifest['prompts'])*49,
        complete=len(selected)==len(manifest['prompts'])*49,contract_hash=manifest['contract_hash'],
        publication_signature=signature,vbench_group_scores=group_scores,
        manifest_sha256=sha(output/'manifests/candidates.jsonl'),
        path_scope='absolute archive paths; regenerate publication after relocating archive',
        label_scope='final_terminal_rgb_mse is episode outcome; only is_intervention_step identifies manipulated action'))
    (snapshot/'README.md').write_text('# Dataset snapshot\n\ntables/ contains trajectory, step and CFG branch JSONL/CSV. Read summary.json for count/scope and SHA256SUMS.json for hashes. Paths reference the immutable experiment archive. Schema differs from threshold/Exact-K training datasets, especially the explicitly permitted last-step skip.\n')
    write_json(snapshot/'SHA256SUMS.json',{str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()})
    temporary=published/f'.current.{os.getpid()}'
    temporary.symlink_to(snapshot.relative_to(published),target_is_directory=True)
    os.replace(temporary,published/'current')
    print(f'Published {len(selected)} trajectories at {snapshot}')
