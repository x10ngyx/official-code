"""Incremental raw-RGB fidelity and explicitly labelled custom VBench scores."""
import csv
from pathlib import Path
import os
import subprocess
import sys
import time
from common import PACKAGES, WORKSPACE, cell_path, completed, read_json, sha, steps, write_json


def csv_file(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    tmp = Path(str(path)+f'.tmp.{os.getpid()}')
    with tmp.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp,path)


def summarize(output):
    output = Path(output)
    manifest = read_json(output/'run.json')
    rows = []
    for s in [0]+steps('all'):
        for prompt in manifest['prompts']:
            folder = cell_path(output,s,prompt['sample_id'])
            if not (folder/'COMPLETE.json').exists():
                continue
            row = read_json(folder/'metrics.json')
            row.pop('per_frame_rgb_mse',None)
            if (folder/'quality.json').exists():
                row.update({'preencode_'+k:v for k,v in read_json(folder/'quality.json')['means'].items()})
            if (folder/'video_metrics/metrics.json').exists():
                row.update({k:v['mean'] for k,v in read_json(folder/'video_metrics/metrics.json')['metrics'].items()})
            if s:
                trace = read_json(folder/'trace.json')['decisions']
                for i,branch in enumerate(['cond','uncond']):
                    row[f'{branch}_sea_relative_l1'] = trace[2*(s-1)+i]['sea_proxy']
                    row[f'{branch}_modulated_input_relative_l1'] = trace[2*(s-1)+i]['raw_proxy']
            rows.append(row)
    csv_file(output/'summary.csv',rows)
    aggregate = []
    for s in [0]+steps('all'):
        group = [r for r in rows if r['skip_step_1based']==s]
        if not group:
            continue
        keys = ['terminal_rgb_mse','generate_seconds','estimated_dit_tflops',
                'psnr_rgb_db','ssim_rgb','lpips_alex_v0_1_spatial']
        record = dict(step_1based=s, completed_prompts=len(group),expected_prompts=len(manifest['prompts']))
        for key in keys:
            available = [r[key] for r in group if key in r]
            if available:
                record[key+'_mean'] = sum(available)/len(available)
                record[key+'_count'] = len(available)
        vb = output/'vbench'/('baseline' if s==0 else f'step_{s:02d}')/'vbench_custom_aggregate_scores.json'
        if vb.exists():
            record['vbench_score'] = read_json(vb)['vbench_score']
            record['vbench_scope'] = 'custom_input local raw mean; not official Total Score'
        aggregate.append(record)
    csv_file(output/'by_step.csv',aggregate)
    return rows


def quality(args):
    import numpy as np
    sys.path.insert(0,str(PACKAGES/'VideoMetrics'))
    from video_metrics.core import psnr_per_frame, ssim_per_frame, LPIPSComputer
    from collect import mse_frames
    from dataset import finalize_cell, load_finalized
    from video_metrics.evaluator import evaluate_pairs, resolve_single_pair, write_evaluation
    output = Path(args.output)
    manifest = read_json(output/'run.json')
    lpips = LPIPSComputer(device=args.device,batch_size=1)
    count = 0
    for step in [0]+steps(args.stage):
        for row in manifest['prompts']:
            folder = cell_path(output,step,row['sample_id'])
            base = cell_path(output,0,row['sample_id'])
            if not (folder/'COMPLETE.json').exists():
                continue
            completed(folder,manifest['contract_hash'])
            completed(base,manifest['contract_hash'])
            if load_finalized(output,step,row) is not None:
                continue
            identity = dict(reference_sha256=sha(base/'rgb_f32.npy'),candidate_sha256=sha(folder/'rgb_f32.npy'),
                            protocol='VideoMetrics kernels on lossless pre-encode RGB float32')
            if (folder/'quality.json').exists():
                previous = read_json(folder/'quality.json')
                if (previous['identity'] != identity or
                    previous['per_frame_sha256'] != sha(folder/'quality_per_frame.csv')):
                    raise ValueError('quality provenance mismatch')
            else:
                ref = np.load(base/'rgb_f32.npy',mmap_mode='r')
                cand = np.load(folder/'rgb_f32.npy',mmap_mode='r')
                frame_rows = []
                for i in range(81):
                    a,b = np.array(ref[i:i+1]),np.array(cand[i:i+1])
                    frame_rows.append(dict(frame_index=i, rgb_mse=mse_frames(a,b)[0],
                         psnr_rgb_db=float(psnr_per_frame(a,b)[0]), ssim_rgb=float(ssim_per_frame(a,b)[0]),
                         lpips_alex_v0_1_spatial=float(lpips.per_frame(a,b)[0])))
                means = {k:float(np.mean([r[k] for r in frame_rows])) for k in frame_rows[0] if k!='frame_index'}
                recorded = read_json(folder/'metrics.json')['terminal_rgb_mse']
                if not np.isclose(recorded,means['rgb_mse'],rtol=1e-12,atol=1e-15):
                    raise ValueError('raw RGB MSE failed independent evaluation')
                csv_file(folder/'quality_per_frame.csv',frame_rows)
                write_json(folder/'quality.json',dict(identity=identity,means=means,
                           per_frame_sha256=sha(folder/'quality_per_frame.csv'),
                           psnr_cap='shared VideoMetrics cap=100 dB for MSE<1e-10',
                           device=args.device, metrics_source=manifest['sources']))
            # Canonical dataset metrics must use MP4 decoding, as in data_collection.
            frame_data,video_data,summary = evaluate_pairs(
                resolve_single_pair(base/'video.mp4',folder/'video.mp4',folder.name),
                metrics=('psnr','ssim','lpips'),device=args.device,
                expected_frames=81,lpips_computer=lpips)
            write_evaluation(folder/'video_metrics',frame_data,video_data,summary,overwrite=True)
            finalize_cell(output,step,row)
            count += 1
            summarize(output)
            print(f'QUALITY step={step} prompt={row["sample_id"]}',flush=True)
            if args.max_new_cells is not None and count>=args.max_new_cells:
                return
    from dataset import publish
    publish(args)


def vbench(args):
    output = Path(args.output)
    manifest = read_json(output/'run.json')
    env = {**os.environ,'PYTHON_BIN':sys.executable,
           'VBENCH_CACHE_DIR':str(WORKSPACE/'models/VBench'),
           'TORCH_HOME':str(WORKSPACE/'models/torch-cache'),
           'HF_HOME':str(WORKSPACE/'models/huggingface'),
           'XDG_CACHE_HOME':str(WORKSPACE/'models/xdg')}
    for step in [0]+steps(args.stage):
        cells = [cell_path(output,step,r['sample_id']) for r in manifest['prompts']]
        if not all((p/'COMPLETE.json').exists() for p in cells):
            continue
        for p in cells:
            completed(p,manifest['contract_hash'])
        label = 'baseline' if step==0 else f'step_{step:02d}'
        staging = output/'vbench_staging'/label
        staging.mkdir(parents=True,exist_ok=True)
        mapping = {}
        for row,p in zip(manifest['prompts'],cells):
            name = row['sample_id']+'.mp4'
            link = staging/name
            if not link.exists():
                link.symlink_to(p/'video.mp4')
            if link.resolve() != (p/'video.mp4').resolve():
                raise ValueError('foreign VBench staging link')
            mapping[name] = row['prompt']
        prompt_map = output/'vbench_staging'/f'{label}_prompts.json'
        write_json(prompt_map,mapping)
        target = output/'vbench'/label
        identity = {p.parent.name+'/'+p.name:sha(p/'video.mp4') for p in cells}
        if (target/'COMPLETE.json').exists():
            marker = read_json(target/'COMPLETE.json')
            if marker['videos'] != identity or marker['aggregate_sha256'] != sha(target/'vbench_custom_aggregate_scores.json'):
                raise ValueError('VBench artifact mismatch')
            continue
        if target.exists():
            target.rename(target.with_name(label+f'.incomplete.{time.time_ns()}'))
        subprocess.run(['bash',str(PACKAGES/'VbenchEvaluation/run_custom_vbench.sh'),
                        str(staging),str(target),str(prompt_map)],env=env,check=True)
        write_json(target/'COMPLETE.json',dict(videos=identity,
                    aggregate_sha256=sha(target/'vbench_custom_aggregate_scores.json'),
                    score_scope='custom_input local raw-score mean, not official VBench Total Score'))
        summarize(output)
    from dataset import publish
    publish(args)


def audit(args):
    from common import validate_trace, validate_timing, job_plan
    output = Path(args.output)
    m = read_json(output/'run.json')
    from dataset import load_finalized
    missing = []
    for s,r in job_plan(m['prompts'],args.stage):
        cell = cell_path(output,s,r['sample_id'])
        if not completed(cell,m['contract_hash']):
            missing.append(str(cell.relative_to(output)))
            continue
        validate_trace(read_json(cell/'trace.json'),s)
        validate_timing(read_json(cell/'timing.json'),s)
        if load_finalized(output,s,r) is None:
            missing.append(str(cell.relative_to(output))+'/dataset_completion')
        if s and not (cell/'quality.json').exists():
            missing.append(str(cell.relative_to(output))+'/quality.json')
        elif s:
            q = read_json(cell/'quality.json')
            base = cell_path(output,0,r['sample_id'])
            if (q['identity']['candidate_sha256'] != sha(cell/'rgb_f32.npy') or
                q['identity']['reference_sha256'] != sha(base/'rgb_f32.npy') or
                q['per_frame_sha256'] != sha(cell/'quality_per_frame.csv')):
                raise ValueError('quality hash mismatch')
    for s in [0]+steps(args.stage):
        label = 'baseline' if s==0 else f'step_{s:02d}'
        folder = output/'vbench'/label
        if not (folder/'COMPLETE.json').exists():
            missing.append('vbench/'+label)
        else:
            marker = read_json(folder/'COMPLETE.json')
            identity = {p.parent.name+'/'+p.name:sha(p/'video.mp4')
                        for p in [cell_path(output,s,r['sample_id']) for r in m['prompts']]}
            if marker['videos'] != identity or marker['aggregate_sha256'] != sha(folder/'vbench_custom_aggregate_scores.json'):
                raise ValueError('VBench hash mismatch')
    write_json(output/f'AUDIT_{args.stage}.json',dict(status='complete' if not missing else 'incomplete',
               missing=missing,stage=args.stage))
    print(f'Audit {args.stage}: {len(missing)} missing artifacts')
    if missing:
        raise SystemExit(2)
