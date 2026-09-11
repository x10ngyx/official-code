"""Paired validation and official-code quality tools, including VBench."""
import os
import json
from pathlib import Path
import subprocess
import sys
from .contracts import PROTOCOL
from .shared import OFFICIAL, MODELS, read, write, sha256, result_dir, under


def validate_vbench_jobs(jobs, mode):
    if mode != 'vbench200':
        return
    rows = [json.loads(line) for line in (OFFICIAL/'Vbench200/prompts.jsonl').read_text().splitlines()]
    expected = {r['sample_id']:r['prompt_en'] for r in rows}
    actual = {r['sample_id']:r['prompt'] for r in jobs}
    if len(jobs) != 200 or len(actual) != 200 or actual != expected:
        raise ValueError('vbench200 requires all 200 canonical sample IDs and exact prompt text; use custom for subsets')


def paired(base, candidate):
    bm, cm = read(base/'manifest.json'), read(candidate/'manifest.json')
    if bm.get('mode') != 'baseline' or cm.get('mode') != 'ours':
        raise ValueError('native baseline and Ours candidate required')
    for m in (bm,cm):
        if m.get('status') != 'complete' or m['protocol'] != PROTOCOL:
            raise ValueError('incomplete/fixed protocol mismatch')
        if [r['sample_id'] for r in m['rows']] != [j['sample_id'] for j in m['jobs']]:
            raise ValueError('manifest coverage mismatch')
    for key in ('protocol','jobs','checkpoint_dir','source_manifest','gpu_uuid','profile_sha256'):
        if bm[key] != cm[key]:
            raise ValueError('paired comparison mismatch: ' + key)
    for root, m in ((base,bm),(candidate,cm)):
        actual = {p.name for p in (root/'videos').glob('*.mp4')}
        if actual != {j['sample_id']+'.mp4' for j in m['jobs']}:
            raise ValueError('video coverage mismatch')
        for row in m['rows']:
            sid = row['sample_id']
            for relative, field in ((f'videos/{sid}.mp4','video_sha256'),(f'timings/{sid}.json','timing_sha256')):
                if sha256(root/relative) != row[field]:
                    raise ValueError('artifact SHA mismatch: ' + relative)
            if m['mode'] == 'ours':
                for relative,field in ((f'traces/{sid}.json','trace_sha256'),(f'features/{sid}.pt','feature_sha256')):
                    if sha256(root/relative) != row[field]:
                        raise ValueError('artifact SHA mismatch: ' + relative)
    return bm,cm


def run(args):
    base,candidate = args.baseline.resolve(),args.candidate.resolve()
    bm,cm = paired(base,candidate)
    validate_vbench_jobs(cm['jobs'], args.vbench_mode)
    output = result_dir(args.output, '# Ours4Wan22 paired evaluation\n\nVideoMetrics RGB PSNR/SSIM/LPIPS, VBench scores and paired full-generate/DiT performance. Commands are preserved in commands.json.')
    cache = under(args.metric_models,MODELS)
    cache.mkdir(parents=True,exist_ok=True)
    prompt_map = {j['sample_id']+'.mp4':j['prompt'] for j in cm['jobs']}
    write(output/'prompts.json',prompt_map)
    env = dict(os.environ, PYTHON_BIN=sys.executable, TORCH_HOME=str(cache/'torch'),
               PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''),
               VBENCH_CACHE_DIR=str(cache/'VBench'), HF_HOME=str(cache/'huggingface'),
               XDG_CACHE_HOME=str(cache/'xdg'))
    for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[variable] = '1'
    commands = [['bash', str(OFFICIAL/'VideoMetrics/run_evaluation.sh'),
        '--reference-dir',str(base/'videos'),'--candidate-dir',str(candidate/'videos'),
        '--extension','.mp4','--expected-frames','45','--device',args.device,
        '--model-cache',str(cache/'torch'),'--output-dir',str(output/'video_metrics')]]
    for label,root in (('baseline',base),('candidate',candidate)):
        if args.vbench_mode == 'vbench200':
            # This tool verifies the full fixed Vbench200 identities and coverage.
            command = ['bash',str(OFFICIAL/'VbenchEvaluation/run_vbench200.sh'),str(root/'videos'),str(output/('vbench_'+label)),'1']
        else:
            command = ['bash',str(OFFICIAL/'VbenchEvaluation/run_custom_vbench.sh'),str(root/'videos'),str(output/('vbench_'+label)),str(output/'prompts.json')]
        commands.append(command)
    write(output/'commands.json',commands)
    for i,command in enumerate(commands):
        with (output/f'tool_{i}.log').open('x') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    quality = read(output/'video_metrics/summary.json')
    scores = {}
    for label in ('baseline','candidate'):
        filename = 'vbench200_aggregate_scores.json' if args.vbench_mode == 'vbench200' else 'vbench_custom_aggregate_scores.json'
        score = read(output/('vbench_'+label)/filename)
        value = score['aggregate_scores']['total_score'] if args.vbench_mode == 'vbench200' else score['vbench_score']
        scores[label] = dict(vbench_score=value, details=score)
    write(output/'summary.json',dict(status='complete',protocol=PROTOCOL,quality=quality,
        target_speedup=cm.get('target_speedup'),skip_budget=cm.get('skip_budget'),
        calibration_sha256=cm.get('calibration_sha256'),calibration=cm.get('calibration'),
        vbench_mode=args.vbench_mode,vbench=scores,baseline=bm['rows'],candidate=cm['rows'],
        latency_speedup_ratio_of_sums=sum(r['pipeline_generate_wall_seconds'] for r in bm['rows'])/sum(r['pipeline_generate_wall_seconds'] for r in cm['rows']),
        dit_tflops_speedup_ratio_of_sums=sum(r['estimated_dit_tflops'] for r in bm['rows'])/sum(r['estimated_dit_tflops'] for r in cm['rows'])))
    return output
