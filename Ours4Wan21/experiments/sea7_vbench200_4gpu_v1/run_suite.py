#!/usr/bin/env python3
"""Queue and run SEA7 e328 K23/K29/K35 on 200 prompts with four GPUs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import csv
import fcntl
import html
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'experiments/vbench5_speed_targets_v1'))
from ours4wan21.contracts import EXP_ROOT, MODEL_ROOT, OFFICIAL, PROTOCOL, WORKSPACE, create_result, dump, sha256
from pipeline_lib import read_json, validate_generation

NAME = 'ours21_sea7_e328_vbench200_4gpu_v1'
PREVIOUS = 'ours21_random3000_12groups_v1_vbench5_speed_targets_v1'
KS = (23, 29, 35)
GPUS = (0, 1, 2, 3)
LABELS = ('baseline', 'K23', 'K29', 'K35')


def now():
    return datetime.now(timezone.utc).isoformat()


def partition(rows):
    ids = [r['sample_id'] for r in rows]
    if len(ids) != 200 or len(set(ids)) != 200 or any(not r.get('prompt_en') for r in rows):
        raise ValueError('expected exactly 200 unique prompts')
    return {g: rows[g::4] for g in GPUS}


def prerequisite_complete(root):
    marker = root / 'COMPLETE.json'
    if not marker.is_file():
        return False
    d = read_json(marker)
    if (d.get('status') != 'complete' or d.get('modes') != 12 or d.get('targets') != 3
            or d.get('candidate_conditions') != 36 or d.get('prompt_count') != 5):
        raise ValueError('invalid five-prompt completion marker')
    for field, relative in [('analysis_sha256', 'analysis/results_long.csv'),
                             ('report_sha256', 'analysis/report.html')]:
        if sha256(root / relative) != d[field]:
            raise ValueError('five-prompt completion hash mismatch')
    if read_json(root / 'analysis/VALIDATION.json').get('status') != 'pass':
        raise ValueError('five-prompt validation did not pass')
    return True


def summarize(base, candidates):
    if len(base) != 200 or len(candidates) != 200:
        raise ValueError('need all 200 paired performance rows')
    expected = {r['sample_id'] for r in base}
    if len(expected) != 200 or {r['sample_id'] for r in candidates} != expected:
        raise ValueError('performance identity mismatch')
    for row in base + candidates:
        if any(not math.isfinite(v) for v in row.values() if isinstance(v, (int, float))):
            raise ValueError('nonfinite performance')
        if row['generate_seconds'] <= 0:
            raise ValueError('nonpositive latency')
    fields = ('generate_seconds', 'dit_cuda_seconds', 't5_cuda_seconds', 'vae_decode_cuda_seconds',
              'dit_tflops', 'estimated_t5_tflops_per_video', 'estimated_vae_decode_tflops_per_video',
              'predictor_tflops', 'predictor_network_cuda_seconds', 'predictor_decision_wall_seconds',
              'latent_feature_wall_seconds')
    result = {f'candidate_{key}_mean': statistics.fmean(r[key] for r in candidates) for key in fields}
    result['baseline_generate_seconds_mean'] = statistics.fmean(r['generate_seconds'] for r in base)
    result['baseline_dit_tflops_mean'] = statistics.fmean(r['dit_tflops'] for r in base)
    result['latency_speedup'] = sum(r['generate_seconds'] for r in base) / sum(r['generate_seconds'] for r in candidates)
    result['dit_tflops_speedup'] = sum(r['dit_tflops'] for r in base) / sum(r['dit_tflops'] for r in candidates)
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if 'wan2.2' not in Path(sys.prefix).name.lower():
        raise ValueError('use Wan2.2 environment')
    root = EXP_ROOT / NAME
    prompts_path = OFFICIAL / 'Vbench200/prompts.jsonl'
    rows = [json.loads(line) for line in prompts_path.read_text().splitlines() if line.strip()]
    shards = partition(rows)
    selection_path = EXP_ROOT / 'ours21_random3000_12groups_v1_sea7_analysis/checkpoint_selection.json'
    selection = read_json(selection_path)
    checkpoint = Path(selection['checkpoint']).resolve(strict=True)
    if selection['checkpoint_epoch'] != 328 or sha256(checkpoint) != selection['checkpoint_sha256']:
        raise ValueError('SEA7 selected e328 checkpoint identity mismatch')
    profile_path = EXP_ROOT / 'wan21_seacache_threshold_collection_v1/calflops_profile.json'
    profile = read_json(profile_path)
    if profile['input']['video_shape_fhw'] != [81, 480, 832] or profile['input']['transformer_blocks'] != 30:
        raise ValueError('wrong model FLOPs profile')
    config = dict(schema='sea7_vbench200_4gpu_v1', protocol=PROTOCOL, mode='sea7',
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), selected_epoch=328,
        prompt_source=str(prompts_path), prompts_sha256=sha256(prompts_path),
        targets=[1.8, 2.4, 3.0], skip_budgets=list(KS), adaptive_rescan=False,
        gpu_ids=list(GPUS), shard_ids={str(g): [r['sample_id'] for r in rs] for g, rs in shards.items()},
        flops_profile=str(profile_path), flops_profile_sha256=sha256(profile_path),
        prerequisite=str(EXP_ROOT / PREVIOUS), vbench_enabled=True,
        vbench_full_info_sha256=sha256(OFFICIAL / 'Vbench200/VBench200_full_info.json'),
        vbench_dimensions_sha256=sha256(OFFICIAL / 'VbenchEvaluation/dimensions.json'),
        baseline_policy='new matched native baseline, same prompt and physical GPU as each candidate')
    if args.resume:
        if read_json(root / 'config.json') != config:
            raise ValueError('resume inputs changed from frozen configuration')
    else:
        create_result(root, '# SEA7 VBench200 four-GPU suite\n\nSee config.json and status.json. prompts/ holds frozen shards; shards/ holds per-GPU native and K results; merged/ contains video symlinks; vbench/, logs/, analysis/ hold evaluation and reports.')
        dump(root / 'config.json', config)
        for folder in ('prompts', 'shards', 'merged', 'vbench', 'logs', 'analysis'):
            (root / folder).mkdir()
            (root / folder / 'README.md').write_text(f'# {folder}\n\nSEA7 VBench200 four-GPU suite {folder} artifacts; see ../README.md.\n')
        for g, rs in shards.items():
            (root / 'prompts' / f'gpu{g}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rs))
        dump(root / 'status.json', dict(status='queued', waiting_for=config['prerequisite'], updated_at=now()))
    for g, rs in shards.items():
        actual = [json.loads(line) for line in (root / 'prompts' / f'gpu{g}.jsonl').read_text().splitlines()]
        if actual != rs:
            raise ValueError('frozen prompt shard changed')
    if args.prepare_only:
        print(json.dumps(dict(status='queued', root=str(root), checkpoint_epoch=328, candidates=600)))
        return

    with (root / 'queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'COMPLETE.json').exists():
            print('suite already complete')
            return
        state = dict(status='queued', stages={}, waiting_for=config['prerequisite'], updated_at=now())
        lock = threading.Lock()
        def update(**values):
            with lock:
                state.update(values, updated_at=now())
                dump(root / 'status.json', state)
        env = dict(os.environ, OURS4WAN21_WORKSPACE=str(WORKSPACE), OURS4WAN21_EXP_BASE=str(EXP_ROOT),
            EXP_BASE=str(EXP_ROOT), PYTHON_BIN=sys.executable, PYTHONDONTWRITEBYTECODE='1',
            CUDA_DEVICE_ORDER='PCI_BUS_ID', TORCH_HOME=str(MODEL_ROOT / 'torch-cache'),
            VBENCH_CACHE_DIR=str(MODEL_ROOT / 'VBench'), HF_HOME=str(MODEL_ROOT / 'VBench/huggingface'),
            XDG_CACHE_HOME=str(MODEL_ROOT / 'VBench/xdg'), OPENBLAS_NUM_THREADS='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        def execute(label, command, gpu):
            with lock:
                state['stages'][label] = dict(status='running', gpu=gpu, started_at=now())
                dump(root / 'status.json', state)
            with (root / 'logs' / f'{label}.log').open('ab') as log:
                subprocess.run(command, cwd=PROJECT, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            with lock:
                state['stages'][label].update(status='complete', completed_at=now())
                dump(root / 'status.json', state)
        def parallel(function, values):
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(function, values))
        def folder(g, label):
            return root / 'shards' / f'gpu{g}' / label
        def generation(g):
            ids = config['shard_ids'][str(g)]
            for label in LABELS:
                d = folder(g, label)
                command = [sys.executable, 'generate.py', '--wan21-root', str(WORKSPACE / 'data/source/Wan2.1-65386b2'),
                    '--checkpoint-dir', str(MODEL_ROOT / 'Wan2.1-T2V-1.3B'), '--prompts',
                    str(root / 'prompts' / f'gpu{g}.jsonl'), '--flops-profile', str(profile_path),
                    '--output-dir', str(d), '--result-parent', str(root)]
                k = None if label == 'baseline' else int(label[1:])
                command += ['--baseline'] if k is None else ['--policy-checkpoint', str(checkpoint), '--state-mode', 'sea7', '--skip-budget', str(k)]
                if not d.exists():
                    execute(f'generate_gpu{g}_{label}', command, g)
                m = validate_generation(d, prompt_ids=ids, method='baseline' if k is None else 'ours', mode='sea7', skip_budget=k)
                if m['protocol'] != PROTOCOL or m['flops_profile_sha256'] != config['flops_profile_sha256']:
                    raise ValueError('generation protocol/profile mismatch')
                if k is not None:
                    bm = read_json(folder(g, 'baseline') / 'run.json')
                    if m['gpu_uuid'] != bm['gpu_uuid'] or m['policy_sha256'] != config['checkpoint_sha256']:
                        raise ValueError('physical GPU/checkpoint pairing mismatch')
                    for sid in ids:
                        trace = read_json(d / 'traces' / f'{sid}.json')
                        if trace['step_reuse'] != k or trace['total_steps'] != 50:
                            raise ValueError('Exact-K trace mismatch')
                    if not (d / 'performance.json').exists():
                        execute(f'performance_gpu{g}_{label}', [sys.executable, 'evaluate.py', 'summarize',
                            '--baseline-dir', str(folder(g, 'baseline')), '--candidate-dir', str(d),
                            '--profile', str(profile_path)], g)
        def video_quality(g):
            for k in KS:
                d = folder(g, f'K{k}')
                out = d / 'quality'
                if not out.exists():
                    execute(f'quality_gpu{g}_K{k}', [sys.executable, str(OFFICIAL / 'VideoMetrics/evaluate.py'),
                        '--reference-dir', str(folder(g, 'baseline') / 'videos'), '--candidate-dir', str(d / 'videos'),
                        '--expected-frames', '81', '--device', 'cuda:0', '--output-dir', str(out)], g)
                q = read_json(out / 'summary.json')
                if q['video_count'] != 50 or q['frame_count_total'] != 4050:
                    raise ValueError('incomplete shard quality')
        def vbench(job):
            g, label = job
            output = root / 'vbench' / label
            if not output.exists():
                execute(f'vbench_{label}', ['bash', str(OFFICIAL / 'VbenchEvaluation/run_vbench200.sh'),
                    str(root / 'merged' / label / 'videos'), str(output), '1'], g)
            scores = read_json(output / 'vbench200_aggregate_scores.json')
            expected_dims = set(read_json(OFFICIAL / 'VbenchEvaluation/dimensions.json')['dimensions'])
            if set(scores['raw_dimension_scores']) != expected_dims or scores['official_full_vbench_score'] is not False:
                raise ValueError('incomplete or wrong VBench200 score scope')

        try:
            update(status='queued')
            while not prerequisite_complete(Path(config['prerequisite'])):
                if (Path(config['prerequisite']) / 'FAILED.json').exists():
                    raise RuntimeError('prerequisite failed; cannot start full evaluation')
                time.sleep(60)
            update(status='waiting_for_gpus')
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
                time.sleep(30)
            update(status='generating')
            parallel(generation, GPUS)
            update(status='video_quality')
            parallel(video_quality, GPUS)
            for label in LABELS:
                videos = root / 'merged' / label / 'videos'
                videos.mkdir(parents=True, exist_ok=True)
                for g in GPUS:
                    for sid in config['shard_ids'][str(g)]:
                        source = folder(g, label) / 'videos' / f'{sid}.mp4'
                        target = videos / source.name
                        if target.is_symlink() and target.resolve() == source.resolve():
                            continue
                        if target.exists() or target.is_symlink():
                            raise ValueError('merged video conflict')
                        target.symlink_to(source)
                if len(list(videos.glob('*.mp4'))) != 200:
                    raise ValueError('incomplete merged videos')
            update(status='vbench')
            parallel(vbench, list(zip(GPUS, LABELS)))
            update(status='analyzing')
            results, per_video, evidence = [], [], {}
            baseline_scores = read_json(root / 'vbench/baseline/vbench200_aggregate_scores.json')['aggregate_scores']
            for k, target_speed in zip(KS, config['targets']):
                br, cr, quality_rows = [], [], []
                for g in GPUS:
                    d = folder(g, f'K{k}')
                    p = read_json(d / 'performance.json')
                    br.extend(p['baseline']); cr.extend(p['candidate'])
                    with (d / 'quality/per_video.csv').open() as stream:
                        qs = list(csv.DictReader(stream))
                    if {q['video_id'] for q in qs} != set(config['shard_ids'][str(g)]):
                        raise ValueError('quality prompt identity mismatch')
                    quality_rows.extend(qs)
                    by_id = {q['video_id']: q for q in qs}
                    for b, c in zip(p['baseline'], p['candidate']):
                        if b['sample_id'] != c['sample_id']:
                            raise ValueError('pair order mismatch')
                        q = by_id[c['sample_id']]
                        per_video.append(dict(k=k, gpu=g, sample_id=c['sample_id'],
                            baseline_seconds=b['generate_seconds'], candidate_seconds=c['generate_seconds'],
                            psnr_rgb_db=q['psnr_rgb_db'], ssim_rgb=q['ssim_rgb'], lpips_alex_v0_1_spatial=q['lpips_alex_v0_1_spatial']))
                    for name in ('performance.json', 'quality/summary.json', 'quality/per_video.csv'):
                        evidence[str((d / name).relative_to(root))] = sha256(d / name)
                scores = read_json(root / 'vbench' / f'K{k}' / 'vbench200_aggregate_scores.json')['aggregate_scores']
                result = dict(k=k, target_speedup=target_speed, checkpoint_epoch=328, **summarize(br, cr))
                for metric in ('psnr_rgb_db', 'ssim_rgb', 'lpips_alex_v0_1_spatial'):
                    result[metric] = statistics.fmean(float(q[metric]) for q in quality_rows)
                result.update(vbench200_total_score=scores['total_score'], vbench200_quality_score=scores['quality_score'],
                    vbench200_semantic_score=scores['semantic_score'], baseline_vbench200_total_score=baseline_scores['total_score'])
                if any(not math.isfinite(v) for v in result.values() if isinstance(v, (float, int))):
                    raise ValueError('nonfinite result')
                results.append(result)
            for label in LABELS:
                path = root / 'vbench' / label / 'vbench200_aggregate_scores.json'
                evidence[str(path.relative_to(root))] = sha256(path)
            out = root / 'analysis'
            write_csv(out / 'results.csv', results)
            write_csv(out / 'per_video.csv', per_video)
            dump(out / 'VALIDATION.json', dict(status='pass', conditions=3, baseline_videos=200,
                candidate_videos=600, quality_frames=48600, dimensions=16, config_sha256=sha256(root / 'config.json'),
                evidence_sha256=evidence, official_full_vbench_score=False))
            report = ['# SEA7 e328 VBench200 四卡测试', '',
                'K23/K29/K35；200条固定prompt、单seed42；同物理GPU baseline配对。',
                'VBench使用standard模式16维和官方归一化加权公式，结果为VBench200 subset score。',
                '完整generate时间不含加载、warmup、视频保存或质量评测；组件时间嵌套其中。', '',
                '|K|实测加速|秒/视频|PSNR dB|SSIM|LPIPS|VBench200 %|',
                '|---|---:|---:|---:|---:|---:|---:|']
            for r in results:
                report.append(f"|{r['k']}|{r['latency_speedup']:.4f}|{r['candidate_generate_seconds_mean']:.3f}|{r['psnr_rgb_db']:.4f}|{r['ssim_rgb']:.5f}|{r['lpips_alex_v0_1_spatial']:.5f}|{100*r['vbench200_total_score']:.3f}|")
            report.extend(['', '全部组件计时和TFLOPs见 results.csv；600条配对指标见 per_video.csv。',
                '单seed、200提示词结果不等于官方完整VBench排行榜成绩。'])
            (out / 'RESULTS.md').write_text('\n'.join(report) + '\n')
            header = ''.join(f'<th>{html.escape(k)}</th>' for k in results[0])
            body = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in r.values()) + '</tr>' for r in results)
            (out / 'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>SEA7 VBench200</title><style>body{font-family:system-ui;margin:2rem}td,th{padding:.5rem;border:1px solid #ddd}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>SEA7 e328 VBench200</h1><pre>' + html.escape('\n'.join(report[2:6])) + '</pre><div style="overflow:auto"><table><tr>' + header + '</tr>' + body + '</table></div><p><a href="results.csv">完整指标 CSV</a> · <a href="per_video.csv">逐视频 CSV</a> · <a href="VALIDATION.json">验收记录</a></p></html>')
            update(status='complete')
            dump(root / 'COMPLETE.json', dict(status='complete', completed_at=now(), conditions=3,
                prompts=200, results_sha256=sha256(out / 'results.csv'), report_sha256=sha256(out / 'report.html')))
        except BaseException as error:
            update(status='failed', error=repr(error))
            dump(root / 'FAILED.json', dict(error=repr(error), failed_at=now()))
            raise


if __name__ == '__main__':
    main()
