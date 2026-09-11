#!/usr/bin/env python3
"""Run the fixed calibrated K23/K29/K35 12x3 five-prompt benchmark."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

PROJECT = Path(__file__).resolve().parents[2]
OFFICIAL = PROJECT.parent
WORKSPACE = PROJECT.parents[2]
sys.path.insert(0, str(PROJECT))
from ours4wan21.contracts import EXP_ROOT, MODEL_ROOT, PROTOCOL, create_result, dump, sha256
from pipeline_lib import (TARGET_K, MODES, TARGETS, condition_name, read_json,
                          validate_generation, validate_quality, vbench_enabled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-name', default='ours21_random3000_12groups_v1')
    parser.add_argument('--run-name', default='ours21_random3000_12groups_v1_vbench5_speed_targets_v1')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--wan21-root', type=Path,
                        default=WORKSPACE / 'data/source/Wan2.1-65386b2')
    parser.add_argument('--checkpoint-dir', type=Path,
                        default=MODEL_ROOT / 'Wan2.1-T2V-1.3B')
    parser.add_argument('--flops-profile', type=Path,
                        default=EXP_ROOT / 'wan21_seacache_threshold_collection_v1/calflops_profile.json')
    parser.add_argument('--subset-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'subset')
    parser.add_argument('--prior-calibration', type=Path,
                        default=EXP_ROOT / 'wan21_seacache_speedup_calibration_v1/analysis/speed_threshold_mapping.calibrated.json')
    parser.add_argument('--target-speedups', type=float, nargs='+', default=list(TARGETS))
    parser.add_argument('--target-k', type=int, nargs='+', default=list(TARGET_K))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-vbench', action='store_true',
                        help='User-requested diagnostic run: retain PSNR/SSIM/LPIPS, omit VBench scoring')
    parser.add_argument('--skip-quality', action='store_true',
                        help='Development only: do not use for a completed formal run')
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = parse_args()
    if 'wan2.2' not in Path(sys.prefix).name.lower():
        raise ValueError('use the Wan2.2 environment')
    if (len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus)
            or not args.gpus):
        raise ValueError('GPU ids must be distinct nonnegative integers')
    targets = tuple(float(value) for value in args.target_speedups)
    if targets != TARGETS:
        raise ValueError(f'formal target set is frozen to {TARGETS}')
    target_k = tuple(int(value) for value in args.target_k)
    if target_k != TARGET_K:
        raise ValueError(f'formal calibrated K set is frozen to {TARGET_K}')

    orchestration = EXP_ROOT / f'{args.suite_name}_orchestration'
    training_complete = read_json(orchestration / 'COMPLETE.json')
    if (training_complete.get('status') != 'complete' or training_complete.get('modes') != 12
            or training_complete.get('epochs_per_mode') != 400):
        raise ValueError('12x400 training orchestration is not complete')

    subset = args.subset_dir.resolve(strict=True)
    prompts = subset / 'prompts.jsonl'
    prompt_map = subset / 'prompt_map.json'
    subset_manifest_path = subset / 'selection_manifest.json'
    subset_manifest = read_json(subset_manifest_path)
    prompt_rows = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
    prompt_ids = [row['sample_id'] for row in prompt_rows]
    if (subset_manifest.get('status') != 'frozen' or prompt_ids !=
            subset_manifest.get('selected_sample_ids') or len(prompt_ids) != 5):
        raise ValueError('five-prompt subset is not frozen or does not match its manifest')
    flops = args.flops_profile.resolve(strict=True)
    profile = read_json(flops)
    if (profile.get('input', {}).get('video_shape_fhw') != [81, 480, 832]
            or profile.get('input', {}).get('transformer_blocks') != 30
            or set(profile.get('component_profiles', {})) != {'t5', 'vae_decode'}):
        raise ValueError('FLOPs profile lacks fixed 81-frame DiT/T5/VAE scope')
    prior_path = args.prior_calibration.resolve(strict=True)
    prior = read_json(prior_path)
    if prior.get('calibration_status') != 'calibrated' or prior.get('model') != PROTOCOL['model']:
        raise ValueError('SeaCache prior calibration is invalid')
    skip_fit = prior.get('skip_fit', {})
    derived = tuple(round((float(skip_fit['slope']) * target +
                           float(skip_fit['intercept'])) / 2) for target in targets)
    if derived != target_k:
        raise ValueError(f'frozen K values {target_k} disagree with prior skip-count fit {derived}')
    target_mapping = [dict(target_speedup=target, skip_budget=skip_budget,
                           fitted_branch_reuse_calls=(float(skip_fit['slope']) * target +
                                                      float(skip_fit['intercept'])),
                           fitted_step_reuse=(float(skip_fit['slope']) * target +
                                              float(skip_fit['intercept'])) / 2,
                           rounding='nearest integer, Python round')
                      for target, skip_budget in zip(targets, target_k)]
    wan21_root = args.wan21_root.resolve(strict=True)
    checkpoint_dir = args.checkpoint_dir.resolve(strict=True)

    checkpoints: dict[str, dict] = {}
    for mode in MODES:
        analysis_dir = EXP_ROOT / f'{args.suite_name}_{mode}_analysis'
        marker = read_json(analysis_dir / 'COMPLETE.json')
        selection = read_json(analysis_dir / 'checkpoint_selection.json')
        selected = (analysis_dir / 'selected_model.pt').resolve(strict=True)
        if (marker.get('status') != 'complete' or selection.get('status') != 'selected'
                or not 301 <= int(selection['checkpoint_epoch']) <= 399
                or sha256(selected) != selection['checkpoint_sha256']):
            raise ValueError(f'invalid post-300 checkpoint selection: {mode}')
        checkpoints[mode] = dict(path=str(selected), sha256=sha256(selected),
                                 epoch=int(selection['checkpoint_epoch']),
                                 min_actor=float(selection['selected']['min_actor']),
                                 mean_normalized_dq=float(selection['selected']['mean_normalized_dq']),
                                 selection_manifest=str(analysis_dir / 'checkpoint_selection.json'),
                                 selection_manifest_sha256=sha256(analysis_dir / 'checkpoint_selection.json'))

    assignments = {gpu: list(MODES[index::len(args.gpus)])
                   for index, gpu in enumerate(args.gpus)}
    config = dict(
        schema='ours4wan21_vbench5_speed_targets_orchestration_v1',
        status='frozen', suite_name=args.suite_name, run_name=args.run_name,
        protocol=PROTOCOL, target_speedups=list(targets), target_k=list(target_k),
        target_k_mapping=target_mapping,
        prompt_subset=dict(path=str(subset), manifest_sha256=sha256(subset_manifest_path),
                           prompts_sha256=sha256(prompts), prompt_map_sha256=sha256(prompt_map),
                           sample_ids=prompt_ids, score_scope=subset_manifest['score_scope'],
                           official_full_vbench_score=False),
        prior=dict(path=str(prior_path), sha256=sha256(prior_path),
                   use='direct target-to-K mapping; no adaptive K scan or per-mode recalibration',
                   skip_fit=prior['skip_fit']),
        flops_profile=dict(path=str(flops), sha256=sha256(flops)),
        wan21_root=str(wan21_root), checkpoint_dir=str(checkpoint_dir),
        gpus=list(args.gpus), assignments={str(key): value for key, value in assignments.items()},
        checkpoints=checkpoints, python=sys.executable,
        quality_enabled=not args.skip_quality, started_at=now())
    root_path = EXP_ROOT / args.run_name
    if args.resume:
        root = root_path.resolve(strict=True)
        existing = read_json(root / 'config.json')
        comparable = lambda value: {key: item for key, item in value.items() if key != 'started_at'}
        if comparable(existing) != comparable(config):
            raise ValueError('resume configuration differs from frozen suite config')
        config = existing
    else:
        root = create_result(root_path,
            '# Ours4Wan21 VBench200 five-prompt speed-target suite\n\n'
            'This suite contains per-GPU native baselines, a fixed prior-calibrated K mapping, '
            '12x3 matched candidates, full component timing/TFLOPs, PSNR/SSIM/LPIPS, and '
            'the explicitly non-official ten-dimension VBench custom-input diagnostic score.')
        for folder in ('logs', 'baselines', 'candidates'):
            (root / folder).mkdir()
        dump(root / 'config.json', config)
        dump(root / 'target_k_mapping.json', dict(
            schema='ours4wan21_prior_calibrated_target_k_v1', status='frozen',
            protocol=PROTOCOL, source=str(prior_path), source_sha256=sha256(prior_path),
            skip_fit=skip_fit, entries=target_mapping,
            method='K = round((branch_reuse_slope * target + intercept) / 2)',
            adaptive_rescan=False,
            limitation='Actual per-mode complete-generate speedups are measured and reported; K is not adjusted afterward.'))

    if args.skip_vbench:
        dump(root / 'VBENCH_SKIPPED_BY_USER.json', dict(status='skipped_by_user',
            video_metrics_enabled=True, reason='User requested no VBench scoring for this diagnostic test'))
    score_vbench = vbench_enabled(root)
    environment = dict(os.environ)
    environment.update(
        OURS4WAN21_WORKSPACE=str(WORKSPACE), OURS4WAN21_EXP_BASE=str(EXP_ROOT),
        EXP_BASE=str(EXP_ROOT), PYTHON_BIN=sys.executable,
        VBENCH_CACHE_DIR=str(MODEL_ROOT / 'VBench'),
        TORCH_HOME=str(MODEL_ROOT / 'torch-cache'),
        HF_HOME=str(MODEL_ROOT / 'VBench/huggingface'),
        XDG_CACHE_HOME=str(MODEL_ROOT / 'VBench/xdg'),
        CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    lock = threading.Lock()
    state = dict(schema='ours4wan21_vbench5_speed_targets_status_v1', status='running',
                 updated_at=now(), stages={}, modes={mode: {} for mode in MODES},
                 vbench_enabled=score_vbench)

    def update(section: str, name: str, **values: object) -> None:
        with lock:
            bucket = state[section].setdefault(name, {})
            bucket.update(values, updated_at=now())
            state['updated_at'] = now()
            dump(root / 'status.json', state)

    def execute(label: str, command: list[str], gpu: int | None = None) -> None:
        log_path = root / 'logs' / f"{label.replace('/', '__')}.log"
        child_env = dict(environment)
        if gpu is not None:
            child_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        update('stages', label, status='running', gpu=gpu, command=command, log=str(log_path))
        with log_path.open('ab') as stream:
            completed = subprocess.run(command, cwd=PROJECT, env=child_env, stdout=stream,
                                       stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            update('stages', label, status='failed', returncode=completed.returncode)
            raise RuntimeError(f'{label} failed; inspect {log_path}')
        update('stages', label, status='complete', returncode=0)

    def baseline_dir(gpu: int) -> Path:
        return root / 'baselines' / f'gpu{gpu}'

    def candidate_dir(mode: str, skip_budget: int) -> Path:
        return root / 'candidates' / condition_name(mode, skip_budget)

    def generate_baseline(gpu: int) -> None:
        output = baseline_dir(gpu)
        if output.exists():
            validate_generation(output, prompt_ids=prompt_ids, method='baseline')
        else:
            execute(f'baseline_gpu{gpu}', [sys.executable, 'generate.py', '--baseline',
                '--wan21-root', str(wan21_root), '--checkpoint-dir', str(checkpoint_dir),
                '--prompts', str(prompts), '--flops-profile', str(flops),
                '--output-dir', str(output), '--result-parent', str(root)], gpu)
            validate_generation(output, prompt_ids=prompt_ids, method='baseline')

    def generate_candidate(mode: str, skip_budget: int, gpu: int) -> float:
        output = candidate_dir(mode, skip_budget)
        if output.exists():
            manifest = validate_generation(output, prompt_ids=prompt_ids, method='ours',
                                           mode=mode, skip_budget=skip_budget)
        else:
            execute(f'generate/{mode}/K{skip_budget:02d}', [sys.executable, 'generate.py',
                '--wan21-root', str(wan21_root), '--checkpoint-dir', str(checkpoint_dir),
                '--policy-checkpoint', checkpoints[mode]['path'], '--state-mode', mode,
                '--skip-budget', str(skip_budget), '--prompts', str(prompts),
                '--flops-profile', str(flops), '--output-dir', str(output),
                '--result-parent', str(root)], gpu)
            manifest = validate_generation(output, prompt_ids=prompt_ids, method='ours',
                                           mode=mode, skip_budget=skip_budget)
        if manifest['gpu_uuid'] != read_json(baseline_dir(gpu) / 'run.json')['gpu_uuid']:
            raise ValueError(f'{mode} K{skip_budget} is not paired to the same physical GPU baseline')
        performance = output / 'performance.json'
        if not performance.exists():
            execute(f'summarize/{mode}/K{skip_budget:02d}', [sys.executable, 'evaluate.py',
                'summarize', '--baseline-dir', str(baseline_dir(gpu)),
                '--candidate-dir', str(output), '--profile', str(flops)], gpu)
        payload = read_json(performance)
        return float(payload['latency_speedup_ratio_of_sums'])

    def run_parallel(label: str, jobs: dict[object, tuple]) -> dict:
        failures, values = [], {}
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = {pool.submit(*job): key for key, job in jobs.items()}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    values[key] = future.result()
                except BaseException as error:
                    failures.append(dict(job=str(key), error=repr(error)))
        if failures:
            state['status'] = 'failed'
            state['failures'] = failures
            dump(root / 'status.json', state)
            dump(root / 'FAILED.json', dict(stage=label, failures=failures, failed_at=now()))
            raise RuntimeError(f'{label} failed: {failures}')
        return values

    run_parallel('baselines', {gpu: (generate_baseline, gpu) for gpu in args.gpus})

    def generation_queue(gpu: int, modes: list[str]) -> None:
        for mode in modes:
            achieved = {}
            for target, skip_budget in zip(targets, target_k):
                achieved[str(target)] = generate_candidate(mode, skip_budget, gpu)
            update('modes', mode, generation='complete', gpu=gpu,
                   target_k=dict(zip((str(value) for value in targets), target_k)),
                   achieved_speedup=achieved)
    run_parallel('generation', {gpu: (generation_queue, gpu, modes)
                                for gpu, modes in assignments.items()})

    def run_vbench(videos: Path, output: Path, label: str, gpu: int) -> None:
        execute(label, ['bash', str(OFFICIAL / 'VbenchEvaluation/run_custom_vbench.sh'),
                        str(videos), str(output), str(prompt_map)], gpu)

    def baseline_vbench(gpu: int) -> None:
        base = baseline_dir(gpu)
        aggregate = base / 'quality/vbench_custom/vbench_custom_aggregate_scores.json'
        marker = base / 'quality/VBENCH_COMPLETE.json'
        if marker.exists():
            payload = read_json(aggregate)
            if payload.get('official_full_vbench_score') is not False:
                raise ValueError(f'baseline VBench scope mismatch on GPU {gpu}')
            return
        quality = base / 'quality'
        if quality.exists():
            raise RuntimeError(f'incomplete baseline VBench directory requires inspection: {quality}')
        quality.mkdir()
        run_vbench(base / 'videos', quality / 'vbench_custom', f'vbench_baseline_gpu{gpu}', gpu)
        dump(marker, dict(status='complete', aggregate_sha256=sha256(aggregate)))

    def candidate_quality(mode: str, skip_budget: int, gpu: int) -> None:
        candidate = candidate_dir(mode, skip_budget)
        if (candidate / 'quality/COMPLETE.json').exists():
            validate_quality(candidate, len(prompt_ids), require_vbench=score_vbench)
            return
        quality = candidate / 'quality'
        if quality.exists():
            raise RuntimeError(f'incomplete quality directory requires inspection: {quality}')
        quality.mkdir()
        execute(f'video_metrics/{mode}/K{skip_budget:02d}', [sys.executable,
            str(OFFICIAL / 'VideoMetrics/evaluate.py'), '--reference-dir',
            str(baseline_dir(gpu) / 'videos'), '--candidate-dir', str(candidate / 'videos'),
            '--expected-frames', '81', '--device', 'cuda:0', '--output-dir',
            str(quality / 'video_metrics')], gpu)
        if score_vbench:
            run_vbench(candidate / 'videos', quality / 'vbench_custom',
                       f'vbench/{mode}/K{skip_budget:02d}', gpu)
        video_summary = quality / 'video_metrics/summary.json'
        vbench_summary = quality / 'vbench_custom/vbench_custom_aggregate_scores.json'
        dump(quality / 'COMPLETE.json', dict(status='quality_complete', prompt_count=len(prompt_ids),
            frames=len(prompt_ids) * 81, video_metrics_sha256=sha256(video_summary),
            vbench_custom_sha256=sha256(vbench_summary) if score_vbench else None,
            vbench_status='complete' if score_vbench else 'skipped_by_user',
            official_full_vbench_score=False))
        validate_quality(candidate, len(prompt_ids), require_vbench=score_vbench)

    if not args.skip_quality:
        if score_vbench:
            run_parallel('baseline_vbench', {gpu: (baseline_vbench, gpu) for gpu in args.gpus})

        def quality_queue(gpu: int, modes: list[str]) -> None:
            for mode in modes:
                for skip_budget in target_k:
                    candidate_quality(mode, skip_budget, gpu)
                    update('modes', mode, quality_k=skip_budget, quality_status='complete', gpu=gpu)
        run_parallel('quality', {gpu: (quality_queue, gpu, modes)
                                 for gpu, modes in assignments.items()})
    else:
        dump(root / 'DEVELOPMENT_ONLY_NO_QUALITY.json', dict(status='skipped_by_cli'))
        return

    dump(root / 'QUALITY_COMPLETE.json', dict(status='complete', modes=12, targets=3,
        candidate_conditions=36, prompt_count=5,
        score_scope=subset_manifest['score_scope'] if score_vbench else 'paired_video_metrics_only',
        official_full_vbench_score=False, vbench_enabled=score_vbench, completed_at=now()))
    if (root / 'analysis/COMPLETE.json').is_file():
        analysis_marker = read_json(root / 'analysis/COMPLETE.json')
        if (analysis_marker.get('status') != 'complete'
                or sha256(root / 'analysis/results_long.csv') != analysis_marker['results_sha256']):
            raise ValueError('existing analysis failed completion validation')
    elif (root / 'analysis').exists():
        raise RuntimeError('partial analysis directory requires inspection before resume')
    else:
        execute('analyze', [sys.executable,
            str(Path(__file__).resolve().parent / 'analyze_results.py'), '--result-root', str(root)])
    if not (root / 'analysis/artifact.json').is_file():
        execute('build_report_artifact', [sys.executable,
            str(Path(__file__).resolve().parent / 'build_report.py'), '--result-root', str(root)])

    plugin_candidates = sorted((WORKSPACE / '.codex/plugins/cache/openai-curated-remote/data-analytics').glob('*'))
    plugin = next((path for path in reversed(plugin_candidates) if (path / 'package.json').is_file()), None)
    if plugin is None:
        raise RuntimeError('data-analytics report packager is unavailable')
    if not (root / 'analysis/report.html').is_file():
        execute('deliver_report', ['npm', '--prefix', str(plugin), 'run', 'report:deliver', '--',
            '--input', str(root / 'analysis/artifact.json'), '--output',
            str(root / 'analysis/report.html')])
    if not (root / 'analysis/report.html').is_file():
        raise RuntimeError('report delivery did not create report.html')
    state['status'] = 'complete'
    state['updated_at'] = now()
    dump(root / 'status.json', state)
    dump(root / 'COMPLETE.json', dict(status='complete', modes=12, targets=3,
        candidate_conditions=36, prompt_count=5, vbench_enabled=score_vbench,
        analysis_sha256=sha256(root / 'analysis/results_long.csv'),
        report_sha256=sha256(root / 'analysis/report.html'), completed_at=now()))


if __name__ == '__main__':
    main()
