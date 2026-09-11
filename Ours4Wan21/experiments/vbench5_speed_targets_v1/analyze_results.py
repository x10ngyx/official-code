#!/usr/bin/env python3
"""Fail-closed audit and tidy readout for the 12x3 five-prompt benchmark."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ours4wan21.contracts import PROTOCOL, dump, sha256
from pipeline_lib import MODES, SHORT, TARGETS, read_json, target_label, validate_generation, validate_quality, vbench_enabled


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError(f'invalid {key}')
    return statistics.fmean(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root.resolve(strict=True)
    config = read_json(root / 'config.json')
    score_vbench = vbench_enabled(root)
    marker = read_json(root / 'QUALITY_COMPLETE.json')
    if (config.get('schema') != 'ours4wan21_vbench5_speed_targets_orchestration_v1'
            or config.get('protocol') != PROTOCOL or config.get('target_speedups') != list(TARGETS)
            or marker.get('status') != 'complete' or marker.get('candidate_conditions') != 36):
        raise ValueError('suite configuration or quality completion is invalid')
    prompt_ids = list(config['prompt_subset']['sample_ids'])
    if len(prompt_ids) != 5:
        raise ValueError('formal suite requires five prompts')
    out = root / 'analysis'
    if out.exists():
        raise FileExistsError(f'refusing to overwrite analysis: {out}')
    out.mkdir()

    gpu_for_mode = {}
    for gpu, modes in config['assignments'].items():
        for mode in modes:
            gpu_for_mode[mode] = int(gpu)
    if set(gpu_for_mode) != set(MODES):
        raise ValueError('mode/GPU assignments are incomplete')

    mapping_path = root / 'target_k_mapping.json'
    mapping = read_json(mapping_path)
    if (mapping.get('status') != 'frozen' or mapping.get('adaptive_rescan') is not False
            or len(mapping.get('entries', [])) != 3):
        raise ValueError('fixed target-to-K mapping is invalid')
    target_to_k = {float(row['target_speedup']): int(row['skip_budget'])
                   for row in mapping['entries']}
    if tuple(target_to_k[target] for target in TARGETS) != (23, 29, 35):
        raise ValueError('formal target-to-K mapping changed')
    evidence_hashes: dict[str, str] = {
        str(mapping_path.relative_to(root)): sha256(mapping_path)}
    rows: list[dict] = []
    per_prompt: list[dict] = []
    mapping_rows = [dict(target_speedup=row['target_speedup'],
                         skip_budget=row['skip_budget'],
                         fitted_branch_reuse_calls=row['fitted_branch_reuse_calls'],
                         fitted_step_reuse=row['fitted_step_reuse'],
                         adaptive_rescan=False,
                         calibration_source_sha256=mapping['source_sha256'])
                    for row in mapping['entries']]
    for mode in MODES:
        gpu = gpu_for_mode[mode]
        baseline_dir = root / 'baselines' / f'gpu{gpu}'
        baseline_manifest = validate_generation(baseline_dir, prompt_ids=prompt_ids, method='baseline')
        baseline_vbench_path = baseline_dir / 'quality/vbench_custom/vbench_custom_aggregate_scores.json'
        baseline_vbench = read_json(baseline_vbench_path) if score_vbench else {}
        if score_vbench and baseline_vbench.get('official_full_vbench_score') is not False:
            raise ValueError(f'baseline VBench scope mismatch: GPU {gpu}')
        for target in TARGETS:
            skip_budget = target_to_k[target]
            candidate_dir = root / 'candidates' / mode / f'K{skip_budget:02d}'
            manifest = validate_generation(candidate_dir, prompt_ids=prompt_ids, method='ours',
                                           mode=mode, skip_budget=skip_budget)
            validate_quality(candidate_dir, len(prompt_ids), require_vbench=score_vbench)
            checkpoint = config['checkpoints'][mode]
            if (manifest.get('policy_sha256') != checkpoint['sha256']
                    or manifest.get('gpu_uuid') != baseline_manifest.get('gpu_uuid')):
                raise ValueError(f'checkpoint or physical GPU mismatch: {mode} {target}')

            performance_path = candidate_dir / 'performance.json'
            performance = read_json(performance_path)
            baseline = performance['baseline']
            candidate = performance['candidate']
            if ([row['sample_id'] for row in baseline] != prompt_ids
                    or [row['sample_id'] for row in candidate] != prompt_ids):
                raise ValueError(f'performance prompt mismatch: {mode} {target}')
            actual_speedup = sum(float(row['generate_seconds']) for row in baseline) / sum(
                float(row['generate_seconds']) for row in candidate)
            if not math.isclose(actual_speedup, float(performance['latency_speedup_ratio_of_sums']),
                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'performance speedup mismatch: {mode} {target}')
            video_path = candidate_dir / 'quality/video_metrics/summary.json'
            video = read_json(video_path)
            metrics = video['metrics']
            vbench_path = candidate_dir / 'quality/vbench_custom/vbench_custom_aggregate_scores.json'
            vbench = read_json(vbench_path) if score_vbench else {}
            predictor = performance['predictor_overhead']
            row = dict(
                mode=mode, label=SHORT[mode],
                kind='control' if mode in ('scalar5', 'sea7') else 'feature',
                target_speedup=target, target_label=target_label(target), skip_budget=skip_budget,
                physical_gpu_id=gpu, gpu_uuid=manifest['gpu_uuid'],
                selected_epoch=checkpoint['epoch'], selected_min_actor_agreement=checkpoint['min_actor'],
                selected_mean_normalized_dq=checkpoint['mean_normalized_dq'],
                target_k_source='prior SeaCache branch-reuse-count fit',
                adaptive_k_rescan=False,
                achieved_latency_speedup=actual_speedup,
                target_relative_error=abs(actual_speedup - target) / target,
                baseline_generate_seconds_mean=mean(baseline, 'generate_seconds'),
                candidate_generate_seconds_mean=mean(candidate, 'generate_seconds'),
                baseline_t5_cuda_seconds_mean=mean(baseline, 't5_cuda_seconds'),
                candidate_t5_cuda_seconds_mean=mean(candidate, 't5_cuda_seconds'),
                baseline_dit_cuda_seconds_mean=mean(baseline, 'dit_cuda_seconds'),
                candidate_dit_cuda_seconds_mean=mean(candidate, 'dit_cuda_seconds'),
                baseline_vae_cuda_seconds_mean=mean(baseline, 'vae_decode_cuda_seconds'),
                candidate_vae_cuda_seconds_mean=mean(candidate, 'vae_decode_cuda_seconds'),
                baseline_dit_tflops_mean=mean(baseline, 'dit_tflops'),
                candidate_dit_tflops_mean=mean(candidate, 'dit_tflops'),
                dit_tflops_speedup=performance['dit_tflops_speedup_ratio_of_sums'],
                t5_tflops_per_video=mean(candidate, 'estimated_t5_tflops_per_video'),
                vae_tflops_per_video=mean(candidate, 'estimated_vae_decode_tflops_per_video'),
                predictor_calls_mean=predictor['predictor_call_count_mean_per_video'],
                predictor_tflops_mean=predictor['predictor_tflops_mean_per_video'],
                predictor_cuda_seconds_mean=predictor['predictor_network_cuda_seconds_mean_per_video'],
                predictor_decision_seconds_mean=predictor['predictor_decision_wall_seconds_mean_per_video'],
                latent_feature_wall_seconds_mean=performance['latent_feature_wall_seconds_total'] / len(prompt_ids),
                latent_feature_tflops=None,
                psnr_rgb_db=metrics['psnr_rgb_db']['mean'],
                ssim_rgb=metrics['ssim_rgb']['mean'],
                lpips_alex_v0_1_spatial=metrics['lpips_alex_v0_1_spatial']['mean'],
                vbench_custom_score=vbench.get('vbench_score'),
                vbench_custom_score_percent=vbench.get('vbench_score_percent'),
                baseline_vbench_custom_score=baseline_vbench.get('vbench_score'),
                vbench_custom_delta=(vbench['vbench_score'] - baseline_vbench['vbench_score']) if score_vbench else None,
                vbench_status='complete' if score_vbench else 'skipped_by_user',
                prompt_count=len(prompt_ids), frames_per_video=81,
                official_full_vbench_score=False,
                candidate_dir=str(candidate_dir.relative_to(root)))
            if not all(math.isfinite(float(value)) for key, value in row.items()
                       if isinstance(value, (int, float)) and not isinstance(value, bool)):
                raise ValueError(f'nonfinite aggregate result: {mode} {target}')
            rows.append(row)

            quality_rows = {}
            with (candidate_dir / 'quality/video_metrics/per_video.csv').open(newline='') as stream:
                quality_rows = {item['video_id']: item for item in csv.DictReader(stream)}
            if set(quality_rows) != set(prompt_ids):
                raise ValueError(f'per-video quality rows mismatch: {mode} {target}')
            by_base = {item['sample_id']: item for item in baseline}
            by_candidate = {item['sample_id']: item for item in candidate}
            for sample_id in prompt_ids:
                b, c, q = by_base[sample_id], by_candidate[sample_id], quality_rows[sample_id]
                per_prompt.append(dict(mode=mode, label=SHORT[mode], target_speedup=target,
                    skip_budget=skip_budget, sample_id=sample_id,
                    baseline_generate_seconds=b['generate_seconds'],
                    candidate_generate_seconds=c['generate_seconds'],
                    prompt_latency_speedup=float(b['generate_seconds']) / float(c['generate_seconds']),
                    candidate_dit_tflops=c['dit_tflops'],
                    psnr_rgb_db=q['psnr_rgb_db'], ssim_rgb=q['ssim_rgb'],
                    lpips_alex_v0_1_spatial=q['lpips_alex_v0_1_spatial']))
            for evidence in (performance_path, video_path, *([vbench_path] if score_vbench else [])):
                evidence_hashes[str(evidence.relative_to(root))] = sha256(evidence)

    if len(rows) != 36 or len(per_prompt) != 180:
        raise ValueError('expected exactly 36 aggregate and 180 prompt-level rows')
    if {(row['mode'], row['target_speedup']) for row in rows} != {
            (mode, target) for mode in MODES for target in TARGETS}:
        raise ValueError('aggregate result matrix is incomplete')
    write_csv(out / 'results_long.csv', rows)
    write_csv(out / 'per_prompt.csv', per_prompt)
    write_csv(out / 'target_k_mapping.csv', mapping_rows)

    max_error = max(float(row['target_relative_error']) for row in rows)
    best_by_target = {target: max((row for row in rows if row['target_speedup'] == target),
                                  key=lambda row: (row['vbench_custom_score'] if score_vbench else row['psnr_rgb_db'], row['psnr_rgb_db']))
                      for target in TARGETS}
    report = ['# Ours4Wan21 five-prompt speed-target results', '',
        'This is a five-prompt matched diagnostic subset drawn from VBench200. It is not an official full VBench or VBench200 score.', '',
        f'- Validated matrix: 12 modes × 3 targets = 36 conditions; 5 videos per condition.',
        f'- Maximum absolute relative target-speed error: {max_error:.2%}.',
        '- Fixed calibrated mapping: 1.8x→K23, 2.4x→K29, 3.0x→K35; no adaptive K rescan.',
        '- Headline latency is ratio-of-sums for complete `pipeline.generate`; model load, warmup, video writing, and evaluation are excluded.',
        '- PSNR/SSIM/LPIPS compare each candidate with the same-prompt native baseline on the same physical GPU.',
        ('- VBench custom score is the unweighted raw mean of ten official custom-input dimensions.' if score_vbench else
         '- VBench scoring was skipped at user request; missing scores are not zero.'), '',
        ('## Highest diagnostic VBench custom score at each target' if score_vbench else
         '## Highest paired PSNR at each target'), '']
    for target, best in best_by_target.items():
        report.append(f'- {target:.1f}×: {best["label"]}, achieved {best["achieved_latency_speedup"]:.3f}×, '
                      + (f'VBench custom {best["vbench_custom_score"]:.4f}, ' if score_vbench else '')
                      + f'PSNR {best["psnr_rgb_db"]:.3f} dB.')
    report.extend(['', 'Exact values and all component definitions are in `results_long.csv`; per-prompt evidence is in `per_prompt.csv`.'])
    (out / 'RESULTS.md').write_text('\n'.join(report) + '\n')
    validation = dict(schema='ours4wan21_vbench5_speed_targets_validation_v1', status='pass',
        validated_at=datetime.now(timezone.utc).isoformat(), modes=12, targets=3,
        conditions=36, prompt_rows=180, prompt_count=5, frames_per_video=81,
        official_full_vbench_score=False, vbench_enabled=score_vbench,
        score_scope=config['prompt_subset']['score_scope'] if score_vbench else 'paired_video_metrics_only',
        max_target_relative_error=max_error, adaptive_k_rescan=False,
        target_k_mapping={'1.8': 23, '2.4': 29, '3.0': 35}, evidence_sha256=evidence_hashes,
        metric_definitions=dict(
            latency='ratio of sums of complete pipeline.generate wall time; excludes model load, one native warmup, MP4 and evaluation',
            dit_tflops='Calflops observed modules plus analytic dense-attention core, summed by actual full/reuse calls',
            t5_vae_tflops='separately profiled per-video operation estimates',
            predictor='FP32 policy MLP only; feature FFT/quantile/reductions excluded from TFLOPs but included in wall latency',
            video_metrics='RGB per-frame PSNR/SSIM/AlexNet-LPIPS, then equal-weight video mean',
            vbench='non-official raw mean of ten official VBench custom-input dimensions' if score_vbench else 'skipped_by_user; not measured'))
    dump(out / 'VALIDATION.json', validation)
    dump(out / 'COMPLETE.json', dict(status='complete', rows=36, per_prompt_rows=180,
        results_sha256=sha256(out / 'results_long.csv'), validation_sha256=sha256(out / 'VALIDATION.json')))
    print(json.dumps(dict(status='pass', conditions=36, max_target_error=max_error,
                          adaptive_k_rescan=False), indent=2))


if __name__ == '__main__':
    main()
