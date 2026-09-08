#!/usr/bin/env python3
"""Estimate GPU0 time inflation with exact computation-path matching."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parent
WORKSPACE = REPO.parents[1]
EXP = Path('/all/yiran07-disk3/huteng_data/exp')
SEA = 'wan22_seacache_vbench200_thr024_038_055_persistent_batch2wave_gpu0123_20260903_002834'
TEA = 'teacache4wan22_vbench200_thr029_045_068_fullwall_staggered_gpu0123_20260904_133701'
MAG = EXP / 'magcache4wan22_targeted_e_scan_gpu123_20260905_165530'
METRICS = ['wall_seconds', 'dit_seconds', 't5_seconds', 'vae_seconds', 'non_dit_seconds']


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    with Path(path).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matched(focal, peers, field, key='schedule'):
    pools = defaultdict(list)
    for row in peers:
        pools[row[key]].append(row[field])
    means = {k: statistics.mean(v) for k, v in pools.items()}
    assert all(row[key] in means for row in focal)
    expected = np.array([means[row[key]] for row in focal])
    observed = np.array([row[field] for row in focal])
    return float(observed.sum() / expected.sum()), expected


def block_bootstrap(rows, repetitions=2000, block_length=5):
    """Circular video-block bootstrap within each GPU, preserving temporal clustering."""
    rng = np.random.default_rng(20260906)
    signatures = {s: i for i, s in enumerate(sorted({r['schedule'] for r in rows}))}
    data = [sorted([r for r in rows if r['gpu'] == gpu], key=lambda r: r['ordinal']) for gpu in range(4)]
    times = np.array([[r['wall_seconds'] for r in group] for group in data])
    codes = np.array([[signatures[r['schedule']] for r in group] for group in data])
    factors, attempts = [], 0
    while len(factors) < repetitions and attempts < repetitions * 10:
        attempts += 1
        starts = rng.integers(0, 50, size=(4, math.ceil(50 / block_length)))
        index = ((starts[:, :, None] + np.arange(block_length)) % 50).reshape(4, -1)[:, :50]
        t = np.take_along_axis(times, index, axis=1)
        c = np.take_along_axis(codes, index, axis=1)
        count = np.bincount(c[1:].ravel(), minlength=len(signatures))
        weight = np.bincount(c[1:].ravel(), weights=t[1:].ravel(), minlength=len(signatures))
        if np.any(count[c[0]] == 0):
            continue
        factors.append(float(t[0].sum() / (weight[c[0]] / count[c[0]]).sum()))
    assert len(factors) == repetitions
    lo, hi = np.quantile(factors, [.025, .975])
    return dict(low=float(lo), high=float(hi), block_length=block_length,
                repetitions=repetitions, attempts=attempts, seed=20260906)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    assert Path(sys.prefix).name == 'wan2.2'
    assert all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
    out = args.output_dir.resolve()
    assert out.is_relative_to(EXP) and out != EXP and not out.exists()
    sources = {}

    def artifact(path):
        path = Path(path).resolve(strict=True)
        if str(path) not in sources:
            sources[str(path)] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return sources[str(path)]

    def read(path):
        artifact(path)
        return json.loads(Path(path).read_text())

    sea_root = REPO / 'SeaCache4Wan22/experiment_results' / (SEA + '_thr0p24')
    baseline = (sea_root / 'baseline').resolve()
    cohorts = [('baseline', 'Baseline', baseline, sea_root)]
    for method, prefix, thresholds in [('SeaCache', SEA, ['0p24', '0p38', '0p55']), ('TeaCache', TEA, ['0p29', '0p45', '0p68'])]:
        for th in thresholds:
            parent = REPO / (method + '4Wan22/experiment_results') / (prefix + '_thr' + th)
            cohorts.append((method.lower() + '_' + th, method + ' ' + th.replace('p', '.'), parent / method.lower(), parent))
    rows, definition = [], {}
    for label, display, directory, parent in cohorts:
        run_config = read(directory.parent / 'run_config.json' if label == 'baseline' else parent / 'run_config.json')
        assert run_config['gpu_ids'] == ['0', '1', '2', '3']
        definition[label] = dict(label=display, directory=str(directory.resolve()), run_config=artifact(directory.parent / 'run_config.json' if label == 'baseline' else parent / 'run_config.json'))
        performance_file = parent / 'performance/per_video.jsonl'
        artifact(performance_file)
        performance = {(v['condition'], v['sample_id']): v for v in map(json.loads, performance_file.read_text().splitlines())}
        seen = set()
        for shard in range(4):
            config = read(directory / f'generation_config.shard_{shard:03d}.json')
            gpu = int(config.get('physical_gpu', run_config['gpu_ids'][shard]))
            assert gpu == shard and config['protocol']['seed'] == 42
            if label == 'baseline':
                owner_logs = list((directory.parent / 'orchestration_logs').glob(f'generate_baseline_shard_{shard}.*.log'))
                assert owner_logs
                for log in owner_logs:
                    artifact(log)
                    assert log.read_text().splitlines()[0] == f'CUDA_VISIBLE_DEVICES={gpu}'
            file = directory / f'generation_manifest.shard_{shard:03d}.jsonl'
            artifact(file)
            for item in map(json.loads, file.read_text().splitlines()):
                sid = item['sample_id']
                assert sid not in seen and item['seed'] == 42
                seen.add(sid)
                tm = read(item['timing'])
                assert tm['status'] == 'success' and tm['error'] is None
                assert tm['cuda_device_name'] == 'NVIDIA RTX A6000'
                if label != 'baseline':
                    assert int(tm['pipeline_lifecycle']['physical_gpu']) == gpu
                    assert tm['pipeline_init_wall_seconds'] == 0
                calls = tm['calls']
                assert len(calls) == 100 and tm['model_forward_call_count'] == 100
                for i, call in enumerate(calls):
                    assert (call['call_index'], call['step_index'], call['model_stage'], call['cfg_branch']) == (i, i // 2, 'high' if i < 64 else 'low', 'cond' if i % 2 == 0 else 'uncond')
                    assert call['blocks_executed'] in (0, 40)
                    assert call['full_compute'] == (call['blocks_executed'] == 40)
                    assert call['reuse'] == (call['blocks_executed'] == 0)
                    assert math.isfinite(call['cuda_seconds']) and call['cuda_seconds'] >= 0
                full = sum(c['full_compute'] for c in calls)
                assert full == tm['full_compute_forward_calls'] and 100 - full == tm['reuse_forward_calls']
                assert label != 'baseline' or full == 100
                assert math.isclose(sum(c['cuda_seconds'] for c in calls), tm['dit_cuda_seconds'], rel_tol=1e-10)
                assert math.isclose(item['pipeline_generate_wall_seconds'], tm['pipeline_generate_wall_seconds'], rel_tol=1e-12)
                pr = performance[(config['condition'], sid)]
                for pk, tk in [('pipeline_generate_wall_seconds', 'pipeline_generate_wall_seconds'), ('dit_cuda_seconds', 'dit_cuda_seconds'), ('full_compute_forward_calls', 'full_compute_forward_calls')]:
                    assert math.isclose(pr[pk], tm[tk], rel_tol=1e-10)
                stamp = re.search(r'(\d{8}T\d{6}Z)', item['log'])
                started = datetime.strptime(stamp.group(1), '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).astimezone(ZoneInfo('Asia/Shanghai')).isoformat() if stamp else ''
                row = dict(cohort=label, label=display, gpu=gpu, sample_id=sid, ordinal=int(item['job_ordinal']) // 4,
                    started_cst=started, wall_seconds=tm['pipeline_generate_wall_seconds'], dit_seconds=tm['dit_cuda_seconds'],
                    t5_seconds=tm['t5_cuda_seconds'], vae_seconds=tm['vae_decode_cuda_seconds'],
                    non_dit_seconds=tm['pipeline_generate_wall_seconds'] - tm['dit_cuda_seconds'],
                    full_calls=full, reuse_calls=100-full,
                    full_call_mean_seconds=sum(c['cuda_seconds'] for c in calls if c['full_compute']) / full,
                    dit_tflops=pr['estimated_dit_tflops'],
                    schedule=''.join('F' if c['full_compute'] else 'R' for c in calls), timing_path=item['timing'])
                assert all(math.isfinite(row[k]) and row[k] > 0 for k in METRICS)
                rows.append(row)
        assert len(seen) == 200
    assert len(rows) == 1400
    summaries, gpu_summary, bins, strata = [], [], [], []
    for label, display, _, _ in cohorts:
        group = [r for r in rows if r['cohort'] == label]
        focal = sorted([r for r in group if r['gpu'] == 0], key=lambda r: r['ordinal'])
        peers = [r for r in group if r['gpu'] != 0]
        assert len(focal) == 50 and len(peers) == 150
        factor, expected = matched(focal, peers, 'wall_seconds')
        dit_factor, expected_dit = matched(focal, peers, 'dit_seconds')
        _, expected_other = matched(focal, peers, 'non_dit_seconds')
        delta = sum(r['wall_seconds'] for r in focal) - expected.sum()
        dit_delta = sum(r['dit_seconds'] for r in focal) - expected_dit.sum()
        other_delta = sum(r['non_dit_seconds'] for r in focal) - expected_other.sum()
        assert math.isclose(delta, dit_delta + other_delta, abs_tol=1e-8)
        ci = block_bootstrap(group)
        path_weights = []
        for sig in sorted({r['schedule'] for r in focal}):
            f = [r for r in focal if r['schedule'] == sig]
            h = [r for r in peers if r['schedule'] == sig]
            path_weights.append(len(h))
            strata.append(dict(cohort=label, schedule=sig, gpu0_count=len(f), reference_count=len(h), gpu0_mean_seconds=statistics.mean(r['wall_seconds'] for r in f), reference_mean_seconds=statistics.mean(r['wall_seconds'] for r in h)))
        first_factor, _ = matched(focal[:10], peers, 'wall_seconds')
        last_factor, _ = matched(focal[-10:], peers, 'wall_seconds')
        excluded = [r for r in group if r['ordinal'] > 0]
        nofirst, _ = matched([r for r in excluded if r['gpu'] == 0], [r for r in excluded if r['gpu'] != 0], 'wall_seconds')
        full_match, _ = matched(focal, peers, 'wall_seconds', key='full_calls')
        ratios = np.array([r['wall_seconds'] for r in focal]) / expected
        summary = dict(cohort=label, label=display, gpu0_videos=50, reference_videos=150,
            matching='exact_100_call_full_reuse_sequence', matched_gpu0_videos=50,
            min_reference_count_per_matched_path=min(path_weights), gpu0_mean_seconds=statistics.mean(r['wall_seconds'] for r in focal),
            reference_raw_mean_seconds=statistics.mean(r['wall_seconds'] for r in peers), reference_matched_mean_seconds=float(expected.mean()),
            multiplier=factor, time_inflation_pct=100*(factor-1), throughput_loss_pct=100*(1-1/factor),
            ci95_low_pct=100*(ci['low']-1), ci95_high_pct=100*(ci['high']-1), bootstrap=ci,
            raw_unmatched_inflation_pct=100*(statistics.mean(r['wall_seconds'] for r in focal)/statistics.mean(r['wall_seconds'] for r in peers)-1),
            dit_inflation_pct=100*(dit_factor-1), dit_share_of_excess_pct=100*dit_delta/delta,
            non_dit_difference_seconds=float(other_delta/50), first10_inflation_pct=100*(first_factor-1),
            last10_inflation_pct=100*(last_factor-1), exclude_first_inflation_pct=100*(nofirst-1),
            full_count_match_inflation_pct=100*(full_match-1), video_ratio_p10_pct=100*(float(np.quantile(ratios,.1))-1),
            video_ratio_p90_pct=100*(float(np.quantile(ratios,.9))-1),
            window_start=min(r['started_cst'] for r in group), window_end_last_start=max(r['started_cst'] for r in group))
        summaries.append(summary)
        for i, row in enumerate(focal):
            row['expected_healthy_seconds'] = float(expected[i])
            row['gpu0_matched_ratio'] = float(ratios[i])
        for gpu in range(4):
            sub = sorted([r for r in group if r['gpu'] == gpu], key=lambda r:r['ordinal'])
            f, ex = matched(sub, peers, 'wall_seconds')
            gpu_summary.append(dict(cohort=label, label=display, gpu=gpu, count=len(sub), wall_mean=statistics.mean(r['wall_seconds'] for r in sub), dit_mean=statistics.mean(r['dit_seconds'] for r in sub), full_calls_mean=statistics.mean(r['full_calls'] for r in sub), matched_ratio=f, matched_time_delta_pct=100*(f-1)))
            for k in range(10):
                portion=sub[k*5:(k+1)*5]
                f, ex = matched(portion, peers, 'wall_seconds')
                bins.append(dict(cohort=label,label=display,gpu=gpu,bin=k+1,sample_start=k*5+1,sample_end=k*5+5,time_inflation_pct=100*(f-1)))
    # Same-prompt baseline repeats are kept separate from the within-batch estimator.
    base_by_id = {r['sample_id']: r for r in rows if r['cohort'] == 'baseline'}
    repeats=defaultdict(list)
    for group in ['target_1p8_gpu1','target_2p4_gpu2','target_3p0_gpu3']:
        plan=read(MAG/group/'plan.json')
        for prompt in plan['prompts']:
            sid=prompt['sample_id'];m=read(MAG/group/'runs/baseline'/sid/'manifest.json');t=read(m['timing']['path'])
            assert m['prompt']==prompt['prompt_en'] and m['protocol']['seed']==42
            assert artifact(m['timing']['path'])['sha256']==m['timing']['sha256']
            repeats[sid].append(t['pipeline_generate_wall_seconds'])
    paired=[]
    for sid, values in repeats.items():
        b=base_by_id[sid]
        paired.append(dict(sample_id=sid, old_gpu=b['gpu'], old_seconds=b['wall_seconds'], new_gpu123_mean_seconds=statistics.mean(values), old_new_ratio=b['wall_seconds']/statistics.mean(values)))
    p0=[r for r in paired if r['old_gpu']==0];ph=[r for r in paired if r['old_gpu']!=0]
    paired0=sum(r['old_seconds'] for r in p0)/sum(r['new_gpu123_mean_seconds'] for r in p0)
    pairedh=sum(r['old_seconds'] for r in ph)/sum(r['new_gpu123_mean_seconds'] for r in ph)
    paired_summary=dict(gpu0_prompts=len(p0), reference_prompts=len(ph), gpu0_old_new_ratio=paired0, healthy_old_new_ratio=pairedh, date_adjusted_gpu0_ratio=paired0/pairedh)
    # Preview only: do not replace any baseline or candidate time artifact.
    factors={r['cohort']:r['multiplier'] for r in summaries}
    preview=[]
    baseline_corrected_sum=sum(r['wall_seconds']/factors['baseline'] if r['gpu']==0 else r['wall_seconds'] for r in base_by_id.values())
    for summary in summaries[1:]:
        label=summary['cohort'];rr=[r for r in rows if r['cohort']==label]
        old_base=sum(r['wall_seconds'] for r in base_by_id.values())
        old_candidate=sum(r['wall_seconds'] for r in rr)
        corrected_candidate=sum(r['wall_seconds']/factors[label] if r['gpu']==0 else r['wall_seconds'] for r in rr)
        preview.append(dict(cohort=label, raw_speedup=old_base/old_candidate, estimated_corrected_speedup=baseline_corrected_sum/corrected_candidate,
                            raw_candidate_mean=old_candidate/200,estimated_corrected_candidate_mean=corrected_candidate/200))
    subset_corrected=sum(base_by_id[sid]['wall_seconds']/factors['baseline'] if base_by_id[sid]['gpu']==0 else base_by_id[sid]['wall_seconds'] for sid in repeats)/11
    summary=dict(status='complete', created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        raw_video_count=1400, raw_cfg_call_count=140000, unique_shared_baseline_count=200,
        gpu0_video_count=350, summaries=summaries, per_gpu=gpu_summary, temporal_bins=bins,
        paired_baseline=paired_summary, correction_preview=preview,
        shared_baseline_raw_mean=sum(r['wall_seconds'] for r in base_by_id.values())/200,
        shared_baseline_estimated_corrected_mean=baseline_corrected_sum/200,
        calibration11_shared_baseline_estimated_corrected_mean=subset_corrected,
        caveats=['Exact action matching controls measured computation, but the GPU partitions contain different prompts.',
                 'Block-bootstrap intervals describe within-batch sampling variation, not future temperature states or causal certainty.',
                 'Historical contemporaneous notes report GPU0 SW thermal slowdown at 91–92 C; continuous per-video clocks/throttle telemetry was not found in these formal result roots.',
                 'Correction preview is a modeled counterfactual, not newly measured inference; raw files remain authoritative and intact.'])
    out.mkdir(parents=True)
    # All rows use the same CSV schema; only GPU0 needs expected reference values.
    for row in rows:
        row.setdefault('expected_healthy_seconds','')
        row.setdefault('gpu0_matched_ratio','')
    write_csv(out/'videos.csv',rows)
    write_csv(out/'gpu_summary.csv',gpu_summary)
    write_csv(out/'temporal_bins.csv',bins)
    write_csv(out/'matched_strata.csv',strata)
    write_csv(out/'same_prompt_baselines.csv',paired)
    write_csv(out/'correction_preview.csv',preview)
    dump(out/'summary.json',summary)
    dump(out/'correction_factors.json',dict(status='estimated_not_applied',scope='Only the named frozen VBench200 batches',formula='GPU0 corrected seconds = raw seconds / multiplier; GPUs1/2/3 factor=1; baseline/candidate owner handled independently',factors=factors))
    for log in ['logs/session_20260902_1035_task_progress_check.md','logs/session_20260902_2320_seacache_vbench200_candidate_failure.md','work/offical-code/TeaCache4Wan22/logs/session_20260904_1349_formal_suite_launch.md']:
        artifact(WORKSPACE/log)
    artifact(Path(__file__))
    dump(out/'source_manifest.json',dict(sources=list(sources.values()),cohorts=definition))
    (out/'README.md').write_text('# GPU0 timing audit\n\nsummary.json and correction_factors.json contain batch-specific estimates. videos.csv has 1,400 unique measured runs; gpu_summary.csv, matched_strata.csv and temporal_bins.csv explain matching and variation. same_prompt_baselines.csv provides independent MagCache repeat checks. correction_preview.csv is estimated arithmetic, not a replacement for original measurements. source_manifest.json tracks raw source SHA. The companion notebook and report document interpretation and validation.\n')
    link=PROJECT/'experiment_results'/out.name
    assert not link.exists() and not link.is_symlink()
    link.symlink_to(out,target_is_directory=True)
    print(json.dumps(dict(output=str(out), summaries=[{k:r[k] for k in ['label','gpu0_mean_seconds','reference_matched_mean_seconds','multiplier','time_inflation_pct','throughput_loss_pct','ci95_low_pct','ci95_high_pct','dit_share_of_excess_pct','first10_inflation_pct','last10_inflation_pct','min_reference_count_per_matched_path']} for r in summaries],paired_baseline=paired_summary,corrected_baseline_mean=summary['shared_baseline_estimated_corrected_mean'],corrected_subset_mean=subset_corrected),ensure_ascii=False))


if __name__ == '__main__':
    main()
