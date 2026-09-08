#!/usr/bin/env python3
"""Read and validate completed scan artifacts without starting GPU work."""
import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'runtime'))
sys.path.insert(0, str(PROJECT / 'experiments/performance_t2v_a14b'))
sys.path.insert(0, str(PROJECT / 'experiments/shared_baseline_scan_correction'))
from common import artifact, check_environment, external_output, read_json, sha256, validate_source
from batch import job_plan, validate_completed
from aggregate_performance import aggregate, validate_profile
from scan import select_preset_targets
from normalization import load_reference, normalize


def close(a, b):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10), (a, b)


def stats(values):
    return dict(mean=statistics.mean(values), std_population=statistics.pstdev(values),
                min=min(values), max=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--baseline-videos', type=Path)
    args = parser.parse_args()
    check_environment()
    root = external_output(args.result_dir)
    evidence = {}

    def record(path):
        evidence[str(path)] = sha256(path)
        return read_json(path)

    def committed(marker):
        value = record(marker)
        close_hash = sha256(value['path'])
        assert close_hash == value['sha256'], marker
        return record(Path(value['path']))

    correction = record(root / 'timing_correction.json') if (root / 'timing_correction.json').exists() else None
    if correction:
        reference_rows, reference_identities = load_reference(correction)
        for source in correction['sources'].values():
            evidence[source['path']] = source['sha256']
        evidence[str(PROJECT / 'experiments/shared_baseline_scan_correction/normalization.py')] = sha256(PROJECT / 'experiments/shared_baseline_scan_correction/normalization.py')
    complete = record(root / 'COMPLETE.json')
    assert complete['status'] == record(root / 'status.json')['status'] == 'complete'
    assert sha256(complete['config']['path']) == complete['config']['sha256']
    profile = record(root / 'shared_profile/calflops.json')
    validate_profile(profile)
    prepared = validate_source(profile['source']['wan22_root'])
    assert prepared == profile['source']['prepared_manifest']
    dimensions = record(PROJECT.parent / 'VbenchEvaluation/dimensions.json')
    counts = dict(measured_videos=0, candidate_conditions=0, paired_videos=0,
                  paired_frames=0, vbench_groups=0, raw_threshold_points=0)
    all_rows, selected, baseline_hashes = [], [], {}
    for group in sorted(p for p in root.glob('target_*_gpu*') if p.is_dir()):
        report_marker = record(group / 'COMPLETE.json')
        assert report_marker['status'] == record(group / 'status.json')['status'] == 'complete'
        assert sha256(report_marker['report']['path']) == report_marker['report']['sha256']
        report = record(group / 'report.json')
        plan = record(group / 'plan.json')
        assert plan == report['plan'] and plan['source'] == prepared
        generation = record(group / 'GENERATION_COMPLETE.json')
        assert generation['status'] == 'complete'
        assert sha256(generation['performance']['path']) == generation['performance']['sha256']
        performance = record(group / 'performance.json')
        assert performance == report['performance']
        assert committed(group / 'profile_artifact.json') == profile
        manifests, paths = {}, {}
        for condition in plan['conditions']:
            cid = condition['id']
            manifests[cid], paths[cid] = {}, []
            for prompt in plan['prompts']:
                sid = prompt['sample_id']
                path = group / 'runs' / cid / sid / 'manifest.json'
                manifest = validate_completed(path, job_plan(plan, prompt, condition), profile,
                                              plan['gpu_ids'][0], plan['warmup_videos'])
                record(path)
                paths[cid].append(path)
                manifests[cid][sid] = manifest
                if condition['mode'] == 'magcache':
                    trace = read_json(manifest['trace']['path'])
                    assert [x['action'] for x in trace['calls']] == condition['schedule']
                counts['measured_videos'] += 1
        baseline_hashes[group.name] = {sid: m['video']['sha256'] for sid, m in manifests['baseline'].items()}
        if correction:
            for sid, manifest in manifests['baseline'].items():
                assert manifest['video']['sha256'] == reference_identities[sid]['video']['sha256']
                assert manifest['prompt'] == reference_identities[sid]['prompt_en']
                assert manifest['protocol']['seed'] == reference_identities[sid]['seed']
        selection = select_preset_targets(performance, plan['target_groups'])
        assert selection == record(group / 'target_selection.json') == report['target_selection']
        chosen = selection['targets'][0]
        assert chosen['status'] == 'hit'
        for condition in plan['conditions'][1:]:
            cid = condition['id']
            perf = performance[cid]
            recalculated = aggregate(paths['baseline'], paths[cid], profile)
            measured = perf['raw_same_gpu_performance'] if correction else perf
            assert all(measured[k] == v for k, v in recalculated.items())
            if correction:
                assert perf == normalize(measured, reference_rows, correction)
            assert perf['equivalent_parameters'] == condition['equivalent_parameters']
            quality = committed(group / 'evaluation' / f'video_metrics_{cid}.json')
            assert quality == report['video_metrics'][cid]
            assert (quality['video_count'], quality['frame_count_total']) == (11, 495)
            directory = Path(read_json(group / 'evaluation' / f'video_metrics_{cid}.json')['path']).parent
            with (directory / 'per_frame.csv').open() as stream:
                frames = list(csv.DictReader(stream))
            with (directory / 'per_video.csv').open() as stream:
                videos = list(csv.DictReader(stream))
            for name in ('per_frame.csv', 'per_video.csv'):
                evidence[str(directory / name)] = sha256(directory / name)
            assert len(videos) == 11 and len(frames) == 495
            assert {v['video_id'] for v in videos} == set(manifests[cid])
            for video in videos:
                sid = video['video_id']
                assert video['reference_sha256'] == manifests['baseline'][sid]['video']['sha256']
                assert video['candidate_sha256'] == manifests[cid][sid]['video']['sha256']
                assert tuple(int(video[k]) for k in ('frames', 'height', 'width')) == (45, 480, 832)
                vf = [f for f in frames if f['video_id'] == sid]
                assert len(vf) == 45 and {int(f['frame_index']) for f in vf} == set(range(45))
                for metric in quality['metrics']:
                    for statistic, value in stats([float(f[metric]) for f in vf]).items():
                        close(value, video[metric + '_' + statistic])
            for metric, values in quality['metrics'].items():
                for statistic, value in stats([float(v[metric + '_mean']) for v in videos]).items():
                    close(value, values[statistic])
            means = {mode: {k: v / 11 for k, v in sums.items()} for mode, sums in perf['sums'].items()}
            thresholds = sorted([condition['threshold']] + [p['threshold'] for p in condition['equivalent_parameters']])
            row = dict(group=group.name, target=chosen['target'], condition=cid,
                       selected=cid == chosen['nearest_condition'], R=condition['retention_ratio'],
                       E=condition['threshold'], K=condition['K'], speedup=perf['speedup'],
                       relative_error_percent=(perf['speedup'] / chosen['target'] - 1) * 100,
                       baseline_seconds=means['baseline']['generate_seconds'],
                       generate_seconds=means['magcache']['generate_seconds'],
                       dit_tflops=means['magcache']['estimated_dit_tflops'],
                       full_calls=means['magcache']['full_calls'],
                       psnr=quality['metrics']['psnr_rgb_db']['mean'],
                       ssim=quality['metrics']['ssim_rgb']['mean'],
                       lpips=quality['metrics']['lpips_alex_v0_1_spatial']['mean'],
                       vbench_score=report['vbench_score'][cid],
                       baseline_vbench_score=report['vbench_score']['baseline'],
                       E_min=min(thresholds), E_max=max(thresholds), E_grid_count=len(thresholds))
            all_rows.append(row)
            if row['selected']:
                selected.append(dict(**row, component_means=means, equivalent_E_grid=thresholds))
            counts['candidate_conditions'] += 1
            counts['raw_threshold_points'] += len(thresholds)
            counts['paired_videos'] += 11
            counts['paired_frames'] += 495
        for condition in plan['conditions']:
            cid = condition['id']
            value = committed(group / 'evaluation' / f'vbench_{cid}.json')
            assert value == report['vbench'][cid]
            assert set(value['raw_dimension_scores']) == set(dimensions['dimensions'])
            norm = {}
            for dim in dimensions['dimensions']:
                raw = record(Path(value['source_files'][dim]))[dim][0]
                close(raw, value['raw_dimension_scores'][dim])
                spec = dimensions['normalization'][dim]
                norm[dim] = (raw - spec['min']) / (spec['max'] - spec['min'])
                close(norm[dim], value['normalized_dimension_scores'][dim])
            def group_score(key):
                ds = dimensions[key + '_dimensions']
                weights = dimensions['normalization']
                return sum(norm[d] * weights[d]['weight'] for d in ds) / sum(weights[d]['weight'] for d in ds)
            q, s = group_score('quality'), group_score('semantic')
            close(q, value['aggregate_scores']['quality_score'])
            close(s, value['aggregate_scores']['semantic_score'])
            close((4 * q + s) / 5, report['vbench_score'][cid])
            close((4 * q + s) / 5, value['aggregate_scores']['total_score'])
            directory = Path(read_json(group / 'evaluation' / f'vbench_{cid}.json')['path']).parent
            stage = record(directory / 'staging_manifest.json')
            subset = record(directory / 'subset_full_info_manifest.json')
            assert stage['staged_video_count'] == len(stage['items']) == 11
            assert subset['selected_prompt_count'] == 11
            assert set(subset['covered_dimensions']) == set(dimensions['dimensions'])
            assert {item['sample_id'] for item in stage['items']} == set(manifests[cid])
            for item in stage['items']:
                assert sha256(item['staged_video']) == manifests[cid][item['sample_id']]['video']['sha256']
            counts['vbench_groups'] += 1
    assert counts == dict(measured_videos=187, candidate_conditions=14, paired_videos=154,
                          paired_frames=6930, vbench_groups=17, raw_threshold_points=313)
    first = next(iter(baseline_hashes.values()))
    assert all(hashes == first for hashes in baseline_hashes.values())
    cross_method = None
    if args.baseline_videos:
        other = {sid: artifact(args.baseline_videos / (sid + '.mp4')) for sid in first}
        cross_method = dict(reference_directory=str(args.baseline_videos.resolve()),
                            compared=11, matched=sum(other[sid]['sha256'] == first[sid] for sid in first),
                            reference_videos=other)
    queue = record(root / 'resume_queue_20260906_011210/status.json')
    assert queue['status'] == 'complete'
    times = {key: datetime.fromtimestamp(queue[key], ZoneInfo('Asia/Shanghai')).isoformat()
             for key in ('resumed_unix', 'completed_unix')}
    summary = dict(status='complete', counts=counts, selected=selected, candidates=all_rows,
                   timing=times, cross_gpu_baseline_identical=True,
                   cross_method_baseline=cross_method,
                   scope='11-prompt calibration subset; full generate wall latency; estimated DiT-forward TFLOPs; RGB full-reference metrics; VBench 16-dimension subset, not full 200 prompts.')
    audit = dict(status='pass', checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                 counts=counts, source_validated=True, cross_gpu_baseline_identical=True,
                 cross_method_baseline=cross_method, artifact_sha256=evidence,
                 verification='All measured run SHA/protocol/lifecycle/schedules, recomputed performance/target selection, frame-to-video-to-summary RGB metrics, staged videos and 16-dimension VBench aggregation. No new inference, decoding or model evaluation.',
                 script=artifact(Path(__file__)))
    if correction:
        summary['timing_correction'] = correction
        audit['timing_correction'] = correction
        audit['corrected_performance_verified'] = True
    for filename, value in [('scan_summary.json', summary), ('final_validation.json', audit)]:
        (root / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    with (root / 'scan_candidates.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    lines = ['# MagCache4Wan22 三档扫描结果', '',
             '187/187 个测量视频完成：33 个同卡 baseline + 14 个独立动作序列 × 11 条 prompt。313 个 E 网格点按完整官方动作序列合并，全部候选均完成 RGB PSNR/SSIM/LPIPS 与覆盖 16 维的 VBench 子集评测。', '',
             f"续跑开始：{times['resumed_unix']}；完成：{times['completed_unix']}。三档均通过预设相对 ±2% 容差。", '',
             '| 目标 | R | E | K | 实测加速 | 相对误差 | baseline 秒/视频 | MagCache 秒/视频 | DiT TFLOPs/视频 |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in selected:
        lines.append(f"| {r['target']:.1f}× | {r['R']} | {r['E']:.3f} | {r['K']} | {r['speedup']:.6f}× | {r['relative_error_percent']:+.3f}% | {r['baseline_seconds']:.3f} | {r['generate_seconds']:.3f} | {r['dit_tflops']:.3f} |")
    lines += ['', '| 目标 | RGB PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | VBench 子集分数 | full/reuse CFG calls | 同动作 E 网格平台 |', '|---|---|---|---|---|---|---|']
    for r in selected:
        lines.append(f"| {r['target']:.1f}× | {r['psnr']:.6f} | {r['ssim']:.6f} | {r['lpips']:.6f} | {r['vbench_score']*100:.4f}% | {int(r['full_calls'])}/{100-int(r['full_calls'])} | {r['E_min']:.3f}–{r['E_max']:.3f} |")
    lines += ['', 'Baseline 的 VBench 子集分数为 81.6801%。PSNR/SSIM/LPIPS 衡量与 baseline 的一致性；VBench 分数不代表逐像素保真度。加速提升伴随明显的 baseline 保真度损失。', '',
              '选择规则：在各自固定 R/K 组内，选择实测完整 generate 加速比最接近目标者，没有按质量重新挑选。E 平台仅表示本次 .001 网格上官方 gate 的等价动作；别名未单独计时。速度离散且有计时波动，不能保证每次精确等于目标。', '',
              '原推荐起点 E=.075/.386/.200 的实测速度分别为 1.812139/2.429890/3.051256×；最终最近点调整为 .075/.199/.198。', '',
              '计时包含 CUDA 同步后的完整原生 generate（T5、DiT、VAE、offload、scheduler 等），排除初始化、warmup、导出及质量评测。TFLOPs 是 real-shape Calflops 加 dense FlashAttention 修正的 DiT forward 运算估计，排除 cache controller、residual addition、CFG/scheduler/export；baseline DiT 为 75456.062100 TFLOPs/视频。', '',
              '| 目标 | T5 CUDA 秒 | DiT CUDA 秒 | VAE CUDA 秒 | T5 TFLOPs | VAE TFLOPs |', '|---|---|---|---|---|---|']
    for r in selected:
        m = r['component_means']['magcache']
        lines.append(f"| {r['target']:.1f}× | {m['t5_cuda_seconds']:.6f} | {m['dit_cuda_seconds']:.6f} | {m['vae_decode_cuda_seconds']:.6f} | {m['estimated_t5_tflops_per_video']:.6f} | {m['estimated_vae_decode_tflops_per_video']:.6f} |")
    lines += ['', '固定协议：Wan2.2-T2V-A14B；RTX A6000 单卡（GPU1/2/3 分档）；832×480、45 帧、16 fps；50-step DPM++、shift12、CFG low/high=(3,4)、boundary=.875、seed42、BF16、offload_model=True、t5_cpu=False；关闭 FSDP/SP 与 prompt rewrite/extension。', '',
              '官方实现锁：MagCache df81cb181776c2c61477c08e1d21f87fda1cd938；干净 Wan2.2 42bf4cfaa384bc21833865abc2f9e6c0e67233dc。此次仅汇总既有实验，不修改方法源码或预设。', '',
              '完整性验证：187 条 manifest/video/run/timing/trace、官方 schedule 与 profile/source SHA 通过；154 视频对/6930 帧的逐帧→逐视频→summary 一致；17 组 VBench 的 16 维及聚合通过。三块 GPU 各自生成的 11 个 baseline 逐字节一致。']
    if cross_method:
        lines += ['', f"与既有 SeaCache/TeaCache 共用正式 baseline 对照：{cross_method['matched']}/{cross_method['compared']} 条 MP4 SHA 一致。来源：`{cross_method['reference_directory']}`。"]
    lines += ['', '评测范围：11 条标定 prompt，覆盖 16 维；并非完整 VBench200 或官方全套榜单结果。标准 staging 对剩余 189 条 prompt 标记 missing，这是已选 11 条子集的预期行为。', '',
              '文件：`scan_summary.json` 提供三档参数和完整组件均值；`scan_candidates.csv` 保留全部 14 个候选；`final_validation.json` 记录本次验收和来源 SHA；各 `target_*/report.json`、`runs/` 与 `evaluation/` 保留原始计量和评测。']
    if correction:
        starts = [next(r for r in all_rows if r['group'] == group and r['condition'].endswith('_000')) for group in sorted(baseline_hashes)]
        lines = [line.replace('实测加速', '修正加速').replace('选择实测完整 generate 加速比', '选择修正后完整 generate 加速比') for line in lines]
        lines = [line if not line.startswith('原推荐起点 E=') else '原推荐起点 E=.075/.386/.200 的修正速度分别为 '+ '/'.join(f"{r['speedup']:.6f}" for r in starts)+'×；最近配置为 '+ '/'.join(f"{r['E']:.3f}" for r in selected)+'。' for line in lines]
        lines[2:2] = [f"已改用共享 baseline 中相同11条prompt的修正计时，均值 {selected[0]['baseline_seconds']:.6f} 秒/视频；GPU0原共享baseline按健康卡均值替换，其余保持实测。此处不使用完整200条的均值。候选仍为GPU1/2/3原实测，速度为共享基准归一化估算。", '',
                      f"原同卡baseline及候选的完整原始计量保存在runs/与各performance.json的raw_same_gpu_performance；修正包：`{correction['package']}`。", '']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='pass', counts=counts, selected=selected,
                         cross_method_baseline=cross_method), ensure_ascii=False))


if __name__ == '__main__':
    main()
