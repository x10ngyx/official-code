#!/usr/bin/env python3
"""Audit stored cross-method baselines and prepare source-preserving reuse links."""
import argparse
import csv
import filecmp
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parent
WORKSPACE = REPO.parents[1]
EXP = Path('/all/yiran07-disk3/huteng_data/exp')
SEA_NAME = 'wan22_seacache_vbench200_thr024_038_055_persistent_batch2wave_gpu0123_20260903_002834'
TEA_NAME = 'teacache4wan22_vbench200_thr029_045_068_fullwall_staggered_gpu0123_20260904_133701'
MAG_NAME = 'magcache4wan22_targeted_e_scan_gpu123_20260905_165530'
TEA_OLD = EXP / 'teacache4wan22_vbench8_threshold_scan_2worker_persistent_20260830_015951'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    assert Path(sys.prefix).name == 'wan2.2'
    assert all(os.environ.get(v) == '1' for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
    out = args.output_dir.resolve()
    assert out.is_relative_to(EXP) and out != EXP
    assert not out.exists(), 'Use a fresh audit output directory.'
    sources, hashes = {}, {}

    def artifact(path):
        path = Path(path).resolve(strict=True)
        if str(path) not in hashes:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            hashes[str(path)] = h.hexdigest()
        value = dict(path=str(path), sha256=hashes[str(path)], bytes=path.stat().st_size)
        sources[str(path)] = value
        return value

    def read(path):
        artifact(path)
        return json.loads(Path(path).read_text())

    def equal_video(path, reference, expected_sha=None):
        a, b = artifact(path), artifact(reference)
        assert a['sha256'] == b['sha256'] and a['bytes'] == b['bytes']
        assert filecmp.cmp(str(path), str(reference), shallow=False)
        if expected_sha is not None:
            assert a['sha256'] == expected_sha
        sa, sb = Path(path).stat(), Path(reference).stat()
        return dict(**a, byte_identical=True, independent_file=(sa.st_dev, sa.st_ino) != (sb.st_dev, sb.st_ino))

    expected = dict(task='t2v-A14B', size_wh=[832, 480], frame_num=45, fps=16,
                    sampling_steps=50, sample_solver='dpm++', shift=12.0,
                    guide_scale_low_high=[3.0, 4.0], boundary=.875, seed=42,
                    param_dtype='torch.bfloat16', offload_model=True, t5_cpu=False)
    sea = REPO / 'SeaCache4Wan22/experiment_results' / (SEA_NAME + '_thr0p24')
    tea = REPO / 'TeaCache4Wan22/experiment_results' / (TEA_NAME + '_thr0p29')
    baseline = (sea / 'baseline').resolve(strict=True)
    validator_file = REPO / 'TeaCache4Wan22/experiments/vbench200_t2v/run_vbench200.py'
    spec = importlib.util.spec_from_file_location('tea_reuse_validator', validator_file)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    artifact(validator_file)
    source_validation = validator.validate_reusable_baseline(
        baseline, REPO / 'TeaCache4Wan22/build/Wan2.2-42bf4cf-prepared/.teacache4wan22_prepared.json',
        WORKSPACE / 'models/Wan2.2-T2V-A14B', 200)
    checkpoint = (WORKSPACE / 'models/Wan2.2-T2V-A14B').resolve(strict=True)
    prompts_file = REPO / 'Vbench200/prompts.jsonl'
    artifact(prompts_file)
    prompts = {x['sample_id']: x for x in map(json.loads, prompts_file.read_text().splitlines())}
    assert len(prompts) == 200
    recorded = {}
    for path in sorted(baseline.glob('generation_config.shard_*.json')):
        cfg = read(path)
        read(cfg['prepared_manifest'])
        assert artifact(cfg['prepared_manifest'])['sha256'] == cfg['prepared_manifest_sha256']
    for path in sorted(baseline.glob('generation_manifest.shard_*.jsonl')):
        artifact(path)
        for row in map(json.loads, path.read_text().splitlines()):
            assert row['sample_id'] not in recorded
            recorded[row['sample_id']] = row
    assert set(recorded) == set(prompts)
    quality_csv = sea / 'evaluation/video_metrics/per_video.csv'
    artifact(quality_csv)
    with quality_csv.open() as stream:
        quality_rows = {r['video_id']: r for r in csv.DictReader(stream)}
    assert set(quality_rows) == set(prompts)
    reuse_rows, timing = [], {}
    for sid, prompt in prompts.items():
        row = recorded[sid]
        assert row['prompt_en'] == prompt['prompt_en'] and row['seed'] == 42
        video = baseline / 'videos' / (sid + '.mp4')
        assert Path(row['video']).resolve() == video.resolve()
        va = artifact(video)
        assert va['sha256'] == quality_rows[sid]['reference_sha256']
        timefile = baseline / 'timings' / (sid + '.json')
        tm = read(timefile)
        assert math.isclose(tm['pipeline_generate_wall_seconds'], row['pipeline_generate_wall_seconds'])
        for key, count in [('t5', 2), ('dit', 100), ('vae_decode', 1)]:
            c = tm['component_latency'][key]
            assert c['call_count'] == count and math.isfinite(c['cuda_seconds']) and c['cuda_seconds'] > 0
        timing[sid] = tm
        reuse_rows.append(dict(sample_id=sid, prompt_en=prompt['prompt_en'], seed=42,
                               video=va, timing=artifact(timefile),
                               generate_seconds=tm['pipeline_generate_wall_seconds']))
    shared_views = []
    for method, prefix, thresholds in [('SeaCache4Wan22', SEA_NAME, ['0p24', '0p38', '0p55']),
                                       ('TeaCache4Wan22', TEA_NAME, ['0p29', '0p45', '0p68'])]:
        for threshold in thresholds:
            directory = REPO / method / 'experiment_results' / (prefix + '_thr' + threshold)
            config = read(directory / 'run_config.json')
            view = directory / 'baseline'
            assert view.resolve() == baseline
            assert Path(config['baseline_reuse']['source']).resolve() == baseline
            assert {p.stem for p in (view / 'videos').glob('*.mp4')} == set(prompts)
            assert all((view / 'videos' / (sid + '.mp4')).resolve() == Path(row['video']).resolve()
                       for sid, row in recorded.items())
            shared_views.append(dict(method=method, threshold=threshold, baseline=str(view),
                                     resolved=str(baseline), video_count=200, same_files=True))
    mag = EXP / MAG_NAME
    comparisons, mag_manifests, timing_comparison = [], {}, []
    for group in ['target_1p8_gpu1', 'target_2p4_gpu2', 'target_3p0_gpu3']:
        report = read(mag / group / 'report.json')
        complete = read(mag / group / 'COMPLETE.json')
        assert complete['status'] == 'complete'
        assert artifact(mag / group / 'report.json')['sha256'] == complete['report']['sha256']
        mag_manifests[group] = {}
        for row in report['plan']['prompts']:
            sid = row['sample_id']
            m = read(mag / group / 'runs/baseline' / sid / 'manifest.json')
            assert m['mode'] == 'baseline' and m['status'] == 'complete'
            assert m['prompt'] == prompts[sid]['prompt_en']
            assert all(m['protocol'][k] == v for k, v in expected.items())
            assert Path(m['checkpoint']).resolve() == checkpoint
            assert m['source']['wan22_commit'] == source_validation['shared_wan22_commit']
            tm = read(m['timing']['path'])
            assert artifact(m['timing']['path'])['sha256'] == m['timing']['sha256']
            assert tm['full_compute_forward_calls'] == 100 and tm['reuse_forward_calls'] == 0
            video = equal_video(m['video']['path'], baseline / 'videos' / (sid + '.mp4'), m['video']['sha256'])
            assert video['independent_file']
            comparisons.append(dict(sample_id=sid, source=group, **video))
            mag_manifests[group][sid] = m
        choice = report['target_selection']['targets'][0]
        perf = report['performance'][choice['nearest_condition']]
        shared_sum = sum(timing[sid]['pipeline_generate_wall_seconds'] for sid in mag_manifests[group])
        timing_comparison.append(dict(target=choice['target'], prompt_count=11,
            mag_baseline_mean_seconds=perf['sums']['baseline']['generate_seconds'] / 11,
            shared_baseline_mean_seconds=shared_sum / 11, original_speedup=perf['speedup'],
            arithmetic_speedup_using_shared_baseline=shared_sum / perf['sums']['magcache']['generate_seconds']))
    first = mag_manifests['target_1p8_gpu1']
    assert len(first) == 11 and all(set(m) == set(first) for m in mag_manifests.values())
    old_paths = sorted((TEA_OLD / 'runs').glob('*/baseline.manifest.json'))
    assert len(old_paths) == 8
    for path in old_paths:
        m = read(path)
        sid = path.parent.name
        log = read(path.parent / 'baseline.log')
        assert sid in first and m['threshold'] == 0 and log['seed'] == 42 and log['status'] == 'success'
        assert m['prompt'] == prompts[sid]['prompt_en'] == log['prompt']
        assert all(m['protocol']['payload'][k] == v for k, v in expected.items())
        assert Path(m['checkpoint']).resolve() == checkpoint
        assert m['prepared_source_manifest']['payload']['wan22_commit'] == source_validation['shared_wan22_commit']
        video = equal_video(m['video']['path'], baseline / 'videos' / (sid + '.mp4'), m['video']['sha256'])
        assert video['independent_file']
        for manifests in mag_manifests.values():
            equal_video(m['video']['path'], manifests[sid]['video']['path'])
        comparisons.append(dict(sample_id=sid, source='TeaCache_historical_independent', **video))
    probes = {}
    for sid in first:
        video = baseline / 'videos' / (sid + '.mp4')
        value = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height,nb_frames,avg_frame_rate', '-of', 'json', str(video)], text=True))
        stream = value['streams'][0]
        assert (stream['width'], stream['height'], stream['nb_frames'], stream['avg_frame_rate']) == (832, 480, '45', '16/1')
        probes[sid] = stream
    assert len(comparisons) == 41
    out.mkdir(parents=True)
    (out / 'shared_baseline').symlink_to(baseline, target_is_directory=True)
    subset = out / 'calibration11_videos'
    subset.mkdir()
    (subset / 'README.md').write_text('# Calibration subset\n\nThe 11 MagCache calibration prompt IDs, linked to the verified shared baseline MP4s. No copied videos or substituted timing files.\n')
    for sid in first:
        (subset / (sid + '.mp4')).symlink_to(baseline / 'videos' / (sid + '.mp4'))
    reuse_manifest = dict(schema='wan22_shared_baseline_reuse_index_v1', status='validated',
        baseline_directory=str(baseline), linked_directory=str(out / 'shared_baseline'),
        protocol=expected, checkpoint=str(checkpoint), video_count=200, rows=reuse_rows,
        timing_policy='Historical recorded complete-generate timings retain their source and lifecycle. MagCache calibration timings and reported speedups are not replaced.',
        integration='Reusable source index; the current MagCache runner does not yet accept an external baseline source.')
    with (out / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    result = dict(status='pass', checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        shared_video_count=200, mag_independent_comparisons=33, historical_tea_independent_videos=8,
        mag_vs_historical_tea_pairs=24, sha_and_byte_mismatches=0, shared_views=shared_views,
        source_validation=source_validation, comparisons=comparisons, video_headers=probes,
        timing_comparison=timing_comparison, source_artifacts=list(sources.values()),
        script=artifact(Path(__file__)))
    for filename, value in [('VALIDATION.json', result), ('baseline_reuse_manifest.json', reuse_manifest)]:
        (out / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    lines = ['# MagCache / TeaCache / SeaCache baseline 复用核验', '',
        '结论：已检查的独立生成baseline完全一致，可复用现有SeaCache/TeaCache共用的200条视频。', '',
        '| 检查对象 | 数量 | 结果 |', '|---|---|---|',
        '| MagCache GPU1/2/3 对共享baseline | 11×3=33条 | SHA256、文件大小及实际字节全部一致 |',
        '| TeaCache历史独立baseline对共享baseline | 8条 | 全部一致，独立文件 |',
        '| MagCache三卡对TeaCache历史独立baseline | 8×3=24对 | 全部一致 |',
        '| SeaCache .24/.38/.55及TeaCache .29/.45/.68 | 6个正式入口，各200条 | 都指向同一套物理文件 |', '',
        '共享200条视频的当前SHA与既有VideoMetrics评测时记录的reference SHA全部匹配；200条prompt/seed、4份生成shard配置、200条计时的100次全计算调用以及T5/DiT/VAE组件记录通过。共同11条视频头均为832×480、45帧、16fps。MagCache没有新生成其余189条，独立跨方法一致性证据覆盖现有11条。', '',
        '共同协议：Wan2.2-T2V-A14B、50-step DPM++、shift12、CFG=(3,4)、boundary=.875、seed42、BF16、单卡model offload、T5 GPU；同一checkpoint路径与Wan2.2 commit 42bf4cf。', '',
        f'可复用源目录：`{baseline}`。本目录`shared_baseline/`直接链接整套视频、计时及生成配置；`calibration11_videos/`提供同11条子集视频入口。`baseline_reuse_manifest.json`冻结200条prompt/video/timing映射和SHA，`comparison.csv`记录独立文件逐条对照，`VALIDATION.json`记录来源及检查。', '',
        '视频相同不代表计时相同。同11条prompt的历史baseline均值为1034.828983秒，本次MagCache三卡baseline均值为992.489396/983.776543/989.400993秒。旧baseline每样本独立进程，MagCache使用持久worker；测量日期、卡和运行状态也有差异，不能仅凭视频相同判定速度口径自动可替换。', '',
        '| 目标 | 原同卡标定速度 | 若仅换用历史同11条baseline时间 |', '|---|---|---|']
    for row in timing_comparison:
        lines.append(f"| {row['target']:.1f}× | {row['original_speedup']:.6f}× | {row['arithmetic_speedup_using_shared_baseline']:.6f}× |")
    lines += ['', '右列仅为更换分母的算术复算，并非新测速或正式200条结果。原MagCache扫描计时、目标选择与报告保持原值；后续统一复用历史计时，应显式保留来源并按同一prompt集合重算加速比。', '',
        '本次已准备数据复用入口，没有修改MagCache生成runner；当前runner尚不支持直接导入SeaCache格式baseline，不能将其复制成MagCache自身manifest或伪装成同卡新测速。未启动新推理、训练或GPU评测。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (out / 'README.md').write_text('# Baseline reuse audit\n\nREPORT.md contains the findings and timing scope. VALIDATION.json records source hashes and checks. comparison.csv lists all 41 independently stored baseline comparisons. baseline_reuse_manifest.json indexes the existing 200 baseline videos/timings. shared_baseline/ and calibration11_videos/ contain symlink references only.\n')
    link = PROJECT / 'experiment_results' / out.name
    assert not link.exists() and not link.is_symlink()
    link.symlink_to(out, target_is_directory=True)
    print(json.dumps(dict(status='pass', output=str(out), mag_baselines=33,
                         historical_tea_baselines=8, shared_baselines=200,
                         mismatches=0, timing_comparison=timing_comparison), ensure_ascii=False))


if __name__ == '__main__':
    main()
