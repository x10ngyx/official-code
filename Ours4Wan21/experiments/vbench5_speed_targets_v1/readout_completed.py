"""Read-only verification/readout; does not repair or advance the pipeline."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_random3000_12groups_v1_vbench5_speed_targets_v1')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    training = ROOT.parent / 'ours21_random3000_12groups_v1_training_readout'
    complete = read(training / 'COMPLETE.json')
    for key, name in [('validation_sha256', 'VALIDATION.json'),
                      ('summary_sha256', 'suite_summary.csv'),
                      ('epoch_metrics_sha256', 'epoch_metrics_long.csv')]:
        assert complete[key] == sha(training / name)
    groups = list(csv.DictReader((training / 'suite_summary.csv').open()))
    results = []
    for group in groups:
        for k in (23, 29, 35):
            d = ROOT / 'candidates' / group['mode'] / f'K{k}'
            p, manifest = read(d / 'performance.json'), read(d / 'run.json')
            qd = d / 'quality/video_metrics'
            summary = read(qd / 'summary.json')
            qs = list(csv.DictReader((qd / 'per_video.csv').open()))
            assert read(d / 'quality/COMPLETE.json')['video_metrics_sha256'] == sha(qd / 'summary.json')
            ids = {r['sample_id'] for r in p['candidate']}
            assert len(ids) == 5 and ids == {r['sample_id'] for r in p['baseline']} == {r['video_id'] for r in qs}
            assert read(d / 'COMPLETE.json')['videos'] == summary['video_count'] == len(qs) == 5
            assert summary['frame_count_total'] == 405
            assert manifest['policy_sha256'] == group['checkpoint_sha256']
            for row in qs:
                assert (int(row['frames']), int(row['height']), int(row['width'])) == (81, 480, 832)
                for side in ('reference', 'candidate'):
                    assert sha(Path(row[side])) == row[side + '_sha256']
                baseline = read(Path(row['reference']).parents[1] / 'run.json')
                assert baseline['gpu_uuid'] == manifest['gpu_uuid']
                trace = read(d / 'traces' / (row['video_id'] + '.json'))
                assert trace['step_reuse'] == k and trace['total_steps'] == 50
            result = dict(label=group['label'], epoch=int(group['selected_epoch']), k=k)
            result['speedup'] = sum(r['generate_seconds'] for r in p['baseline']) / sum(r['generate_seconds'] for r in p['candidate'])
            result['seconds'] = statistics.fmean(r['generate_seconds'] for r in p['candidate'])
            for metric in ('psnr_rgb_db', 'ssim_rgb', 'lpips_alex_v0_1_spatial'):
                value = statistics.fmean(float(r[metric + '_mean']) for r in qs)
                assert math.isfinite(value) and abs(value - summary['metrics'][metric]['mean']) < 1e-10
                result[metric] = value
            results.append(result)
    assert len(groups) == 12 and len(results) == 36
    print(json.dumps(dict(validation='pass', conditions=36, video_pairs=180,
                          frame_pairs=14580, results=results), indent=2))


if __name__ == '__main__':
    main()
