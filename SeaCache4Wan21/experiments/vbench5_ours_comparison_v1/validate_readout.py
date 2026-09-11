"""Read-only independent verification of the completed four-GPU comparison."""
import csv
import json
import math
import statistics
import run_4gpu as suite


def main():
    root = suite.ROOT
    config = suite.read(root/'config.json')
    done = suite.read(root/'COMPLETE.json')
    assert done['status'] == 'complete'
    assert done['comparison_sha256'] == suite.sha256(root/'analysis/comparison.csv')
    validation = suite.read(root/'analysis/VALIDATION.json')
    assert validation['status'] == 'pass' and validation['config_sha256'] == suite.sha256(root/'config.json')
    for path, digest in {**config['source_sha256'], **validation['quality_sha256']}.items():
        assert suite.sha256(suite.Path(path)) == digest
    with (root/'analysis/comparison.csv').open() as f:
        comparison = list(csv.DictReader(f))
    assert len(comparison) == len({(r['label'],r['target']) for r in comparison}) == 39
    by_gpu = {}
    for g in range(4):
        expected = [j for j in config['jobs'] if j['gpu'] == g]
        assert suite.read(root/f'COMPLETE_gpu{g}.json')['videos'] == len(expected)
        assert suite.read(root/f'run_gpu{g}.json')['gpu_uuid'] == config['gpu_uuids'][str(g)]
        rows = suite.read(root/f'components_gpu{g}.json')['rows']
        assert {(r['target_index'], r['sample_id']) for r in rows} == {(j['target_index'],j['sample_id']) for j in expected}
        by_gpu[g] = rows
    report = []
    profile = suite.read(suite.common.PROFILE)
    full = profile['per_model_forward']['estimated_full_flops']
    always = profile['per_model_forward']['estimated_always_on_flops']
    for t, target in enumerate(suite.common.TARGETS):
        out = root/f'target_{t}'
        jobs = [j for j in config['jobs'] if j['target_index'] == t]
        metrics = suite.common.quality_means(out/'quality',[j['sample_id'] for j in jobs])
        base_seconds, candidate_seconds, tflops, reused = [], [], [], []
        for j in jobs:
            sid, g = j['sample_id'], j['gpu']
            reference = suite.common.PREVIOUS/'baselines'/f'gpu{g}'/'videos'/f'{sid}.mp4'
            assert (out/'references'/f'{sid}.mp4').resolve() == reference.resolve()
            b = suite.read(reference.parents[1]/'timings'/f'{sid}.json')
            c = suite.read(out/'timings'/f'{sid}.json')
            trace = suite.read(out/'traces'/f'{sid}.json')
            assert c['status'] == 'success' and c['implementation'] == 'seacache'
            assert trace['threshold'] == j['threshold'] and trace['total_branch_calls'] == len(c['calls']) == 100
            assert all(call['blocks_executed'] in (0,30) for call in c['calls'])
            skipped = sum(call['blocks_executed'] == 0 for call in c['calls'])
            assert skipped == trace['reuse'] and trace['reuse']+trace['recompute'] == 100
            flops = sum(always+(full-always)*call['blocks_executed']/30 for call in c['calls'])/1e12
            row = next(r for r in by_gpu[g] if r['target_index'] == t and r['sample_id'] == sid)
            assert math.isclose(flops,row['dit_tflops'],rel_tol=1e-12)
            assert row['generate_seconds'] == c['pipeline_generate_wall_seconds']
            base_seconds.append(b['pipeline_generate_wall_seconds'])
            candidate_seconds.append(c['pipeline_generate_wall_seconds'])
            tflops.append(flops);reused.append(skipped)
        calculated = dict(speedup=sum(base_seconds)/sum(candidate_seconds),seconds=statistics.fmean(candidate_seconds),
                          dit_tflops=statistics.fmean(tflops),**metrics)
        row = next(r for r in comparison if r['label']=='SeaCache' and float(r['target'])==target)
        for key,value in calculated.items():
            assert math.isfinite(value) and math.isclose(value,float(row[key]),rel_tol=1e-10)
        sea7 = next(r for r in comparison if r['label']=='SEA7' and float(r['target'])==target)
        report.append(dict(target=target,**calculated,reuse_branch_calls=reused,
            psnr_delta_vs_sea7=metrics['psnr_rgb_db']-float(sea7['psnr_rgb_db']),
            ssim_delta_vs_sea7=metrics['ssim_rgb']-float(sea7['ssim_rgb']),
            lpips_delta_vs_sea7=metrics['lpips_alex_v0_1_spatial']-float(sea7['lpips_alex_v0_1_spatial'])))
    print(json.dumps(dict(status='pass',sea_video_pairs=15,sea_frame_pairs=1215,conditions=39,results=report),indent=2))


if __name__ == '__main__':
    main()
