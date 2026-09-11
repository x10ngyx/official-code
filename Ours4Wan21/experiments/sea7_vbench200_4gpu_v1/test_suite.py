"""CPU checks for the SEA7 full-set queue; never launch inference."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_suite as suite


def prompt_rows():
    return [dict(sample_id=f'p{i:03d}', prompt_en=f'Prompt {i}') for i in range(200)]


def measured(sid, seconds):
    row = dict(sample_id=sid, generate_seconds=seconds)
    for key in ('dit_cuda_seconds', 't5_cuda_seconds', 'vae_decode_cuda_seconds', 'dit_tflops',
                'estimated_t5_tflops_per_video', 'estimated_vae_decode_tflops_per_video',
                'predictor_tflops', 'predictor_network_cuda_seconds', 'predictor_decision_wall_seconds',
                'latent_feature_wall_seconds'):
        row[key] = 1.
    return row


class SuiteTests(unittest.TestCase):
    def test_partition_exact_cover(self):
        shards = suite.partition(prompt_rows())
        self.assertEqual([len(rows) for rows in shards.values()], [50] * 4)
        ids = [r['sample_id'] for rs in shards.values() for r in rs]
        self.assertEqual(len(set(ids)), 200)
        with self.assertRaises(ValueError):
            suite.partition(prompt_rows()[:-1] + [prompt_rows()[0]])

    def test_ratio_of_sums_and_no_partial_aggregation(self):
        base = [measured(f'p{i}', 100 + i) for i in range(200)]
        candidate = [measured(f'p{i}', (100 + i) / 2) for i in range(200)]
        self.assertEqual(suite.summarize(base, candidate)['latency_speedup'], 2)
        with self.assertRaises(ValueError):
            suite.summarize(base, candidate[:-1])

    def test_prerequisite_checks_reports_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertFalse(suite.prerequisite_complete(root))
            suite.dump(root / 'analysis/VALIDATION.json', {'status': 'pass'})
            (root / 'analysis/results_long.csv').write_text('test')
            (root / 'analysis/report.html').write_text('test')
            suite.dump(root / 'COMPLETE.json', dict(status='complete', modes=12, targets=3,
                candidate_conditions=36, prompt_count=5,
                analysis_sha256=suite.sha256(root / 'analysis/results_long.csv'),
                report_sha256=suite.sha256(root / 'analysis/report.html')))
            self.assertTrue(suite.prerequisite_complete(root))
            (root / 'analysis/report.html').write_text('changed')
            with self.assertRaises(ValueError):
                suite.prerequisite_complete(root)

    def test_resume_full_matrix_to_report_without_gpu(self):
        """Exercise all shard validation, 600-row joins and final report generation."""
        with tempfile.TemporaryDirectory() as temporary:
            exp = Path(temporary)
            checkpoint = exp / 'epoch_328.pt'
            checkpoint.write_bytes(b'synthetic checkpoint')
            selection_dir = exp / 'ours21_random3000_12groups_v1_sea7_analysis'
            suite.dump(selection_dir / 'checkpoint_selection.json', dict(checkpoint=str(checkpoint),
                checkpoint_epoch=328, checkpoint_sha256=suite.sha256(checkpoint)))
            profile = exp / 'wan21_seacache_threshold_collection_v1/calflops_profile.json'
            suite.dump(profile, dict(input=dict(video_shape_fhw=[81, 480, 832], transformer_blocks=30)))
            def create(path, description):
                path.mkdir()
                (path / 'README.md').write_text(description)
                return path
            with patch.object(suite, 'EXP_ROOT', exp), patch.object(suite, 'create_result', create), \
                    patch.object(sys, 'argv', ['run_suite', '--prepare-only']):
                suite.main()
            root = exp / suite.NAME
            config = suite.read_json(root / 'config.json')
            for g in suite.GPUS:
                ids = config['shard_ids'][str(g)]
                base_rows = [measured(sid, 200.) for sid in ids]
                for label in suite.LABELS:
                    d = root / 'shards' / f'gpu{g}' / label
                    k = None if label == 'baseline' else int(label[1:])
                    suite.dump(d / 'run.json', dict(prompts=[dict(sample_id=sid) for sid in ids],
                        method='baseline' if k is None else 'ours', state_mode='sea7', skip_budget=k,
                        protocol=suite.PROTOCOL, flops_profile_sha256=suite.sha256(profile),
                        gpu_uuid=f'GPU-{g}', policy_sha256=suite.sha256(checkpoint)))
                    suite.dump(d / 'COMPLETE.json', dict(videos=50))
                    for sid in ids:
                        (d / 'videos').mkdir(exist_ok=True)
                        (d / 'videos' / f'{sid}.mp4').write_bytes(b'synthetic video')
                        suite.dump(d / 'timings' / f'{sid}.json', {})
                        suite.dump(d / 'traces' / f'{sid}.json', dict(step_reuse=k, total_steps=50))
                    if k is not None:
                        suite.dump(d / 'performance.json', dict(baseline=base_rows,
                            candidate=[measured(sid, 100.) for sid in ids]))
                        suite.dump(d / 'quality/summary.json', dict(video_count=50, frame_count_total=4050))
                        suite.write_csv(d / 'quality/per_video.csv', [dict(video_id=sid, psnr_rgb_db=25.,
                            ssim_rgb=.8, lpips_alex_v0_1_spatial=.1) for sid in ids])
            dimensions = suite.read_json(suite.OFFICIAL / 'VbenchEvaluation/dimensions.json')['dimensions']
            for label in suite.LABELS:
                suite.dump(root / 'vbench' / label / 'vbench200_aggregate_scores.json', dict(
                    raw_dimension_scores={d: .8 for d in dimensions}, official_full_vbench_score=False,
                    aggregate_scores=dict(total_score=.8, quality_score=.8, semantic_score=.8)))
            with patch.object(suite, 'EXP_ROOT', exp), patch.object(sys, 'argv', ['run_suite', '--resume']), \
                    patch.object(suite, 'prerequisite_complete', return_value=True), \
                    patch.object(suite.subprocess, 'check_output', return_value=''), \
                    patch.object(suite.subprocess, 'run', side_effect=AssertionError('GPU call forbidden')):
                suite.main()
            self.assertEqual(suite.read_json(root / 'COMPLETE.json')['conditions'], 3)
            self.assertEqual(len(list((root / 'merged/K23/videos').glob('*.mp4'))), 200)
            self.assertEqual(len((root / 'analysis/per_video.csv').read_text().splitlines()), 601)


if __name__ == '__main__':
    unittest.main()
