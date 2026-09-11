"""CPU contracts for the five-prompt exact-K benchmark pipeline."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENT = PROJECT / 'experiments/vbench5_speed_targets_v1'
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(EXPERIMENT))

from ours4wan21.contracts import dump, sha256
from build_subset import load_rows, select_maximum_coverage
from pipeline_lib import MODES, TARGETS, TARGET_K, validate_quality, vbench_enabled
from ours4wan21.inference import physical_gpu_uuid
import build_report


class VBenchFivePipelineTests(unittest.TestCase):
    def test_skip_vbench_retains_video_quality_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(vbench_enabled(root))
            dump(root / 'VBENCH_SKIPPED_BY_USER.json', dict(status='skipped_by_user', video_metrics_enabled=True))
            self.assertFalse(vbench_enabled(root))
            dump(root / 'quality/COMPLETE.json', dict(status='quality_complete', vbench_status='skipped_by_user'))
            dump(root / 'quality/video_metrics/summary.json', dict(video_count=5, frame_count_total=405))
            validate_quality(root, 5, require_vbench=False)
            with self.assertRaises(FileNotFoundError):
                validate_quality(root, 5)
            dump(root / 'quality/video_metrics/summary.json', dict(video_count=4, frame_count_total=324))
            with self.assertRaises(ValueError):
                validate_quality(root, 5, require_vbench=False)

    def test_physical_gpu_uuid_supports_torch_and_nvidia_smi_fallbacks(self):
        native_torch = SimpleNamespace(cuda=SimpleNamespace(
            get_device_properties=lambda _: SimpleNamespace(uuid='GPU-native')))
        self.assertEqual(physical_gpu_uuid(native_torch), 'GPU-native')

        legacy_torch = SimpleNamespace(cuda=SimpleNamespace(
            get_device_properties=lambda _: SimpleNamespace()))
        calls = []
        def process_query(command, text):
            calls.append(command)
            return f'{os.getpid()}, GPU-process\n999, GPU-other\n'
        self.assertEqual(physical_gpu_uuid(legacy_torch, check_output=process_query),
                         'GPU-process')
        self.assertEqual(len(calls), 1)

        def index_query(command, text):
            if '--query-compute-apps=pid,gpu_uuid' in command:
                return ''
            self.assertIn('--id=2', command)
            return 'GPU-index-two\n'
        self.assertEqual(physical_gpu_uuid(
            legacy_torch, environ={'CUDA_VISIBLE_DEVICES': '2'},
            check_output=index_query), 'GPU-index-two')

    def test_frozen_subset_is_exact_maximum_coverage(self):
        source = PROJECT.parent / 'Vbench200/prompts.jsonl'
        selected, covered = select_maximum_coverage(load_rows(source))
        self.assertEqual([row['sample_id'] for row in selected], [
            'vbench200_001', 'vbench200_016', 'vbench200_056',
            'vbench200_135', 'vbench200_159'])
        self.assertEqual(len(covered), 10)
        frozen = [json.loads(line) for line in (EXPERIMENT / 'subset/prompts.jsonl').read_text().splitlines()]
        self.assertEqual(frozen, selected)
        manifest = json.loads((EXPERIMENT / 'subset/selection_manifest.json').read_text())
        self.assertFalse(manifest['official_full_vbench_score'])
        self.assertEqual(manifest['selected_sample_ids'], [row['sample_id'] for row in selected])

    def test_fixed_calibrated_target_k_has_no_adaptive_path(self):
        self.assertEqual(dict(zip(TARGETS, TARGET_K)), {1.8: 23, 2.4: 29, 3.0: 35})
        source = (EXPERIMENT / 'run_pipeline.py').read_text()
        self.assertIn('adaptive_rescan=False', source)
        self.assertNotIn('propose_k', source)
        self.assertNotIn('isotonic', source)

    def test_report_builder_emits_full_technical_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = root / 'analysis'
            analysis.mkdir()
            rows = []
            for index, mode in enumerate(MODES):
                for target in TARGETS:
                    rows.append(dict(mode=mode, label=f'Mode{index:02d}',
                        kind='control' if index < 2 else 'feature', target_speedup=target,
                        target_label=f'{target:.1f}x', skip_budget=round(target * 10),
                        achieved_latency_speedup=target * .99, target_relative_error=.01,
                        calibration_tolerance_met=True, candidate_generate_seconds_mean=10 / target,
                        candidate_t5_cuda_seconds_mean=.2,
                        candidate_dit_cuda_seconds_mean=8 / target,
                        candidate_vae_cuda_seconds_mean=1.2,
                        candidate_dit_tflops_mean=100 / target,
                        t5_tflops_per_video=9.7, vae_tflops_per_video=274.2,
                        predictor_tflops_mean=1e-5, psnr_rgb_db=24 - target,
                        ssim_rgb=.8, lpips_alex_v0_1_spatial=.12,
                        vbench_custom_score=.7, vbench_custom_delta=-.01,
                        latent_feature_wall_seconds_mean=index * .001,
                        predictor_decision_seconds_mean=.002, selected_epoch=350))
            results_path = analysis / 'results_long.csv'
            with results_path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            with (analysis / 'target_k_mapping.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['target_speedup', 'skip_budget',
                    'fitted_branch_reuse_calls', 'fitted_step_reuse', 'adaptive_rescan'])
                writer.writeheader()
                for target, skip_budget in zip(TARGETS, TARGET_K):
                    writer.writerow(dict(target_speedup=target, skip_budget=skip_budget,
                        fitted_branch_reuse_calls=2 * skip_budget,
                        fitted_step_reuse=skip_budget, adaptive_rescan=False))
            dump(analysis / 'VALIDATION.json', dict(status='pass', modes=12, targets=3,
                conditions=36, prompt_count=5, frames_per_video=81,
                official_full_vbench_score=False))
            dump(analysis / 'COMPLETE.json', dict(status='complete', results_sha256=sha256(results_path)))
            with patch.object(sys, 'argv', ['build_report.py', '--result-root', str(root)]):
                build_report.main()
            artifact = json.loads((analysis / 'artifact.json').read_text())
            self.assertEqual(artifact['surface'], 'report')
            self.assertEqual(len(artifact['manifest']['charts']), 5)
            self.assertEqual(len(artifact['manifest']['tables']), 2)
            bodies = '\n'.join(block.get('body', '') for block in artifact['manifest']['blocks'])
            for title in ('技术摘要', '数据、协议与指标口径', '实验设计与稳健性检查',
                          '限制与不确定性', '建议的后续验收', '仍需回答的问题'):
                self.assertIn(title, bodies)
            # The reduced-scope report must accept absent scores without fabricating zeros.
            for row in rows:
                row['vbench_custom_score'] = None
                row['vbench_custom_delta'] = None
            with results_path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            validation = json.loads((analysis / 'VALIDATION.json').read_text())
            validation['vbench_enabled'] = False
            dump(analysis / 'VALIDATION.json', validation)
            dump(analysis / 'COMPLETE.json', dict(status='complete', results_sha256=sha256(results_path)))
            with patch.object(sys, 'argv', ['build_report.py', '--result-root', str(root)]):
                build_report.main()
            artifact = json.loads((analysis / 'artifact.json').read_text())
            self.assertEqual(len(artifact['manifest']['charts']), 4)
            self.assertEqual(len(artifact['manifest']['cards']), 4)
            self.assertNotIn('vbench_custom_score', [c['field'] for c in artifact['manifest']['tables'][0]['columns']])
            self.assertTrue(all(row['vbench_custom_score'] is None for row in artifact['snapshot']['datasets']['results']))


if __name__ == '__main__':
    unittest.main()
