from __future__ import annotations

import json
import unittest
from pathlib import Path


DATA_PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL_CODE = DATA_PROJECT.parents[1]


class StaticContractTests(unittest.TestCase):
    def test_pending_mapping_is_really_blank(self) -> None:
        payload = json.loads((DATA_PROJECT / "configs/speed_threshold_mapping.pending.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["calibration_status"], "pending")
        self.assertIsNone(payload["threshold_bounds"])
        self.assertIsNone(payload["mapping"])
        self.assertIsNone(payload["fit_source"])

    def test_speedup_uses_inference_time(self) -> None:
        source = (DATA_PROJECT / "src/ours4wan21_data/performance.py").read_text(encoding="utf-8")
        self.assertIn('baseline_seconds / candidate_seconds', source)
        self.assertIn('pipeline_generate_wall_seconds', source)
        self.assertNotIn('process_wall_seconds', source)

    def test_launcher_sets_all_thread_limits(self) -> None:
        launchers = (
            DATA_PROJECT / "experiments/random_threshold_collection_v1/launch_4gpu.sh",
            DATA_PROJECT / "experiments/seacache_threshold_collection_v1/launch_4gpu.sh",
        )
        for launcher in launchers:
            source = launcher.read_text(encoding="utf-8")
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                self.assertIn(f"export {key}=1", source)

    def test_launchers_use_only_the_project_wan22_environment(self) -> None:
        launchers = (
            DATA_PROJECT / "experiments/random_threshold_collection_v1/launch_4gpu.sh",
            DATA_PROJECT / "experiments/seacache_threshold_collection_v1/launch_4gpu.sh",
        )
        for launcher in launchers:
            source = launcher.read_text(encoding="utf-8")
            self.assertIn("WAN22_PYTHON", source)
            self.assertIn("-n wan2.2 python", source)
            self.assertNotIn("WAN21_PYTHON", source)
            self.assertNotIn("-n Wan2.1", source)

    def test_random_stage_queue_waits_without_signalling_other_jobs(self) -> None:
        queue = (
            DATA_PROJECT
            / "experiments/random_threshold_collection_v1/queue_after_calibration.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("baselines_ready", queue)
        self.assertIn("mapping_ready", queue)
        self.assertIn("nvidia-smi", queue)
        self.assertIn('launch_4gpu.sh\" candidates', queue)
        self.assertIn('launch_4gpu.sh\" finalize', queue)
        self.assertNotIn("kill ", queue)
        self.assertNotIn("pkill", queue)

    def test_random_finalize_can_skip_vbench_for_scoped_training_data(self) -> None:
        launcher = (
            DATA_PROJECT
            / "experiments/random_threshold_collection_v1/launch_4gpu.sh"
        ).read_text(encoding="utf-8")
        audit = (DATA_PROJECT / "src/ours4wan21_data/audit.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("skip_vbench_${vbench_scope}.json", launcher)
        self.assertIn("skipping VBench by archive control", launcher)
        self.assertIn("audit_args+=(--skip-vbench)", launcher)
        self.assertIn('"--skip-vbench"', audit)
        self.assertIn('"vbench_required": require_vbench', audit)

    def test_seacache_grid_matches_frozen_wan22_list(self) -> None:
        payload = json.loads(
            (DATA_PROJECT / "configs/seacache_thresholds.wan22_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            payload["thresholds"],
            [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70],
        )
        self.assertEqual(payload["thresholds_per_prompt"], 3)
        self.assertTrue(payload["sampling_without_replacement"])

    def test_no_offload_runtime_places_t5_on_the_visible_gpu(self) -> None:
        runtime = (
            DATA_PROJECT / "src/ours4wan21_data/runtime.py"
        ).read_text(encoding="utf-8")
        self.assertIn("self.text_encoder.model.to(self.device)", runtime)
        self.assertIn(
            "pipeline.text_encoder.model.to(pipeline.device)",
            runtime,
        )

    def test_filtered_distance_is_collected_and_published_per_cfg_branch(self) -> None:
        controller = (DATA_PROJECT / "src/ours4wan21_data/controller.py").read_text(encoding="utf-8")
        publisher = (DATA_PROJECT / "src/ours4wan21_data/publisher.py").read_text(encoding="utf-8")
        for field in (
            "filtered_relative_l1",
            "accumulated_distance_before",
            "accumulated_distance_with_current",
            "accumulated_distance_after",
            "previous_step_same_cfg_branch",
            "sea_filtered_first_block_modulated_input",
        ):
            self.assertIn(field, controller)
        self.assertIn('BRANCHES = ("cond", "uncond")', controller)
        self.assertIn("branch_transitions.jsonl", publisher)
        self.assertIn("branch_transitions.csv", publisher)

    def test_official_code_shared_resources_are_used(self) -> None:
        shared = (
            OFFICIAL_CODE / "VideoMetrics/evaluate.py",
            OFFICIAL_CODE / "VideoMetrics/video_metrics/evaluator.py",
            OFFICIAL_CODE / "CalflopsEvaluation/calflops_eval/manual_ops.py",
            DATA_PROJECT.parent / "upstream_lock.json",
        )
        self.assertTrue(all(path.is_file() for path in shared))
        collector = (DATA_PROJECT / "src/ours4wan21_data/collector.py").read_text(encoding="utf-8")
        profile = (DATA_PROJECT / "experiments/calflops_profile_v1/profile_wan21_dit.py").read_text(encoding="utf-8")
        self.assertIn("FullReferenceMetricEvaluator", collector)
        self.assertIn('SELECTED_METRICS = ("psnr", "ssim", "lpips")', (
            DATA_PROJECT / "src/ours4wan21_data/metrics.py"
        ).read_text(encoding="utf-8"))
        self.assertIn('OFFICIAL_CODE / "CalflopsEvaluation"', profile)
        self.assertFalse((DATA_PROJECT / "src/ours4wan21_data/psnr.py").exists())
        self.assertFalse((DATA_PROJECT / "src/ours4wan21_data/flops_formula.py").exists())
        self.assertFalse((DATA_PROJECT / "tools/compute_psnr.py").exists())


if __name__ == "__main__":
    unittest.main()
