from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "threshold_scan_vbench8_t2v_a14b"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ThresholdScanTests(unittest.TestCase):
    def test_frozen_inputs_are_valid(self) -> None:
        plan = load_module("threshold_scan_plan", EXPERIMENT_DIR / "plan_scan.py")
        config, prompts = plan.validate_inputs()
        self.assertEqual(len(prompts), 11)
        self.assertEqual(
            len({dimension for row in prompts for dimension in row["dimension"]}),
            16,
        )
        self.assertEqual(len(config["thresholds"]), 16)
        self.assertEqual(config["target_speedups"], [1.8, 2.4, 3.0])

    def test_target_selection_keeps_observation_and_interpolation_separate(self) -> None:
        finalize = load_module(
            "threshold_scan_finalize", EXPERIMENT_DIR / "finalize_scan.py"
        )
        rows = []
        for threshold, speedup in ((0.15, 1.5), (0.20, 1.7), (0.25, 1.9)):
            rows.append(
                {
                    "threshold": threshold,
                    "latency_speedup_ratio_of_sums": speedup,
                    "psnr_rgb_db": 25.0,
                    "ssim_rgb": 0.9,
                    "lpips_alex_v0_1_spatial": 0.1,
                    "candidate_vbench_score": 0.7,
                }
            )
        result = finalize.nearest_targets(rows, [1.8])[0]
        self.assertEqual(result["nearest_measured_threshold"], 0.20)
        self.assertAlmostEqual(
            result["interpolation_diagnostic"]["linearly_interpolated_threshold"],
            0.225,
        )

    def test_persistent_worker_is_part_of_the_frozen_runner(self) -> None:
        worker = (EXPERIMENT_DIR / "scan_worker.py").read_text(encoding="utf-8")
        wrapper = (EXPERIMENT_DIR / "run_worker.sh").read_text(encoding="utf-8")
        self.assertIn("pipeline = wan.WanT2V(", worker)
        self.assertEqual(worker.count("pipeline = wan.WanT2V("), 1)
        self.assertIn("_PipelineProfiler(", worker)
        self.assertIn("scan_worker.py", wrapper)


if __name__ == "__main__":
    unittest.main()
