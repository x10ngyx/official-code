from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_DIR / "experiments" / "threshold_scan_vbench10_full_metrics"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ThresholdScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_module("threshold_scan_runner_test", EXPERIMENT_DIR / "run_scan.py")
        cls.analyzer = load_module(
            "threshold_scan_analyzer_test", EXPERIMENT_DIR / "analyze_scan.py"
        )

    def test_locked_grid_and_prompt_subset(self) -> None:
        self.assertEqual(len(self.runner.DEFAULT_SAMPLE_IDS), 10)
        self.assertEqual(len(set(self.runner.DEFAULT_SAMPLE_IDS)), 10)
        self.assertEqual(len(self.runner.DEFAULT_THRESHOLDS), 18)
        self.assertEqual(self.runner.DEFAULT_THRESHOLDS[0], 0.15)
        self.assertEqual(self.runner.DEFAULT_THRESHOLDS[-1], 0.80)
        front_steps = [
            right - left
            for left, right in zip(
                self.runner.DEFAULT_THRESHOLDS[:6],
                self.runner.DEFAULT_THRESHOLDS[1:6],
            )
        ]
        self.assertTrue(all(abs(step - 0.01) < 1e-12 for step in front_steps))
        self.assertGreater(
            self.runner.DEFAULT_THRESHOLDS[-1] - self.runner.DEFAULT_THRESHOLDS[-2],
            max(front_steps),
        )

    def test_trace_flops_uses_executed_block_fraction(self) -> None:
        actual = self.analyzer.estimate_trace_flops(
            [{"blocks_executed": 2}, {"blocks_executed": 0}],
            full_flops=100.0,
            always_on_flops=10.0,
            block_count=2,
        )
        self.assertEqual(actual, 110.0)

    def test_target_selection_retains_observed_and_interpolated_values(self) -> None:
        rows = [
            {"threshold": 0.15, "inference_speedup_ratio_of_sums": 1.75},
            {"threshold": 0.16, "inference_speedup_ratio_of_sums": 1.85},
            {"threshold": 0.20, "inference_speedup_ratio_of_sums": 2.10},
        ]
        selected = self.analyzer.select_target(rows, 1.8)
        self.assertTrue(selected["target_bracketed_by_observed_grid"])
        self.assertEqual(selected["nearest_observed"]["threshold"], 0.15)
        self.assertAlmostEqual(selected["linear_interpolation"]["threshold"], 0.155)


if __name__ == "__main__":
    unittest.main()
