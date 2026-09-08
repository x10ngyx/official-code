from __future__ import annotations
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("targeted_scan", PROJECT / "experiments/targeted_threshold_scan/run_scan.py")
scan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scan)


def point(threshold, speedup):
    return dict(method=dict(threshold=threshold, retention_ratio=.2, probe_depth=1), speedup=speedup)


class TargetedScanTests(unittest.TestCase):
    def test_speed_focus_avoids_outward_and_out_of_range_refinement(self):
        config = dict(scan.read_json(PROJECT / "experiments/speedrange_gpu123/presets.json"),
                      target_speedups=[3.0])
        scan.validate_presets(config)
        values = scan.refine({"a": point(.2, 2.0), "b": point(.4, 3.6)}, config)
        self.assertEqual(values, [.3])
        self.assertEqual(scan.refine({"a": point(.2, 3.8), "b": point(.4, 3.9)}, config), [.1])
        values = scan.refine({"a": point(.2, 1.1), "b": point(.4, 1.2)}, config)
        self.assertEqual(values, [.8])

    def setUp(self):
        self.config = scan.read_json(PROJECT / "experiments/targeted_threshold_scan/presets.json")

    def test_official_constants_and_relative_tolerance(self):
        scan.validate_presets(self.config)
        rows = scan.selection({"a": point(.1, 1.835), "b": point(.3, 2.449), "c": point(.8, 2.95)}, self.config)["targets"]
        self.assertEqual([x["status"] for x in rows], ["matched", "unmatched", "matched"])
        self.assertAlmostEqual(rows[1]["tolerance"], .048)
        with self.assertRaises(ValueError):
            scan.validate_presets(dict(self.config, retention_ratio=.1))

    def test_refinement_handles_nonmonotonic_points_and_boundaries(self):
        config = dict(self.config, target_speedups=[2.4])
        measured = {"a": point(.1, 2.), "b": point(.2, 2.8), "c": point(.4, 2.1)}
        values = scan.refine(measured, config)
        self.assertIn(.15, values)
        self.assertIn(.3, values)
        self.assertEqual(scan.refine({"a": point(8., 2.)}, config), [4.])
        self.assertEqual(scan.refine({"a": point(.1, 2.4)}, config), [])
        self.assertEqual(scan.refine({"a": point(.001, 2.)}, config), [])

    def test_execute_resume_profiles_and_selected_quality(self):
        config = dict(self.config, coarse_thresholds=[.1, .3], target_speedups=[1.8], max_rounds=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = scan.parser().parse_args(["--output-dir", tmp, "--gpu-ids", "2", "--defer-evaluation"])
            calls = []
            def run_round(scan=False, argv=None):
                parsed = scan_module.suite.parser(True).parse_args(argv)
                calls.append(parsed)
                directory = parsed.output_dir
                directory.mkdir(exist_ok=True)
                data = {f"c{i}": point(t, { .1: 1.6, .3: 2.0, .2: 1.8, .05: 1.3 }[t])
                        for i, t in enumerate(parsed.thresholds)}
                for name, value in {"profile_artifact.json": {"path": str(root / "profile.json")},
                                    "performance.json": data,
                                    "plan.json": {"conditions": [{"id": "baseline"}] + [{"id": x} for x in data]}}.items():
                    (directory / name).write_text(json.dumps(value))
            scan_module = scan
            with patch.object(scan.suite, "main", side_effect=run_round), \
                 patch.object(scan.suite, "evaluate", return_value={"vbench_score": {}}) as evaluate:
                scan.execute(args, config, root)
                self.assertEqual(len(calls), 2)
                self.assertIsNone(calls[0].calflops_profile)
                self.assertEqual(calls[1].calflops_profile, root / "profile.json")
                self.assertFalse((root / "COMPLETE.json").exists())
                self.assertTrue(scan.read_json(root / "SCAN_COMPLETE.json")["all_targets_matched"])
                evaluate.assert_not_called()
                args.resume = True
                args.defer_evaluation = False
                scan.execute(args, config, root)
                self.assertTrue(all(x.resume for x in calls[2:]))
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual([c["id"] for c in evaluate.call_args.args[1]["conditions"]], ["baseline", "c1"])
                self.assertTrue((root / "COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
