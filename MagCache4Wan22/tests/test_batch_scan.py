from __future__ import annotations
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "runtime"))
from batch import jobs_for_worker, validate_completed, worker
from common import artifact, read_json
from scan import grid_conditions, official_schedule, preset_conditions, select_targets, select_preset_targets
from generation import generate_native
from test_official_parity import import_complete_upstream, inputs, models, run_forward
from official import load_official, method_args
from test_reporting import fixture


class BatchScanTests(unittest.TestCase):
    def test_grid_schedule_matches_complete_official_forward(self):
        original = import_complete_upstream()
        core = load_official(lambda *a: None)
        for threshold, k, retention in ((.06, 2, .4), (.8, 6, .05), (.2, 3, .1), (.04, 0, .2),
                                       (.075, 2, .2), (.386, 4, .2), (.2, 5, .1)):
            pair = models()
            with contextlib.redirect_stdout(io.StringIO()):
                original.init_magcache(pair[0], core.ratios, method_args(threshold, k, retention), 32)
                _, counts = run_forward(pair, inputs(), 32)
                planned = official_schedule(threshold, k, retention)
            self.assertEqual(planned, ["reuse" if n == 0 else "recompute" for n in counts])
        self.assertEqual(official_schedule(.06, 2, .4).count("reuse"), 33)

    def test_dedup_and_targets_do_not_confuse_compute_proxy_with_latency(self):
        conditions = grid_conditions([0., .01], [0], [.1, .4])
        self.assertEqual(len(conditions), 1)
        self.assertEqual(len(conditions[0]["equivalent_parameters"]), 3)
        performance = {"a": dict(speedup=1.81, method={"K": 2}, dit_flops_speedup=3.),
                       "b": dict(speedup=2.42, method={"K": 4}, dit_flops_speedup=4.)}
        selected = select_targets(performance)["targets"]
        self.assertEqual([r["status"] for r in selected], ["hit", "hit", "unmet"])
        self.assertEqual(selected[-1]["measured_speedup"], 2.42)
        with self.assertRaisesRegex(ValueError, "initial CFG"):
            grid_conditions([.06], [2], [.01])

    def test_native_fixed_protocol(self):
        calls = []
        class Pipeline:
            def generate(self, *args, **kwargs):
                calls.append((args, kwargs))
                return "video"
        self.assertEqual(generate_native(Pipeline(), "prompt"), "video")
        self.assertEqual(calls, [(("prompt",), dict(size=(832, 480), frame_num=45, shift=12,
                          sample_solver="dpm++", sampling_steps=50, guide_scale=(3., 4.), seed=42,
                          offload_model=True))])

    def test_target_preset_grid_keeps_centers_and_all_aliases(self):
        presets = read_json(PROJECT / "experiments/targeted_threshold_scan/presets.json")
        conditions, groups = preset_conditions(presets)
        self.assertEqual([g["grid_point_count"] for g in groups], [31, 241, 41])
        by_id = {c["id"]: c for c in conditions}
        centers = []
        for group in groups:
            candidates = [by_id[name] for name in group["condition_ids"]]
            self.assertEqual(candidates[0]["threshold"], group["recommended_threshold"])
            centers.append(candidates[0]["planned_full_calls"])
            params = [p for c in candidates for p in [c] + c["equivalent_parameters"]]
            self.assertEqual(len(params), group["grid_point_count"])
            self.assertEqual(len(set(p["threshold"] for p in params)), group["grid_point_count"])
            self.assertTrue(all(p["K"] == group["K"] and p["retention_ratio"] == group["retention_ratio"]
                                for p in params))
            self.assertAlmostEqual(group["tolerance"], .02 * group["target"])
        self.assertEqual(centers, [52, 37, 28])
        self.assertLess(len(conditions), 313)

    def test_target_selection_cannot_borrow_a_closer_other_rk_point(self):
        groups = [dict(target=1.8, tolerance=.036, tolerance_kind="relative", relative_tolerance=.02,
                       recommended_threshold=.075, K=2, retention_ratio=.2, condition_ids=["a", "b"])]
        performance = {"a": dict(speedup=1.75, method=dict(K=2, retention_ratio=.2)),
                       "b": dict(speedup=1.88, method=dict(K=2, retention_ratio=.2)),
                       "other_rk": dict(speedup=1.8, method=dict(K=4, retention_ratio=.1))}
        selected = select_preset_targets(performance, groups)["targets"][0]
        self.assertEqual(selected["nearest_condition"], "a")
        self.assertEqual(selected["status"], "unmet")
        performance["a"]["speedup"] = 1.82
        self.assertEqual(select_preset_targets(performance, groups)["targets"][0]["status"], "hit")
        performance["a"]["method"]["K"] = 4
        with self.assertRaisesRegex(ValueError, "R/K"):
            select_preset_targets(performance, groups)

    def test_presets_validate_decimal_grid_and_requested_targets(self):
        presets = read_json(PROJECT / "experiments/targeted_threshold_scan/presets.json")
        conditions, groups = preset_conditions(presets, targets=[3.0], tolerance=.01)
        self.assertEqual([g["target"] for g in groups], [3.0])
        self.assertEqual(groups[0]["tolerance"], .01)
        self.assertTrue(all(c["K"] == 5 and c["retention_ratio"] == .1 for c in conditions))
        for targets in ([2.0], [1.8, 1.8], []):
            with self.assertRaises(ValueError):
                preset_conditions(presets, targets=targets)
        for update in (dict(step=0), dict(stop=.059), dict(step=.007), dict(start=float("nan"))):
            invalid = copy.deepcopy(presets)
            invalid["targets"][0]["threshold_range"].update(update)
            with self.assertRaises(ValueError):
                preset_conditions(invalid)
        invalid = copy.deepcopy(presets)
        invalid["targets"][0]["recommended_threshold"] = .0755
        with self.assertRaisesRegex(ValueError, "lie on the grid"):
            preset_conditions(invalid)

    def test_manual_grid_cannot_silently_override_target_presets(self):
        import suite
        argv = ["--source", "/source", "--checkpoint", "/checkpoint", "--output-dir", "/result",
                "--target-presets", str(PROJECT / "experiments/targeted_threshold_scan/presets.json"),
                "--ks", "6"]
        args = suite.parser(scan=True).parse_args(argv)
        with patch("suite.validate_source", return_value={}), patch("suite.checkpoint_path"), \
             patch("suite.load_prompts", return_value=[{}]):
            with self.assertRaisesRegex(ValueError, "manual R/K/E grid"):
                suite.build_plan(args, scan=True)

    def test_deferred_evaluation_resume_never_marks_partial_suite_complete(self):
        import suite
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            plan = dict(target_speedups=[], gpu_ids=["0"])
            argv = ["--source", tmp, "--checkpoint", tmp, "--output-dir", str(root)]
            def performance(*args):
                (root / "performance.json").write_text('{}\n')
                return {}
            with patch("suite.check_environment"), patch("suite.build_plan", return_value=plan), \
                 patch("suite.external_output", return_value=root), patch("suite.index_result"), \
                 patch("suite.ensure_profile", return_value={}), patch("suite.launch_workers") as launch, \
                 patch("suite.measured_performance", side_effect=performance), \
                 patch("suite.evaluate", return_value={}) as evaluate, \
                 contextlib.redirect_stdout(io.StringIO()):
                suite.main(argv=argv + ["--defer-evaluation"])
                self.assertTrue((root / "GENERATION_COMPLETE.json").exists())
                self.assertFalse((root / "COMPLETE.json").exists())
                self.assertEqual(read_json(root / "status.json")["status"], "pending")
                evaluate.assert_not_called()
                suite.main(argv=argv + ["--resume"])
                self.assertTrue((root / "COMPLETE.json").exists())
                self.assertEqual(launch.call_count, 1)
                self.assertEqual(evaluate.call_count, 1)

    def test_one_load_same_gpu_shards_and_resume_without_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_root = root / "fixture"
            fixture_root.mkdir()
            _, profile = fixture(fixture_root)
            plan = dict(source=profile["source"]["prepared_manifest"], checkpoint=str(fixture_root),
                        prompts=[dict(sample_id="one", prompt_en="first"), dict(sample_id="two", prompt_en="second")],
                        gpu_ids=["7"], warmup_videos=1,
                        conditions=[dict(id="baseline", mode="baseline", threshold=.06, K=2, retention_ratio=.4),
                                    dict(id="candidate", mode="magcache", threshold=.06, K=2, retention_ratio=.4,
                                         schedule=["recompute" if i % 2 == 0 else "reuse" for i in range(100)])])
            # No Wan imports / real weights needed to exercise the process contract.
            plan["source"]["source"] = str(root)
            (root / "plan.json").write_text(json.dumps(plan))
            (root / "profile.json").write_text(json.dumps(profile))
            (root / "profile_artifact.json").write_text(json.dumps(artifact(root / "profile.json")))
            observed = []
            def generate(pipeline, core, options, expected, directory, **kwargs):
                observed.append((id(pipeline), expected["sample_id"], expected["mode"]))
                directory.mkdir(parents=True)
                paths, _ = fixture(directory)
                selected = paths[0 if expected["mode"] == "baseline" else 1]
                manifest = read_json(selected)
                manifest.update(expected)
                Path(manifest["run"]["path"]).write_text(json.dumps(expected))
                manifest["run"] = artifact(manifest["run"]["path"])
                timing = read_json(manifest["timing"]["path"])
                timing.update(pipeline_init_wall_seconds=0., pipeline_lifecycle=kwargs["lifecycle"])
                Path(manifest["timing"]["path"]).write_text(json.dumps(timing))
                manifest["timing"] = artifact(manifest["timing"]["path"])
                if "trace" in manifest:
                    trace = read_json(manifest["trace"]["path"])
                    trace["method"] = expected["method"]
                    Path(manifest["trace"]["path"]).write_text(json.dumps(trace))
                    manifest["trace"] = artifact(manifest["trace"]["path"])
                destination = directory / "manifest.json"
                destination.write_text(json.dumps(manifest))
                return destination
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="7"), \
                 patch("batch.validate_source", return_value=plan["source"]), \
                 patch("batch.create_pipeline", return_value=(object(), 12.)) as create, \
                 patch("batch.load_official"), patch("batch.generate_run", side_effect=generate), \
                 patch("batch.generate_native", return_value="warmup") as warmup, \
                 patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
                worker(root, 0)
                self.assertEqual(create.call_count, 1)
                self.assertEqual(warmup.call_count, 1)
                self.assertEqual(len(observed), 4)
                self.assertEqual(len({r[0] for r in observed}), 1)
                worker(root, 0, resume=True)
                self.assertEqual(create.call_count, 1)
                self.assertEqual(read_json(root / "worker_status/worker_000.json")["pipeline_init_count"], 0)
                # Completed data is rechecked; prompt or physical GPU changes are rejected.
                condition = plan["conditions"][0]
                from batch import job_plan
                expected = job_plan(plan, plan["prompts"][0], condition)
                path = root / "runs/baseline/one/manifest.json"
                with self.assertRaisesRegex(ValueError, "lifecycle"):
                    validate_completed(path, expected, profile, "8")
                expected["prompt"] = "changed"
                with self.assertRaisesRegex(ValueError, "input mismatch"):
                    validate_completed(path, expected, profile, "7")
            plan["gpu_ids"] = ["0", "1"]
            for index in range(2):
                jobs = list(jobs_for_worker(plan, index))
                self.assertEqual(len({r[0]["sample_id"] for r in jobs}), 1)
                self.assertEqual([r[1]["mode"] for r in jobs], ["baseline", "magcache"])


if __name__ == "__main__":
    unittest.main()
