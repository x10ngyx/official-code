from __future__ import annotations
import contextlib
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
from scan import grid_conditions, select_targets
from generation import generate_native
from test_reporting import fixture

class BatchScanTests(unittest.TestCase):

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
                        conditions=[dict(id="baseline", mode="baseline", threshold=.08, retention_ratio=.2),
                                    dict(id="candidate", mode="dicache", threshold=.08, retention_ratio=.2)])
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
                self.assertEqual([r[1]["mode"] for r in jobs], ["baseline", "dicache"])

    def test_scan_does_not_predict_feature_dependent_schedules(self):
        conditions = grid_conditions([.08, .2, .08], [.2, .4])
        self.assertEqual(len(conditions), 4)
        self.assertTrue(all("schedule" not in c and "K" not in c for c in conditions))
        measured = {"a": dict(speedup=1.81, method={}, dit_flops_speedup=9.),
                    "b": dict(speedup=2.42, method={}, dit_flops_speedup=10.)}
        selected = select_targets(measured, [1.8, 2.4, 3.], .1)["targets"]
        self.assertEqual([r["status"] for r in selected], ["matched", "matched", "unmatched"])
        self.assertEqual(selected[-1]["speedup"], 2.42)
        with self.assertRaises(ValueError):
            grid_conditions([.08], [.001])

if __name__ == "__main__":
    unittest.main()
