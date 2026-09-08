import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dicache_reuse", PROJECT / "experiments/vbench200_reuse_gpu123/run_gpu123.py")
reuse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reuse)


class ReusedBaselineTests(unittest.TestCase):
    def test_user_thresholds_and_candidate_only_plans(self):
        self.assertEqual([(x["gpu"], x["threshold"], x["retention_ratio"]) for x in reuse.TARGETS],
                         [("1", .075, .2), ("2", .072, .2), ("3", .396, .2)])
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(source=tmp, checkpoint=tmp, prompts=tmp, worker_ready_timeout=1800,
                                      calflops_profile=Path(tmp) / "profile.json", baseline_audit=Path(tmp) / "audit.json")
            def build(ns, scan):
                self.assertFalse(scan)
                self.assertEqual(ns.retention_ratio, .2)
                return dict(source={}, checkpoint=tmp, prompts=[dict(sample_id="one")],
                            conditions=[dict(id="baseline"), dict(id="candidate", threshold=ns.threshold, mode="dicache")])
            def read(path):
                return dict(source=dict(prepared_manifest={})) if Path(path).name == "profile.json" else dict(checkpoint=tmp)
            with patch.object(reuse, "build_plan", side_effect=build), patch.object(reuse, "read_json", side_effect=read), \
                 patch.object(reuse, "validate_profile"), patch.object(reuse, "validate_baseline", return_value=({}, {})):
                for target in reuse.TARGETS:
                    plan, _ = reuse.make_plan(args, target)
                    self.assertEqual(plan["generated_videos"], 1)
                    self.assertEqual(len(plan["conditions"]), 1)
                    self.assertEqual(plan["conditions"][0]["threshold"], target["threshold"])
                    self.assertEqual(plan["conditions"][0]["mode"], "dicache")
                    self.assertNotIn("schedule", plan["conditions"][0])
                    self.assertEqual(plan["requested_target_speedup"], target["target_speedup"])

    def test_reused_reporting_uses_measured_latency_and_probe_cost(self):
        baseline = dict(sample_id="one", generate_seconds=100., estimated_dit_tflops=1000.,
                        t5_cuda_seconds=2., dit_cuda_seconds=90., vae_decode_cuda_seconds=3.)
        candidate = dict(baseline, generate_seconds=40., estimated_dit_tflops=400., dit_cuda_seconds=30.)
        manifest = dict(sample_id="one", prompt="text", method=dict(threshold=.072), video=dict(path="candidate.mp4"))
        with tempfile.TemporaryDirectory() as tmp:
            plan = dict(conditions=[dict(id="candidate")], prompts=[dict(sample_id="one", prompt_en="text")],
                        gpu_ids=["2"], warmup_videos=1, requested_target_speedup=2.4,
                        baseline_reuse=dict(baseline_directory=tmp))
            with patch.object(reuse, "job_plan", return_value={}), \
                 patch.object(reuse, "validate_completed", return_value=manifest), \
                 patch.object(reuse, "summarize_manifest", return_value=(manifest, candidate)), \
                 patch.object(reuse, "baseline_metric_row", return_value=baseline), \
                 patch.object(reuse, "link_video"), patch.object(reuse, "read_json", return_value={}):
                value = reuse.aggregate_reused(Path(tmp), plan, {}, {"one": {}})
                self.assertEqual(value["speedup"], 2.5)
                self.assertEqual(value["requested_target_speedup"], 2.4)
                self.assertEqual(value["sums"]["dicache"]["estimated_dit_tflops"], 400.)
                self.assertEqual(value["sums"]["baseline"]["generate_seconds"], 100.)


if __name__ == "__main__":
    unittest.main()
