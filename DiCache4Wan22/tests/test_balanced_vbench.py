import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/vbench200_balanced_gpu123"))
import rebalance_gpu123 as balanced
from prepare import TARGETS


class BalancedVBenchTests(unittest.TestCase):
    def test_corrected_threshold_and_exact_remaining_coverage(self):
        self.assertEqual([t["threshold"] for t in TARGETS], [.075, .172, .396])
        contexts = {n: {"plan": {"prompts": [{"sample_id": str(i)} for i in range(200)]}}
                    for n in balanced.TARGET_DIRS}
        completed = {n: set() for n in contexts}
        completed[balanced.TARGET_DIRS[0]] = {"0", "1"}
        means = dict(zip(contexts, [547., 402., 343.]))
        with patch.object(balanced, "completed_and_means", return_value=(completed, means)):
            a = balanced.build_assignment(Path("unused"), contexts)
        jobs = [j for rows in a["assignments"].values() for j in rows]
        pairs = {(j["target"], j["sample_id"]) for j in jobs}
        self.assertEqual(len(jobs), 598)
        self.assertEqual(len(pairs), 598)
        self.assertNotIn((balanced.TARGET_DIRS[0], "0"), pairs)
        self.assertEqual(sum(j["target"] == balanced.TARGET_DIRS[1] for j in jobs), 200)
        loads = list(a["estimated_load_seconds"].values())
        self.assertLessEqual(max(loads) - min(loads), max(means.values()))

    def test_cross_gpu_accepts_valid_lifecycle_but_rejects_old_threshold(self):
        ctx = dict(plan={}, condition={}, profile={}, prompts={"sample": {}})
        manifest = dict(method={"threshold": .172}, timing={"path": "timing"})
        life = dict(persistent_pipeline=True, physical_gpu="3",
                    pipeline_init_accounted_in_this_sample=False,
                    profiler_freshly_installed_for_sample=True, warmup_videos=1)
        timing = dict(pipeline_lifecycle=life, pipeline_init_wall_seconds=0)
        with patch.object(balanced, "job_plan", return_value={"method": {"threshold": .172}}), \
             patch.object(balanced, "summarize_manifest", return_value=(manifest, {})), \
             patch.object(balanced, "read_json", return_value=timing):
            balanced.validate_any_gpu(Path("sample/manifest.json"), ctx)
            manifest["method"]["threshold"] = .072
            with self.assertRaisesRegex(ValueError, "input mismatch"):
                balanced.validate_any_gpu(Path("sample/manifest.json"), ctx)
            manifest["method"]["threshold"] = .172
            life["physical_gpu"] = "0"
            with self.assertRaisesRegex(ValueError, "lifecycle"):
                balanced.validate_any_gpu(Path("sample/manifest.json"), ctx)
