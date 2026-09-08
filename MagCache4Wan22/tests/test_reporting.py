from __future__ import annotations
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "runtime"))
sys.path.insert(0, str(PROJECT / "experiments/performance_t2v_a14b"))
sys.path.insert(0, str(PROJECT / "experiments/paired_benchmark"))
from common import artifact, external_output, read_json
from aggregate_performance import aggregate
from run_benchmark import load_prompts


def fixture(root):
    def save(path, value):
        path.write_text(json.dumps(value))
        return artifact(path)
    protocol = read_json(PROJECT / "configs/wan22_t2v_a14b_50step_dpmpp.json")
    source = {"wan22_commit": "fixture", "magcache_commit": "fixture"}
    profile = dict(schema="magcache4wan22_calflops_profile_v2",
                   source=dict(prepared_manifest=source, checkpoint_dir=str(root)),
                   input=dict(video_shape_fhw=[45, 480, 832], output_fps=16, sampling_steps=50,
                              solver="dpm++", shift=12., guide_scale_low_high=[3., 4.], boundary=.875,
                              seed=42, parameter_dtype="bfloat16", stage_steps={"high": 32, "low": 18}),
                   component_profiles={key: dict(calls_per_video=count, estimated_flops_per_video=cost * 1e12,
                                                estimated_tflops_per_video=cost)
                                       for key, count, cost in (("t5", 2, 2.), ("vae_decode", 1, 3.))},
                   stages={stage: dict(model=dict(transformer_blocks=40),
                                       branches={branch: dict(estimated_full_flops=10e12,
                                                              estimated_always_on_flops=1e12)
                                                 for branch in ("cond", "uncond")})
                           for stage in ("high", "low")})
    paths = []
    for mode, seconds in (("baseline", 200.), ("magcache", 100.)):
        calls = []
        for i in range(100):
            # Unequal CFG decisions catch accidental step-level reuse averaging.
            reused = mode == "magcache" and i % 2 == 1
            calls.append(dict(call_index=i, step_index=i // 2, model_stage="high" if i < 64 else "low",
                              cfg_branch="cond" if i % 2 == 0 else "uncond", blocks_executed=0 if reused else 40,
                              reuse=reused, full_compute=not reused, cuda_seconds=.5, host_span_seconds=.6))
        reuse = sum(c["reuse"] for c in calls)
        timing = dict(schema_version=2, status="success", implementation=mode, error=None,
                      pipeline_generate_wall_seconds=seconds, model_forward_call_count=100,
                      model_forward_cuda_seconds=50., full_compute_forward_calls=100 - reuse,
                      reuse_forward_calls=reuse, transformer_block_count_by_stage=dict(high=40, low=40), calls=calls,
                      component_latency={key: dict(call_count=count, cuda_seconds=cost, host_span_seconds=cost)
                                         for key, count, cost in (("t5", 2, 2.), ("dit", 100, 50.), ("vae_decode", 1, 3.))})
        run = dict(schema="magcache4wan22_run_v1", mode=mode, prompt="fixture prompt", sample_id="one",
                   checkpoint=str(root), protocol=protocol, method=dict(threshold=.06, K=2, retention_ratio=.4), source=source)
        video = root / (mode + ".mp4")
        video.write_bytes(b"unit-test-fixture-not-a-video")
        manifest = dict(**run, status="complete", run=save(root / (mode + ".run.json"), run),
                        video=artifact(video), timing=save(root / (mode + ".timing.json"), timing))
        if mode == "magcache":
            trace = dict(schema="magcache4wan22_trace_v1", method=run["method"],
                         calls=[dict(**c, action="reuse" if c["reuse"] else "recompute") for c in calls])
            manifest["trace"] = save(root / "trace.json", trace)
        path = root / (mode + ".manifest.json")
        save(path, manifest)
        paths.append(path)
    return paths, profile


class ReportingTests(unittest.TestCase):
    def test_component_and_independent_cfg_flops(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, profile = fixture(Path(temp))
            result = aggregate([paths[0]], [paths[1]], profile)
        self.assertEqual(result["speedup"], 2.)
        self.assertEqual(result["sums"]["baseline"]["estimated_dit_tflops"], 1000.)
        self.assertEqual(result["sums"]["magcache"]["estimated_dit_tflops"], 550.)
        self.assertEqual(result["sums"]["magcache"]["estimated_t5_tflops_per_video"], 2.)
        self.assertEqual(result["sums"]["magcache"]["estimated_vae_decode_tflops_per_video"], 3.)

    def test_mismatched_profile_and_tampered_trace_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, profile = fixture(root)
            bad = copy.deepcopy(profile)
            bad["input"]["video_shape_fhw"][0] = 81
            with self.assertRaisesRegex(ValueError, "profile input mismatch"):
                aggregate([paths[0]], [paths[1]], bad)
            trace = read_json(root / "trace.json")
            trace["calls"][1]["blocks_executed"] = 40
            (root / "trace.json").write_text(json.dumps(trace))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                aggregate([paths[0]], [paths[1]], profile)
            manifest = read_json(paths[1])
            manifest["trace"] = artifact(root / "trace.json")
            paths[1].write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "trace and measured"):
                aggregate([paths[0]], [paths[1]], profile)

    def test_missing_components_and_duplicate_pairs_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, profile = fixture(root)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                aggregate([paths[0], paths[0]], [paths[1], paths[1]], profile)
            timing_path = root / "magcache.timing.json"
            timing = read_json(timing_path)
            del timing["component_latency"]["t5"]
            timing_path.write_text(json.dumps(timing))
            manifest = read_json(paths[1])
            manifest["timing"] = artifact(timing_path)
            paths[1].write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "t5 call count"):
                aggregate([paths[0]], [paths[1]], profile)

    def test_vbench_protocol_coverage_and_result_root(self):
        prompts = PROJECT.parent / "Vbench200/prompts.jsonl"
        self.assertEqual(len(load_prompts(prompts, "standard")), 200)
        with tempfile.TemporaryDirectory() as temp:
            subset = Path(temp) / "prompts.jsonl"
            subset.write_text(prompts.read_text().splitlines()[0] + "\n")
            with self.assertRaisesRegex(ValueError, "16 dimensions"):
                load_prompts(subset, "standard")
            self.assertEqual(len(load_prompts(subset, "custom")), 1)
        with self.assertRaises(ValueError):
            external_output(PROJECT / "bad-result")

if __name__ == "__main__":
    unittest.main()
