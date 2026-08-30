from __future__ import annotations

import importlib.util
import json
import unittest
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_DIR / "experiments" / "performance_t2v_1_3b"


def load_module(name: str, filename: str):
    path = EXPERIMENT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Vbench200PerformanceTests(unittest.TestCase):
    def test_trace_aggregation_uses_inference_only_latency_and_actual_cache_path(self):
        module = load_module("vbench200_performance", "aggregate_performance.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_paths = {}
            candidate_paths = {}
            for label, implementation, executed, latency, cuda_seconds, destination in (
                ("baseline", "wan21", 2, 20.0, 4.0, baseline_paths),
                ("seacache", "seacache", None, 10.0, 2.0, candidate_paths),
            ):
                path = root / f"{label}.json"
                calls = []
                for index in range(100):
                    blocks = executed if executed is not None else (2 if index % 2 == 0 else 0)
                    calls.append({"blocks_executed": blocks})
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "status": "success",
                            "implementation": implementation,
                            "pipeline_init_wall_seconds": 99.0,
                            "pipeline_generate_wall_seconds": latency,
                            "model_forward_cuda_seconds": cuda_seconds,
                            "transformer_block_count": 2,
                            "full_compute_forward_calls": sum(
                                call["blocks_executed"] == 2 for call in calls
                            ),
                            "reuse_forward_calls": sum(
                                call["blocks_executed"] == 0 for call in calls
                            ),
                            "calls": calls,
                            "component_latency": {
                                "t5": {"call_count": 2, "cuda_seconds": 1.0, "host_span_seconds": 1.1},
                                "dit": {"call_count": 100, "cuda_seconds": cuda_seconds, "host_span_seconds": cuda_seconds},
                                "vae_decode": {"call_count": 1, "cuda_seconds": 0.5, "host_span_seconds": 0.6},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                destination["sample"] = path

            baseline, _ = module.summarize_condition(
                label="baseline",
                paths=baseline_paths,
                expected_implementation="wan21",
                expected_count=1,
                full_forward_flops=1_000.0,
                always_on_flops=100.0,
                block_count=2,
                component_tflops={
                    "estimated_t5_tflops_per_video": 3.0,
                    "estimated_vae_decode_tflops_per_video": 4.0,
                },
            )
            seacache, _ = module.summarize_condition(
                label="seacache",
                paths=candidate_paths,
                expected_implementation="seacache",
                expected_count=1,
                full_forward_flops=1_000.0,
                always_on_flops=100.0,
                block_count=2,
                component_tflops={
                    "estimated_t5_tflops_per_video": 3.0,
                    "estimated_vae_decode_tflops_per_video": 4.0,
                },
            )
            self.assertEqual(
                baseline["end_to_end_inference_latency_seconds"]["mean"], 20.0
            )
            self.assertEqual(
                seacache["end_to_end_inference_latency_seconds"]["mean"], 10.0
            )
            self.assertEqual(baseline["estimated_dit_total_tflops"], 100_000 / 1e12)
            self.assertEqual(seacache["estimated_dit_total_tflops"], 55_000 / 1e12)
            self.assertEqual(seacache["total_full_compute_forward_calls"], 50)
            self.assertEqual(seacache["total_reuse_forward_calls"], 50)


if __name__ == "__main__":
    unittest.main()
