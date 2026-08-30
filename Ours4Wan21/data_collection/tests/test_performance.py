from __future__ import annotations

import sys
import unittest
from pathlib import Path


DATA_PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_PROJECT / "src"))

from ours4wan21_data.performance import compare_matched, summarize_timing  # noqa: E402


def timing(block_counts: list[int], seconds: float) -> dict:
    return {
        "schema_version": 2,
        "status": "success",
        "pipeline_generate_wall_seconds": seconds,
        "model_forward_cuda_seconds": seconds / 2,
        "component_latency": {
            "t5": {"call_count": 2, "cuda_seconds": 1.0, "host_span_seconds": 1.2},
            "dit": {"call_count": 100, "cuda_seconds": seconds / 2, "host_span_seconds": seconds},
            "vae_decode": {"call_count": 1, "cuda_seconds": 2.0, "host_span_seconds": 2.2},
        },
        "calls": [
            {"blocks_executed": count, "full_compute": count == 30, "reuse": count == 0}
            for count in block_counts
        ],
    }


PROFILE = {
    "scope": "synthetic",
    "per_model_forward": {
        "transformer_blocks": 30,
        "estimated_full_flops": 100e12,
        "estimated_always_on_flops": 10e12,
    },
    "component_profiles": {
        "t5": {
            "calls_per_video": 2,
            "estimated_flops_per_video": 2e12,
            "estimated_tflops_per_video": 2.0,
        },
        "vae_decode": {
            "calls_per_video": 1,
            "estimated_flops_per_video": 3e12,
            "estimated_tflops_per_video": 3.0,
        },
    },
}


class PerformanceTests(unittest.TestCase):
    def test_trace_weighted_tflops_and_separate_speedups(self) -> None:
        baseline = summarize_timing(timing([30] * 100, 20.0), PROFILE)
        candidate = summarize_timing(timing([30] * 40 + [0] * 60, 8.0), PROFILE)
        self.assertAlmostEqual(baseline["estimated_dit_tflops_per_video"], 10000.0)
        self.assertAlmostEqual(candidate["estimated_dit_tflops_per_video"], 4600.0)
        self.assertAlmostEqual(candidate["estimated_achieved_tflops_per_second"], 1150.0)
        self.assertAlmostEqual(candidate["estimated_t5_tflops_per_video"], 2.0)
        self.assertAlmostEqual(candidate["estimated_vae_decode_tflops_per_video"], 3.0)
        comparison = compare_matched(baseline, candidate)
        self.assertAlmostEqual(comparison["inference_latency_speedup"], 2.5)
        self.assertAlmostEqual(comparison["dit_flops_speedup"], 10000.0 / 4600.0)
        self.assertNotEqual(comparison["inference_latency_speedup"], comparison["dit_flops_speedup"])


if __name__ == "__main__":
    unittest.main()
