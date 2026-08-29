from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_DIR / "experiments" / "vbench200_t2v"


def load_module(name: str, filename: str):
    path = EXPERIMENT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Vbench200PerformanceTests(unittest.TestCase):
    def test_four_gpu_launcher_locks_wan21_protocol_and_threshold(self):
        module = load_module("vbench200_launcher", "run_vbench200_4gpu.py")
        args = Namespace(
            generation_python=Path("/env/bin/python"),
            wan21_root=Path("/source/Wan2.1"),
            checkpoint_dir=Path("/models/Wan2.1-T2V-1.3B"),
            teacache_thresh=0.08,
            use_ret_steps=False,
            resume=False,
        )
        command = module.generation_command(
            args,
            implementation="teacache",
            output_dir=Path("/results/teacache"),
            shard_index=3,
        )
        self.assertEqual(
            command[command.index("--implementation") + 1], "teacache"
        )
        self.assertEqual(command[command.index("--task") + 1], "t2v-1.3B")
        self.assertEqual(command[command.index("--size") + 1], "832*480")
        self.assertEqual(command[command.index("--frame-num") + 1], "81")
        self.assertEqual(command[command.index("--sample-steps") + 1], "50")
        self.assertEqual(command[command.index("--sample-solver") + 1], "unipc")
        self.assertEqual(command[command.index("--sample-shift") + 1], "5")
        self.assertEqual(command[command.index("--guide-scale") + 1], "5")
        self.assertEqual(command[command.index("--seeds") + 1], "42")
        self.assertEqual(command[command.index("--num-shards") + 1], "4")
        self.assertEqual(command[command.index("--shard-index") + 1], "3")
        self.assertEqual(command[command.index("--teacache-thresh") + 1], "0.08")
        self.assertNotIn("--no-offload-model", command)
        self.assertNotIn("--t5-on-gpu", command)

    def test_trace_aggregation_uses_inference_only_latency_and_actual_cache_path(self):
        module = load_module("vbench200_performance", "aggregate_performance.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_paths = {}
            candidate_paths = {}
            for label, implementation, executed, latency, cuda_seconds, destination in (
                ("baseline", "wan21", 2, 20.0, 4.0, baseline_paths),
                ("teacache", "teacache", None, 10.0, 2.0, candidate_paths),
            ):
                path = root / f"{label}.json"
                calls = []
                for index in range(100):
                    blocks = executed if executed is not None else (2 if index % 2 == 0 else 0)
                    calls.append({"blocks_executed": blocks})
                path.write_text(
                    json.dumps(
                        {
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
            )
            teacache, _ = module.summarize_condition(
                label="teacache",
                paths=candidate_paths,
                expected_implementation="teacache",
                expected_count=1,
                full_forward_flops=1_000.0,
                always_on_flops=100.0,
                block_count=2,
            )
            self.assertEqual(
                baseline["end_to_end_inference_latency_seconds"]["mean"], 20.0
            )
            self.assertEqual(
                teacache["end_to_end_inference_latency_seconds"]["mean"], 10.0
            )
            self.assertEqual(baseline["estimated_dit_total_tflops"], 100_000 / 1e12)
            self.assertEqual(teacache["estimated_dit_total_tflops"], 55_000 / 1e12)
            self.assertEqual(teacache["total_full_compute_forward_calls"], 50)
            self.assertEqual(teacache["total_reuse_forward_calls"], 50)


if __name__ == "__main__":
    unittest.main()
