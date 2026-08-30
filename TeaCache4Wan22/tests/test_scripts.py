from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_MODULE_PATH = PROJECT_ROOT / "scripts" / "package_coefficients.py"
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    "teacache_package_coefficients", PACKAGE_MODULE_PATH
)
PACKAGE_MODULE = importlib.util.module_from_spec(PACKAGE_SPEC)
assert PACKAGE_SPEC.loader is not None
PACKAGE_SPEC.loader.exec_module(PACKAGE_MODULE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ScriptIntegrationTest(unittest.TestCase):
    def test_repository_url_normalization_only_ignores_git_suffix(self) -> None:
        normalize = PACKAGE_MODULE.canonical_git_repository
        self.assertEqual(
            normalize("https://github.com/KaiyueSun98/T2V-CompBench.git/"),
            "https://github.com/KaiyueSun98/T2V-CompBench",
        )
        self.assertEqual(
            normalize("https://github.com/KaiyueSun98/T2V-CompBench"),
            "https://github.com/KaiyueSun98/T2V-CompBench",
        )
        self.assertNotEqual(
            normalize("https://github.com/example/T2V-CompBench.git"),
            normalize("https://github.com/KaiyueSun98/T2V-CompBench"),
        )

    def test_manifests_and_ratio_of_sums_speedup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            root = Path(temporary_text)
            source = root / "Wan2.2"
            checkpoint = root / "checkpoint"
            source.mkdir()
            checkpoint.mkdir()
            protocol_path = (
                PROJECT_ROOT / "configs" / "wan22_t2v_a14b_50step_dpmpp.json"
            )
            (source / ".teacache4wan22_prepared.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "mode": "prepared",
                        "wan22_commit": "fixture",
                        "protocol_sha256": sha256(protocol_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            protocol = json.loads(
                protocol_path.read_text(encoding="utf-8")
            )
            runtime_keys = (
                "task",
                "size_wh",
                "frame_num",
                "sampling_steps",
                "sample_solver",
                "shift",
                "guide_scale_low_high",
                "boundary",
                "param_dtype",
                "use_ret_steps",
            )
            runtime_protocol = {key: protocol[key] for key in runtime_keys}

            def timing_payload(method: str, generate_seconds: float) -> dict:
                calls = []
                for call_index in range(100):
                    step_index = call_index // 2
                    calls.append(
                        {
                            "call_index": call_index,
                            "step_index": step_index,
                            "model_stage": "high" if step_index < 32 else "low",
                            "cfg_branch": "cond" if call_index % 2 == 0 else "uncond",
                            "blocks_executed": 2,
                            "host_span_seconds": 0.2,
                            "cuda_seconds": 0.1,
                            "full_compute": True,
                            "reuse": False,
                        }
                    )
                return {
                    "schema_version": 2,
                    "status": "success",
                    "implementation": "teacache" if method == "teacache" else "wan22",
                    "latency_scope": {},
                    "cuda_device": "cuda:0",
                    "cuda_device_name": "fixture",
                    "pipeline_init_wall_seconds": 2.0,
                    "pipeline_generate_wall_seconds": generate_seconds,
                    "model_forward_call_count": 100,
                    "model_forward_cuda_seconds": 10.0,
                    "model_forward_host_span_seconds": 20.0,
                    "transformer_block_count_by_stage": {"high": 2, "low": 2},
                    "full_compute_forward_calls": 100,
                    "reuse_forward_calls": 0,
                    "component_latency": {
                        "t5": {"call_count": 2, "cuda_seconds": 1.0, "host_span_seconds": 1.1},
                        "dit": {"call_count": 100, "cuda_seconds": 10.0, "host_span_seconds": 20.0},
                        "vae_decode": {"call_count": 1, "cuda_seconds": 0.5, "host_span_seconds": 0.6},
                    },
                    "calls": calls,
                    "error": None,
                }

            baseline_timing = timing_payload("none", 10.0)
            teacache_timing = timing_payload("teacache", 5.0)
            baseline_timing_path = root / "baseline.timing.json"
            teacache_timing_path = root / "teacache.timing.json"
            baseline_timing_path.write_text(
                json.dumps(baseline_timing) + "\n", encoding="utf-8"
            )
            teacache_timing_path.write_text(
                json.dumps(teacache_timing) + "\n", encoding="utf-8"
            )

            coefficient_path = root / "coefficients.json"
            coefficient_path.write_text(
                json.dumps(
                    {
                        "schema": "teacache4wan22_coefficients_v1",
                        "protocol": runtime_protocol,
                        "stages": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            decisions = []
            for step in range(50):
                reason = {0: "global_first", 32: "stage_first", 49: "global_final"}.get(
                    step
                )
                decisions.append(
                    {
                        "step_index": step,
                        "stage": "high" if step < 32 else "low",
                        "action": "recompute",
                        "forced_reason": reason,
                        "branches": {"cond": "recompute", "uncond": "recompute"},
                    }
                )
            trace_path = root / "teacache.trace.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "schema": "teacache4wan22_trace_v1",
                        "threshold": 0.1,
                        "coefficients_sha256": sha256(coefficient_path),
                        "coefficient_protocol": runtime_protocol,
                        "decisions": decisions,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            common = {
                "source": source,
                "checkpoint": checkpoint,
                "prompt": "fixture prompt",
            }
            manifests = {}
            for method, threshold, timing_path in (
                ("baseline", "0", baseline_timing_path),
                ("teacache", "0.1", teacache_timing_path),
            ):
                video = root / f"{method}.mp4"
                log = root / f"{method}.log"
                manifest = root / f"{method}.manifest.json"
                video.write_bytes(f"{method}-video".encode())
                log.write_text(f"{method}-log\n", encoding="utf-8")
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "write_run_manifest.py"),
                    "--output",
                    str(manifest),
                    "--source",
                    str(common["source"]),
                    "--checkpoint",
                    str(common["checkpoint"]),
                    "--threshold",
                    threshold,
                    "--prompt",
                    common["prompt"],
                    "--video",
                    str(video),
                    "--timing",
                    str(timing_path),
                    "--log",
                    str(log),
                ]
                if method == "teacache":
                    command.extend(
                        [
                            "--coefficients",
                            str(coefficient_path),
                            "--trace",
                            str(trace_path),
                        ]
                    )
                subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
                manifests[method] = manifest

            output = root / "speedup.json"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "compare_runs.py"),
                    "--baseline-manifest",
                    str(manifests["baseline"]),
                    "--teacache-manifest",
                    str(manifests["teacache"]),
                    "--output",
                    str(output),
                ],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["pair_count"], 1)
            self.assertEqual(
                result["baseline_pipeline_generate_wall_seconds_sum"], 10.0
            )
            self.assertEqual(
                result["teacache_pipeline_generate_wall_seconds_sum"], 5.0
            )
            self.assertEqual(result["inference_only_speedup_ratio_of_sums"], 2.0)


if __name__ == "__main__":
    unittest.main()
