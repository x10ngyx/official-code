from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_dit_forward(*, candidate: bool, block_count: int = 2) -> dict:
    calls = []
    for call_index in range(100):
        step = call_index // 2
        stage = "high" if step < 32 else "low"
        recompute = not candidate or step in {0, 32, 49}
        executed = block_count if recompute else 0
        calls.append(
            {
                "call_index": call_index,
                "step_index": step,
                "model_stage": stage,
                "cfg_branch": "cond" if call_index % 2 == 0 else "uncond",
                "blocks_executed": executed,
                "host_span_seconds": 0.2,
                "cuda_seconds": 0.1,
                "full_compute": recompute,
                "reuse": not recompute,
            }
        )
    return {
        "cuda_device": "cuda:0",
        "cuda_device_name": "fixture",
        "transformer_block_count_by_stage": {
            "high": block_count,
            "low": block_count,
        },
        "model_forward_call_count": 100,
        "model_forward_cuda_seconds": 10.0,
        "model_forward_host_span_seconds": 20.0,
        "full_compute_forward_calls": sum(row["full_compute"] for row in calls),
        "reuse_forward_calls": sum(row["reuse"] for row in calls),
        "component_latency": {
            "t5": {"call_count": 2, "cuda_seconds": 1.0, "host_span_seconds": 1.1},
            "dit": {"call_count": 100, "cuda_seconds": 10.0, "host_span_seconds": 20.0},
            "vae_decode": {"call_count": 1, "cuda_seconds": 0.5, "host_span_seconds": 0.6},
        },
        "calls": calls,
    }


class PerformanceAggregationTests(unittest.TestCase):
    def test_wan21_style_timing_and_trace_weighted_flops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            root = Path(temporary_text)
            source = root / "Wan2.2"
            checkpoint = root / "checkpoint"
            source.mkdir()
            checkpoint.mkdir()
            protocol_path = (
                PROJECT_ROOT / "configs" / "wan22_t2v_a14b_50step_dpmpp.json"
            )
            prepared_manifest = {
                "status": "pass",
                "mode": "prepared",
                "wan22_commit": "fixture",
                "patch_sha256": "fixture-patch",
                "runtime_sha256": "fixture-runtime",
                "timing_runtime_sha256": "fixture-timing",
                "protocol_sha256": sha256(protocol_path),
            }
            (source / ".seacache4wan22_prepared.json").write_text(
                json.dumps(prepared_manifest) + "\n", encoding="utf-8"
            )

            timings = {}
            for method, candidate, inference_seconds in (
                ("baseline", False, 10.0),
                ("seacache", True, 5.0),
            ):
                dit = make_dit_forward(candidate=candidate)
                timings[method] = {
                    "schema_version": 2,
                    "status": "success",
                    "implementation": (
                        "wan22" if method == "baseline" else "seacache"
                    ),
                    "latency_scope": {},
                    "pipeline_init_wall_seconds": 2.0,
                    "pipeline_generate_wall_seconds": inference_seconds,
                    **dit,
                    "error": None,
                }
                (root / f"{method}.timing.json").write_text(
                    json.dumps(timings[method]) + "\n", encoding="utf-8"
                )

            decisions = []
            for step in range(50):
                recompute = step in {0, 32, 49}
                action = "recompute" if recompute else "reuse"
                decisions.append(
                    {
                        "step_index": step,
                        "stage": "high" if step < 32 else "low",
                        "action": action,
                        "branches": {"cond": action, "uncond": action},
                    }
                )
            trace_path = root / "seacache.trace.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "schema": "seacache4wan22_trace_v1",
                        "threshold": 0.1,
                        "use_ret_steps": False,
                        "total_steps": 50,
                        "decisions": decisions,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifests = {}
            for method, threshold in (("baseline", "0"), ("seacache", "0.1")):
                video = root / f"{method}.mp4"
                log = root / f"{method}.log"
                manifest = root / f"{method}.manifest.json"
                video.write_bytes(b"fixture-video")
                log.write_text("fixture-log\n", encoding="utf-8")
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "write_run_manifest.py"),
                    "--output",
                    str(manifest),
                    "--source",
                    str(source),
                    "--checkpoint",
                    str(checkpoint),
                    "--threshold",
                    threshold,
                    "--prompt",
                    "fixture prompt",
                    "--video",
                    str(video),
                    "--timing",
                    str(root / f"{method}.timing.json"),
                    "--log",
                    str(log),
                ]
                if method == "seacache":
                    command.extend(
                        [
                            "--trace",
                            str(trace_path),
                        ]
                    )
                subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
                manifests[method] = manifest

            profile_path = root / "calflops.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "schema": "seacache4wan22_calflops_profile_v2",
                        "scope": "fixture DiT forward-only",
                        "source": {
                            "checkpoint_dir": str(checkpoint.resolve()),
                            "prepared_manifest": prepared_manifest,
                        },
                        "input": {
                            "video_shape_fhw": [45, 480, 832],
                            "stage_steps": {"high": 32, "low": 18},
                            "transformer_blocks": 2,
                        },
                        "stages": {
                            "high": {
                                "model": {"transformer_blocks": 2},
                                "branches": {
                                    branch: {
                                        "estimated_full_flops": 100.0,
                                        "estimated_always_on_flops": 10.0,
                                    }
                                    for branch in ("cond", "uncond")
                                },
                            },
                            "low": {
                                "model": {"transformer_blocks": 2},
                                "branches": {
                                    branch: {
                                        "estimated_full_flops": 80.0,
                                        "estimated_always_on_flops": 8.0,
                                    }
                                    for branch in ("cond", "uncond")
                                },
                            },
                        },
                        "component_profiles": {
                            "t5": {"calls_per_video": 2, "estimated_flops_per_video": 3.0e12, "estimated_tflops_per_video": 3.0},
                            "vae_decode": {"calls_per_video": 1, "estimated_flops_per_video": 4.0e12, "estimated_tflops_per_video": 4.0},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "performance"
            subprocess.run(
                [
                    sys.executable,
                    str(
                        PROJECT_ROOT
                        / "experiments"
                        / "performance_t2v_a14b"
                        / "aggregate_performance.py"
                    ),
                    "--baseline-manifest",
                    str(manifests["baseline"]),
                    "--seacache-manifest",
                    str(manifests["seacache"]),
                    "--calflops-profile",
                    str(profile_path),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            result = json.loads((output_dir / "summary.json").read_text())
            per_video = [
                json.loads(line)
                for line in (output_dir / "per_video.jsonl").read_text().splitlines()
            ]

        self.assertEqual(result["comparison"]["latency_speedup_ratio_of_sums"], 2.0)
        self.assertEqual(per_video[0]["estimated_dit_flops"], 9280.0)
        self.assertEqual(per_video[1]["estimated_dit_flops"], 1396.0)
        self.assertAlmostEqual(
            result["comparison"]["dit_flops_speedup_ratio_of_sums"],
            9280.0 / 1396.0,
        )


if __name__ == "__main__":
    unittest.main()
