from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
EXP_ROOT = Path("/mnt/hdd/xiongyuxiang/tmp/exp")
RUNNER_PATH = PROJECT / "experiments" / "vbench200_t2v" / "run_vbench200.py"
SPEC = importlib.util.spec_from_file_location("teacache_vbench200_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def baseline_timing() -> dict:
    calls = []
    for index in range(100):
        stage = "high" if index // 2 < 32 else "low"
        calls.append({
            "call_index": index,
            "step_index": index // 2,
            "model_stage": stage,
            "cfg_branch": "cond" if index % 2 == 0 else "uncond",
            "blocks_executed": 2,
            "cuda_seconds": 0.1,
            "full_compute": True,
            "reuse": False,
        })
    return {
        "schema_version": 2,
        "status": "success",
        "implementation": "wan22",
        "pipeline_generate_wall_seconds": 1.0,
        "transformer_block_count_by_stage": {"high": 2, "low": 2},
        "full_compute_forward_calls": 100,
        "reuse_forward_calls": 0,
        "calls": calls,
    }


class Vbench200PipelineTests(unittest.TestCase):
    def make_baseline(self, root: Path) -> tuple[Path, Path, Path]:
        source = root / "baseline"
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        prepared = root / "sea_source" / ".seacache4wan22_prepared.json"
        write_json(prepared, {
            "schema": "seacache4wan22_prepared_v1",
            "status": "pass",
            "mode": "prepared",
            "wan22_commit": RUNNER.WAN22_COMMIT,
        })
        prompt_hash = RUNNER.sha256(
            RUNNER.REPOSITORY_DIR / "Vbench200" / "prompts.jsonl"
        )
        for shard in range(4):
            write_json(source / f"generation_config.shard_{shard:03d}.json", {
                "schema": "seacache4wan22_vbench200_generation_v1",
                "condition": "baseline",
                "threshold": None,
                "use_ret_steps": False,
                "selected_prompt_count": 1,
                "prompt_manifest_sha256": prompt_hash,
                "protocol": RUNNER.generation_protocol(),
                "thread_env": RUNNER.THREAD_ENV,
                "prepared_manifest": str(prepared),
                "prepared_manifest_sha256": RUNNER.sha256(prepared),
                "checkpoint_dir": str(checkpoint),
                "shard_index": shard,
                "num_shards": 4,
            })
        video = source / "videos" / "vbench200_001.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        timing = source / "timings" / "vbench200_001.json"
        write_json(timing, baseline_timing())
        current = root / ".teacache4wan22_prepared.json"
        write_json(current, {
            "schema": "teacache4wan22_prepared_tree_validation_v1",
            "status": "pass",
            "mode": "prepared",
            "wan22_commit": RUNNER.WAN22_COMMIT,
        })
        return source, current, checkpoint

    def test_cross_method_baseline_requires_all_full_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, current, checkpoint = self.make_baseline(Path(temporary))
            provenance = RUNNER.validate_reusable_baseline(
                source, current, checkpoint, 1
            )
            self.assertEqual(
                provenance["reuse_mode"],
                "direct_directory_symlink_no_regeneration",
            )
            timing_path = source / "timings" / "vbench200_001.json"
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            timing["calls"][1]["blocks_executed"] = 0
            timing["calls"][1]["full_compute"] = False
            timing["calls"][1]["reuse"] = True
            timing["full_compute_forward_calls"] = 99
            timing["reuse_forward_calls"] = 1
            write_json(timing_path, timing)
            with self.assertRaisesRegex(ValueError, "100-full-call"):
                RUNNER.validate_reusable_baseline(source, current, checkpoint, 1)

    def test_formal_aggregate_validates_trace_and_calflops_contract(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="teacache-vbench200-aggregate-test.", dir=EXP_ROOT
        ) as temporary:
            root = Path(temporary)
            coefficient_sha = "a" * 64
            baseline = baseline_timing()
            candidate = baseline_timing()
            candidate["implementation"] = "teacache"
            candidate["pipeline_generate_wall_seconds"] = 0.5
            candidate["pipeline_lifecycle"] = {
                "persistent_pipeline": True,
                "pipeline_init_accounted_in_this_sample": False,
            }
            decisions = []
            full_steps = {0, 32, 49}
            for index, call in enumerate(candidate["calls"]):
                recompute = index // 2 in full_steps
                call["blocks_executed"] = 2 if recompute else 0
                call["full_compute"] = recompute
                call["reuse"] = not recompute
                call["cuda_seconds"] = 0.1
            for step in range(50):
                recompute = step in full_steps
                action = "recompute" if recompute else "reuse"
                decisions.append({
                    "step_index": step,
                    "stage": "high" if step < 32 else "low",
                    "action": action,
                    "branches": {"cond": action, "uncond": action},
                })
            for payload, condition in ((baseline, "baseline"), (candidate, "teacache")):
                payload["model_forward_call_count"] = 100
                payload["model_forward_cuda_seconds"] = 10.0
                payload["full_compute_forward_calls"] = sum(
                    int(call["full_compute"]) for call in payload["calls"]
                )
                payload["reuse_forward_calls"] = sum(
                    int(call["reuse"]) for call in payload["calls"]
                )
                payload["component_latency"] = {
                    "t5": {"call_count": 2, "cuda_seconds": 1.0, "host_span_seconds": 1.1},
                    "dit": {"call_count": 100, "cuda_seconds": 10.0, "host_span_seconds": 11.0},
                    "vae_decode": {"call_count": 1, "cuda_seconds": 0.5, "host_span_seconds": 0.6},
                }
                write_json(
                    root / condition / "timings" / "vbench200_001.json", payload
                )
            write_json(root / "teacache" / "traces" / "vbench200_001.json", {
                "schema": "teacache4wan22_trace_v1",
                "threshold": 0.29,
                "use_ret_steps": False,
                "coefficients_sha256": coefficient_sha,
                "coefficient_protocol": RUNNER.COEFFICIENT_PROTOCOL
                if hasattr(RUNNER, "COEFFICIENT_PROTOCOL")
                else {
                    "task": "t2v-A14B", "size_wh": [832, 480], "frame_num": 45,
                    "sampling_steps": 50, "sample_solver": "dpm++", "shift": 12.0,
                    "guide_scale_low_high": [3.0, 4.0], "boundary": 0.875,
                    "param_dtype": "torch.bfloat16", "use_ret_steps": False,
                },
                "total_steps": 50,
                "recompute": 3,
                "reuse": 47,
                "decisions": decisions,
            })
            profile = root / "calflops.json"
            write_json(profile, {
                "schema": "teacache4wan22_calflops_profile_v2",
                "scope": "fixture",
                "input": {"stage_steps": {"high": 32, "low": 18}},
                "stages": {
                    stage: {"branches": {
                        branch: {
                            "estimated_full_flops": 100.0,
                            "estimated_always_on_flops": 10.0,
                        }
                        for branch in ("cond", "uncond")
                    }}
                    for stage in ("high", "low")
                },
                "component_profiles": {
                    "t5": {"calls_per_video": 2, "estimated_flops_per_video": 3e12, "estimated_tflops_per_video": 3.0},
                    "vae_decode": {"calls_per_video": 1, "estimated_flops_per_video": 4e12, "estimated_tflops_per_video": 4.0},
                },
            })
            output = root / "performance"
            subprocess.run([
                sys.executable,
                str(PROJECT / "experiments" / "vbench200_t2v" / "aggregate_performance.py"),
                "--baseline-dir", str(root / "baseline"),
                "--teacache-dir", str(root / "teacache"),
                "--calflops-profile", str(profile),
                "--output-dir", str(output),
                "--expected-videos", "1",
                "--expected-threshold", "0.29",
                "--expected-coefficients-sha256", coefficient_sha,
            ], check=True, stdout=subprocess.PIPE, text=True)
            result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                result["conditions"]["baseline"]["total_full_compute_forward_calls"],
                100,
            )
            self.assertEqual(
                result["conditions"]["teacache"]["total_reuse_forward_calls"],
                94,
            )


if __name__ == "__main__":
    unittest.main()
