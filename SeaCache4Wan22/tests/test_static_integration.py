from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


class StaticIntegrationTests(unittest.TestCase):
    def test_patch_touches_only_three_integration_files(self) -> None:
        patch = (PROJECT / "patches" / "wan22_42bf4cf_seacache.patch").read_text(
            encoding="utf-8"
        )
        touched = re.findall(r"^diff --git a/(\S+) b/(\S+)$", patch, re.MULTILINE)
        self.assertEqual(
            touched,
            [
                ("generate.py", "generate.py"),
                ("wan/modules/model.py", "wan/modules/model.py"),
                ("wan/text2video.py", "wan/text2video.py"),
            ],
        )

    def test_no_other_cache_implementation_markers(self) -> None:
        active_files = [
            PROJECT / "runtime" / "seacache.py",
            PROJECT / "runtime" / "inference_timing.py",
            PROJECT / "patches" / "wan22_42bf4cf_seacache.patch",
        ]
        text = "\n".join(path.read_text(encoding="utf-8") for path in active_files).lower()
        for forbidden in ("block_cache", "cfg_cache", "zeustimestep", "teacache"):
            self.assertNotIn(forbidden, text)

    def test_cli_exposes_only_none_and_seacache(self) -> None:
        patch = (PROJECT / "patches" / "wan22_42bf4cf_seacache.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn('choices=["none", "seacache"]', patch)

    def test_patch_carries_reference_modulated_norm_helper(self) -> None:
        patch = (PROJECT / "patches" / "wan22_42bf4cf_seacache.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn("+    def _modulated_norm1(self, x, e):", patch)
        self.assertIn("+        x_m = self._modulated_norm1(x, e)", patch)
        self.assertIn(
            "+            feature = self.blocks[0]._modulated_norm1(x, e_first)",
            patch,
        )

    def test_vbench200_uses_one_persistent_pipeline_per_worker(self) -> None:
        source = (
            PROJECT / "experiments" / "vbench200_t2v" / "generate_vbench200.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(source.count("pipeline = wan.WanT2V("), 1)
        self.assertIn('"persistent_pipeline": True', source)
        self.assertIn("profiler = _PipelineProfiler(", source)
        self.assertIn("for ordinal, sample in jobs:", source)
        self.assertNotIn('args.wan22_root / "generate.py"', source)
        self.assertNotIn("subprocess.run(command", source)

    def test_vbench200_launcher_staggers_workers_in_two_waves(self) -> None:
        launcher = (
            PROJECT
            / "experiments"
            / "vbench200_t2v"
            / "launch_threshold_suite_024_038_055.sh"
        ).read_text(encoding="utf-8")
        orchestrator = (
            PROJECT / "experiments" / "vbench200_t2v" / "run_vbench200.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--worker-launch-wave-size 2", launcher)
        self.assertIn("WAN22_START_THRESHOLD", launcher)
        self.assertIn("-n wan2.2 python", launcher)
        self.assertNotIn("/home/huteng", launcher)
        self.assertIn(
            "WAN22_BASELINE_SOURCE is required when starting after threshold 0.24",
            launcher,
        )
        self.assertIn("pipeline_initialization_count", orchestrator)
        self.assertIn("wait_until_wave_ready(wave)", orchestrator)


if __name__ == "__main__":
    unittest.main()
