from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import generate


class StaticIntegrationTests(unittest.TestCase):
    def test_launcher_uses_only_the_project_wan22_environment(self) -> None:
        source = (PROJECT / "run_wan21.sh").read_text(encoding="utf-8")
        self.assertIn("WAN22_PYTHON", source)
        self.assertIn("-n wan2.2 python", source)
        self.assertNotIn("WAN21_PYTHON", source)
        self.assertNotIn("-n Wan2.1", source)

    def test_baseline_does_not_enable_magcache(self) -> None:
        wrapper, remaining = generate.parse_wrapper_args(["--task", "t2v-14B"])
        self.assertFalse(wrapper.enable_magcache)
        self.assertEqual(remaining, ["--task", "t2v-14B"])

    def test_magcache_options_require_explicit_enable(self) -> None:
        with mock.patch.object(sys, "stderr"):
            with self.assertRaises(SystemExit):
                generate.parse_wrapper_args(["--magcache_thresh", "0.2"])

    def test_only_magcache_method_is_present(self) -> None:
        active_files = (
            "generate.py",
            "inference_timing.py",
            "magcache.py",
            "wan21_integration.py",
        )
        python_sources = "\n".join(
            (PROJECT / name).read_text(encoding="utf-8")
            for name in active_files
        ).lower()
        for forbidden in ("block_cache", "cfg_cache", "zeustimestep", "teacache"):
            self.assertNotIn(forbidden, python_sources)


if __name__ == "__main__":
    unittest.main()
