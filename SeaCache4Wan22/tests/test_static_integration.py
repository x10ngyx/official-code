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


if __name__ == "__main__":
    unittest.main()
