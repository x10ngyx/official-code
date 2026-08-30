from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aggregate_custom_vbench_scores import main  # noqa: E402
from build_subset_full_info import build_subset  # noqa: E402
from evaluate_custom_vbench import CUSTOM_DIMENSIONS, load_prompt_map  # noqa: E402


class CustomVBenchTests(unittest.TestCase):
    def test_standard_mode_partial_subset_keeps_metadata_and_all_dimensions(self) -> None:
        prompts = [
            json.loads(line)
            for line in (ROOT.parent / "Vbench200" / "prompts.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        selected_ids = {
            "vbench200_001",
            "vbench200_026",
            "vbench200_051",
            "vbench200_056",
            "vbench200_076",
            "vbench200_085",
            "vbench200_101",
            "vbench200_126",
            "vbench200_151",
            "vbench200_176",
            "vbench200_177",
        }
        selected = [row for row in prompts if row["sample_id"] in selected_ids]
        full_info = json.loads(
            (ROOT.parent / "Vbench200" / "VBench200_full_info.json").read_text(
                encoding="utf-8"
            )
        )
        dimensions = json.loads(
            (ROOT / "dimensions.json").read_text(encoding="utf-8")
        )["dimensions"]
        subset, metadata = build_subset(
            full_info,
            {"items": [{"prompt_en": row["prompt_en"]} for row in selected]},
            dimensions,
        )
        self.assertEqual(metadata["selected_prompt_count"], 11)
        self.assertEqual(set(metadata["covered_dimensions"]), set(dimensions))
        self.assertEqual({row["prompt_en"] for row in subset}, {
            row["prompt_en"] for row in selected
        })
        for row in subset:
            if "auxiliary_info" in row:
                self.assertIsInstance(row["auxiliary_info"], dict)

    def test_prompt_map_must_exactly_match_video_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            (videos / "one.mp4").touch()
            prompt_map = root / "prompts.json"
            prompt_map.write_text(
                json.dumps({"one.mp4": "A test prompt"}), encoding="utf-8"
            )
            self.assertEqual(
                load_prompt_map(prompt_map, videos),
                {"one.mp4": "A test prompt"},
            )
            prompt_map.write_text(
                json.dumps({"missing.mp4": "A test prompt"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "exactly match"):
                load_prompt_map(prompt_map, videos)

    def test_raw_mean_aggregate_is_explicitly_not_official_full_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scores = root / "scores"
            scores.mkdir()
            expected = {}
            for index, dimension in enumerate(CUSTOM_DIMENSIONS, start=1):
                value = index / 100.0
                expected[dimension] = value
                (scores / f"custom_{dimension}_eval_results.json").write_text(
                    json.dumps({dimension: [value, []]}), encoding="utf-8"
                )
            output = root / "aggregate.json"
            with patch.object(
                sys,
                "argv",
                ["aggregate", "--score-dir", str(scores), "--output", str(output)],
            ):
                main()
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["protocol"], "vbench_custom_input_raw_mean_v1")
            self.assertFalse(payload["official_full_vbench_score"])
            self.assertEqual(payload["raw_dimension_scores"], expected)
            self.assertAlmostEqual(payload["vbench_score"], sum(expected.values()) / 10)


if __name__ == "__main__":
    unittest.main()
