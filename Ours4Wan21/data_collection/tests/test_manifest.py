from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


DATA_PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_PROJECT / "src"))

from ours4wan21_data.manifest import (  # noqa: E402
    MAPPING_SCHEMA,
    PROTOCOL,
    build_plan,
    materialize,
    validate_plan,
    validate_runnable,
    write_jsonl,
)


PROMPT_POOL = DATA_PROJECT / "resources/prompts/openvidhd_balanced_5000.upstream.jsonl"
PENDING = DATA_PROJECT / "configs/speed_threshold_mapping.pending.json"
PROMPT_POOL_SHA256 = "fb5d5d73f86b84d10d8e55154b789ac8549c74e90f33c1d4d2a02d67a5cde3e5"
SELECTED_ID_SHA256 = "4316be6b8be97af36221bb521e674814d6385a13e5d7936b4cc4769c772e4805"


class ManifestContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.summary = build_plan(PROMPT_POOL, 20260722)

    def test_plan_counts_ranges_and_empty_mapping(self) -> None:
        validate_plan(self.rows)
        self.assertEqual(len(self.rows), 9000)
        self.assertEqual(len({row["sample_id"] for row in self.rows}), 3000)
        self.assertEqual(Counter(Counter(row["sample_id"] for row in self.rows).values()), Counter({3: 3000}))
        self.assertEqual(Counter(row["shard_index"] for row in self.rows), Counter({0: 2250, 1: 2250, 2: 2250, 3: 2250}))
        self.assertTrue(all(1.5 <= row["target_speedup"] <= 3.5 for row in self.rows))
        self.assertTrue(all(0.2 <= row["q"] <= 1.0 for row in self.rows))
        self.assertTrue(all(row["mean_threshold"] is None and row["threshold_path"] is None for row in self.rows))
        self.assertFalse(self.summary["candidate_runnable"])

    def test_deterministic_plan(self) -> None:
        repeated, repeated_summary = build_plan(PROMPT_POOL, 20260722)
        self.assertEqual(repeated, self.rows)
        self.assertEqual(repeated_summary, self.summary)

    def test_prompt_selection_matches_established_openvid_sample(self) -> None:
        selected = [self.rows[index * 3]["sample_id"] for index in range(3000)]
        digest = hashlib.sha256(("\n".join(selected) + "\n").encode()).hexdigest()
        self.assertEqual(digest, SELECTED_ID_SHA256)
        self.assertEqual(self.summary["prompt_selection_seed"], 2026073001)
        self.assertEqual(self.summary["selected_part_count"], 98)

    def test_bundled_prompt_snapshot_is_exact(self) -> None:
        self.assertEqual(hashlib.sha256(PROMPT_POOL.read_bytes()).hexdigest(), PROMPT_POOL_SHA256)
        with PROMPT_POOL.open(encoding="utf-8") as handle:
            self.assertEqual(sum(1 for line in handle if line.strip()), 5000)

    def test_pending_mapping_blocks_and_calibrated_mapping_materializes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.jsonl"
            write_jsonl(plan, self.rows)
            with self.assertRaisesRegex(ValueError, "not calibrated"):
                materialize(plan, PENDING)
            calibration = root / "calibrated.json"
            fit_source = root / "calibration_observations.jsonl"
            fit_source.write_text('{"speedup": 1.5, "threshold": 0.03}\n', encoding="utf-8")
            import hashlib
            calibration.write_text(json.dumps({
                "schema": MAPPING_SCHEMA,
                "calibration_status": "calibrated",
                "model": "Wan2.1-T2V-1.3B",
                "protocol": {key: value for key, value in PROTOCOL.items() if key != "model"},
                "target_speedup_domain": [1.5, 3.5],
                "fit_source": str(fit_source),
                "fit_source_sha256": hashlib.sha256(fit_source.read_bytes()).hexdigest(),
                "threshold_bounds": [0.01, 0.20],
                "mapping": {
                    "kind": "monotone_piecewise_linear",
                    "speedups": [1.5, 2.5, 3.5],
                    "mean_thresholds": [0.03, 0.08, 0.16],
                },
            }), encoding="utf-8")
            runnable, summary = materialize(plan, calibration)
            validate_runnable(runnable)
            self.assertEqual(len(runnable), 9000)
            self.assertTrue(summary["candidate_runnable"])
            self.assertTrue(all(len(row["threshold_path"]) == 50 for row in runnable))
            self.assertTrue(all(0.01 <= min(row["threshold_path"]) <= max(row["threshold_path"]) <= 0.20 for row in runnable))


if __name__ == "__main__":
    unittest.main()
