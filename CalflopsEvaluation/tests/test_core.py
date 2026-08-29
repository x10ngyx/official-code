from __future__ import annotations

import os
import unittest

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

from calflops_eval.aggregation import aggregate_trace
from calflops_eval.manual_ops import dense_attention_counts, elementwise_flops


class ManualOpsTest(unittest.TestCase):
    def test_dense_attention_formula(self) -> None:
        actual = dense_attention_counts(
            batch_size=1,
            query_tokens=4,
            key_value_tokens=5,
            num_heads=2,
            head_dim=3,
        )
        score_elements = 1 * 4 * 5 * 2
        self.assertEqual(actual["macs"], 2 * score_elements * 3)
        self.assertEqual(actual["flops"], 4 * score_elements * 3 + 5 * score_elements)

    def test_elementwise_flops(self) -> None:
        self.assertEqual(elementwise_flops(num_elements=12, operations_per_element=3), 36)


class TraceAggregationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cost_table = {
            "components": {
                "high_full": {"flops": 100},
                "low_full": {"flops": 80},
                "controller": {"flops": 2},
                "reuse_path": {"flops": 1},
            }
        }
        self.mapping = {
            "sample_field": "sample_id",
            "stage_field": "model_stage",
            "action_field": "decision",
            "baseline_action": "baseline",
            "action_components": {
                "baseline": {
                    "high": ["high_full", "high_full"],
                    "low": ["low_full", "low_full"],
                },
                "recompute": {
                    "high": ["controller", "high_full", "high_full"],
                    "low": ["controller", "low_full", "low_full"],
                },
                "reuse": {"$default": ["controller", "reuse_path"]},
            },
        }

    def test_ratio_of_sums(self) -> None:
        rows = [
            {"sample_id": "a", "model_stage": "high", "decision": "recompute"},
            {"sample_id": "a", "model_stage": "high", "decision": "reuse"},
            {"sample_id": "a", "model_stage": "low", "decision": "recompute"},
            {"sample_id": "a", "model_stage": "low", "decision": "reuse"},
        ]
        actual = aggregate_trace(
            cost_table=self.cost_table, mapping=self.mapping, rows=rows
        )
        self.assertEqual(actual["baseline_total_flops"], 720)
        self.assertEqual(actual["candidate_total_flops"], 370)
        self.assertAlmostEqual(actual["flops_speedup_ratio_of_sums"], 720 / 370)

    def test_unknown_action_fails_closed(self) -> None:
        rows = [{"sample_id": "a", "model_stage": "high", "decision": "unknown"}]
        with self.assertRaisesRegex(KeyError, "No component mapping"):
            aggregate_trace(cost_table=self.cost_table, mapping=self.mapping, rows=rows)


if __name__ == "__main__":
    unittest.main()
