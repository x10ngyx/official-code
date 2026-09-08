from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from taylorseer import TaylorSeerState, expected_full_steps  # noqa: E402


class TaylorSeerStateTests(unittest.TestCase):
    def test_locked_schedule_and_cfg_streams(self) -> None:
        state = TaylorSeerState(fresh_threshold=3, sample_steps=8)
        state.reset()
        observed = [state.begin_call() for _ in range(16)]
        full_steps = [step for step, stream, kind in observed if stream == "cond_stream" and kind == "full"]
        self.assertEqual(full_steps, [0, 3, 6])
        self.assertEqual(expected_full_steps(8, 3), full_steps)
        for index in range(0, len(observed), 2):
            self.assertEqual(observed[index][2], observed[index + 1][2])

    def test_first_order_derivative_and_prediction_are_branch_local(self) -> None:
        state = TaylorSeerState(fresh_threshold=2, sample_steps=4)
        state.reset()
        state.begin_call()
        state.update_feature(0, "ffn", torch.tensor([2.0]))
        state.begin_call()
        state.update_feature(0, "ffn", torch.tensor([20.0]))
        state.begin_call()
        state.begin_call()
        state.begin_call()
        state.update_feature(0, "ffn", torch.tensor([6.0]))
        state.begin_call()
        state.update_feature(0, "ffn", torch.tensor([28.0]))
        state.begin_call()
        self.assertEqual(state.current_stream, "cond_stream")
        self.assertEqual(state.predict_feature(0, "ffn").item(), 8.0)
        state.begin_call()
        self.assertEqual(state.current_stream, "uncond_stream")
        self.assertEqual(state.predict_feature(0, "ffn").item(), 32.0)

    def test_new_sample_requires_explicit_reset(self) -> None:
        state = TaylorSeerState(fresh_threshold=2, sample_steps=1)
        state.reset()
        state.begin_call()
        state.begin_call()
        with self.assertRaises(RuntimeError):
            state.begin_call()


if __name__ == "__main__":
    unittest.main()
