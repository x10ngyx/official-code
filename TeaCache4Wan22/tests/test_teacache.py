from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "runtime" / "teacache.py"
SPEC = importlib.util.spec_from_file_location("teacache_runtime", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
TeaCacheConfig = MODULE.TeaCacheConfig
TeaCacheController = MODULE.TeaCacheController


PROTOCOL = {
    "task": "t2v-A14B",
    "size_wh": [832, 480],
    "frame_num": 45,
    "sampling_steps": 50,
    "sample_solver": "dpm++",
    "shift": 12.0,
    "guide_scale_low_high": [3.0, 4.0],
    "boundary": 0.875,
    "param_dtype": "torch.bfloat16",
    "use_ret_steps": False,
}


class TeaCacheControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.coefficients = self.root / "coefficients.json"
        self.coefficients.write_text(
            json.dumps(
                {
                    "schema": "teacache4wan22_coefficients_v1",
                    "protocol": PROTOCOL,
                    "stages": {
                        "high": {"coefficients_descending": [0, 0, 0, 1, 0]},
                        "low": {"coefficients_descending": [0, 0, 0, 1, 0]},
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def controller(self, threshold: float = 0.75) -> object:
        return TeaCacheController(
            TeaCacheConfig(
                threshold=threshold,
                coefficients_path=str(self.coefficients),
                trace_path=str(self.root / "trace.json"),
            )
        )

    @staticmethod
    def execute(
        controller: object,
        stage: str,
        step: int,
        total: int,
        feature_value: float,
    ) -> bool:
        feature = torch.tensor([[feature_value, feature_value]], dtype=torch.float32)
        reuse = controller.plan_step(stage, step, total, feature)
        for branch, value in (("cond", 1.0), ("uncond", 2.0)):
            same = controller.plan_step(stage, step, total, feature.clone())
            assert same == reuse
            if reuse:
                observed = controller.reuse_residual(stage, branch, step)
                assert torch.equal(observed, torch.full((1, 2), value))
            else:
                controller.record_recompute(
                    stage, branch, step, torch.full((1, 2), value)
                )
        return reuse

    def test_shared_gate_stage_boundary_and_final_recompute(self) -> None:
        controller = self.controller()
        controller.validate_runtime_protocol(PROTOCOL)
        self.assertFalse(self.execute(controller, "high", 0, 5, 1.0))
        self.assertTrue(self.execute(controller, "high", 1, 5, 1.1))
        self.assertFalse(self.execute(controller, "high", 2, 5, 2.0))
        controller.clear_stage("high")
        self.assertFalse(self.execute(controller, "low", 3, 5, 1.0))
        self.assertFalse(self.execute(controller, "low", 4, 5, 1.1))
        controller.validate_complete(5)
        summary = controller.summary()
        self.assertEqual(summary["reuse"], 1)
        self.assertEqual(summary["recompute"], 4)
        self.assertEqual(summary["per_stage"]["high"]["reuse_path"], [1])
        trace = controller.write_trace(extra={"test": True})
        self.assertEqual(trace, self.root / "trace.json")
        payload = json.loads(trace.read_text(encoding="utf-8"))
        self.assertTrue(payload["test"])
        self.assertEqual(payload["decisions"][3]["forced_reason"], "stage_first")
        self.assertEqual(payload["decisions"][4]["forced_reason"], "global_final")

    def test_cfg_feature_mismatch_fails_closed(self) -> None:
        controller = self.controller()
        feature = torch.ones((1, 2))
        controller.plan_step("high", 0, 2, feature)
        with self.assertRaisesRegex(RuntimeError, "different TeaCache gate features"):
            controller.plan_step("high", 0, 2, feature * 2)

    def test_protocol_mismatch_is_rejected(self) -> None:
        controller = self.controller()
        mismatched = dict(PROTOCOL)
        mismatched["sampling_steps"] = 40
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            controller.validate_runtime_protocol(mismatched)

    def test_retention_and_nonpositive_threshold_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TeaCacheConfig(0.0, str(self.coefficients))
        with self.assertRaises(ValueError):
            TeaCacheConfig(0.1, str(self.coefficients), use_ret_steps=True)

    def test_unvalidated_raw_fit_is_rejected(self) -> None:
        self.coefficients.write_text(
            json.dumps({"primary": {}, "run_config": PROTOCOL}), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "validated"):
            self.controller()


if __name__ == "__main__":
    unittest.main()
