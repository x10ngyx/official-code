from __future__ import annotations

import unittest

from analyze_calibration import linear_fit, select_fit_rows


class CalibrationAnalysisTests(unittest.TestCase):
    def test_linear_fit_recovers_exact_line(self) -> None:
        fit = linear_fit([(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)])
        self.assertAlmostEqual(fit["slope"], 2.0)
        self.assertAlmostEqual(fit["intercept"], 1.0)
        self.assertAlmostEqual(fit["r_squared"], 1.0)
        self.assertAlmostEqual(fit["rmse"], 0.0)

    def test_fit_selection_includes_domain_brackets(self) -> None:
        rows = [
            {"threshold": 0.04, "speedup": 1.2},
            {"threshold": 0.08, "speedup": 1.7},
            {"threshold": 0.12, "speedup": 2.5},
            {"threshold": 0.20, "speedup": 3.8},
            {"threshold": 0.24, "speedup": 4.2},
        ]
        selected = select_fit_rows(rows, (1.5, 3.5))
        self.assertEqual([row["threshold"] for row in selected], [0.04, 0.08, 0.12, 0.20])

    def test_fit_selection_rejects_unbracketed_domain(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not bracket"):
            select_fit_rows([
                {"threshold": 0.04, "speedup": 1.6},
                {"threshold": 0.08, "speedup": 2.0},
                {"threshold": 0.12, "speedup": 3.2},
            ], (1.5, 3.5))


if __name__ == "__main__":
    unittest.main()
