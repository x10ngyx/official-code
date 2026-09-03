from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
EXP_ROOT = Path("/all/yiran07-disk3/huteng_data/exp")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class Vbench200ReportingTests(unittest.TestCase):
    def test_final_report_uses_aggregate_model_forward_cuda_field(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="seacache4wan22-vbench-report-test.", dir=EXP_ROOT
        ) as temporary:
            root = Path(temporary)
            write_json(
                root / "run_config.json",
                {
                    "threshold": 0.24,
                    "use_ret_steps": False,
                    "protocol": {},
                    "generation_runner": {
                        "candidate": "persistent",
                        "baseline": "legacy",
                        "lifecycle_caveat": "test caveat",
                    },
                },
            )
            condition = {
                "pipeline_generate_wall_seconds": {
                    "mean": 10.0,
                    "p50": 10.0,
                    "p90": 10.0,
                },
                "estimated_dit_tflops_per_video": {"mean": 100.0},
                "t5_cuda_seconds": {"mean": 1.0},
                "model_forward_cuda_seconds": {"mean": 8.0},
                "vae_decode_cuda_seconds": {"mean": 1.0},
                "estimated_t5_tflops_per_video": 3.0,
                "estimated_vae_decode_tflops_per_video": 4.0,
            }
            write_json(
                root / "performance" / "summary.json",
                {
                    "latency_definition": {},
                    "flops_definition": {},
                    "conditions": {
                        "baseline": condition,
                        "seacache": condition,
                    },
                    "comparison": {
                        "latency_speedup_ratio_of_sums": 1.0,
                        "dit_flops_speedup_ratio_of_sums": 1.0,
                    },
                },
            )
            write_json(
                root / "evaluation" / "video_metrics" / "summary.json",
                {
                    "protocol_id": "rgb_full_reference_v1",
                    "metrics": {
                        key: {"mean": value}
                        for key, value in {
                            "psnr_rgb_db": 30.0,
                            "ssim_rgb": 0.9,
                            "lpips_alex_v0_1_spatial": 0.1,
                        }.items()
                    },
                },
            )
            vbench = {
                "aggregate_scores": {
                    "quality_score": 0.8,
                    "semantic_score": 0.7,
                    "total_score": 0.78,
                }
            }
            write_json(
                root
                / "evaluation"
                / "vbench_reference"
                / "vbench200_aggregate_scores.json",
                vbench,
            )
            write_json(
                root
                / "evaluation"
                / "vbench_candidate"
                / "vbench200_aggregate_scores.json",
                vbench,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(
                        PROJECT
                        / "experiments"
                        / "vbench200_t2v"
                        / "build_final_report.py"
                    ),
                    "--result-dir",
                    str(root),
                ],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            report = json.loads(
                (root / "benchmark_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                report["performance"]["baseline"]["dit_cuda_seconds_mean"],
                8.0,
            )
            self.assertEqual(
                report["generation_runner"]["lifecycle_caveat"],
                "test caveat",
            )
            self.assertIn(
                "Runner lifecycle caveat: test caveat",
                (root / "benchmark_report.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
