#!/usr/bin/env python3
"""Build the canonical technical-report artifact for the low-speed scan."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_RESULT_ROOT = Path(
    "/mnt/hdd/xiongyuxiang/tmp/exp/teacache_wan21_vbench10_low_speed_no_offload_scan"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, float | int | str]]:
    integer_fields = {
        "prompt_count",
        "full_compute_forward_calls",
        "reuse_forward_calls",
    }
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            row: dict[str, float | int | str] = {}
            for key, value in source.items():
                if key in integer_fields:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            row["threshold_label"] = f"{float(row['threshold']):.3f}"
            rows.append(row)
    return rows


def make_target_rows(
    summary: dict[str, Any], rows: list[dict[str, float | int | str]]
) -> list[dict[str, Any]]:
    target = next(item for item in summary["targets"] if item["target_speedup"] == 1.8)
    interpolation = target["linear_interpolation"]
    by_threshold = {float(row["threshold"]): row for row in rows}
    lower = by_threshold[float(interpolation["left_threshold"])]
    upper = by_threshold[float(interpolation["right_threshold"])]
    return [
        {
            "status": "observed lower",
            "threshold": float(lower["threshold"]),
            "full_inference_speedup": float(
                lower["inference_speedup_ratio_of_sums"]
            ),
            "absolute_error": abs(
                float(lower["inference_speedup_ratio_of_sums"]) - 1.8
            ),
            "psnr_rgb_db": float(lower["psnr_rgb_db"]),
            "interpretation": "nearest measured point by absolute error; below 1.8x",
        },
        {
            "status": "linear interpolation",
            "threshold": float(interpolation["threshold"]),
            "full_inference_speedup": 1.8,
            "absolute_error": 0.0,
            "psnr_rgb_db": None,
            "interpretation": "descriptive only; threshold was not directly measured",
        },
        {
            "status": "observed upper",
            "threshold": float(upper["threshold"]),
            "full_inference_speedup": float(
                upper["inference_speedup_ratio_of_sums"]
            ),
            "absolute_error": abs(
                float(upper["inference_speedup_ratio_of_sums"]) - 1.8
            ),
            "psnr_rgb_db": float(upper["psnr_rgb_db"]),
            "interpretation": "measured point to use when at least 1.8x is required",
        },
    ]


def write_sqlite(
    path: Path,
    rows: list[dict[str, float | int | str]],
    target_rows: list[dict[str, Any]],
) -> None:
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE threshold_summary (
                threshold REAL PRIMARY KEY,
                threshold_label TEXT NOT NULL,
                prompt_count INTEGER NOT NULL,
                full_inference_speedup REAL NOT NULL,
                normalized_speedup REAL NOT NULL,
                pipeline_seconds_mean REAL NOT NULL,
                text_encoding_seconds_mean REAL NOT NULL,
                denoising_core_seconds_mean REAL NOT NULL,
                vae_decode_seconds_mean REAL NOT NULL,
                pipeline_other_seconds_mean REAL NOT NULL,
                dit_cuda_speedup REAL NOT NULL,
                reuse_fraction REAL NOT NULL,
                dit_tflops_per_video REAL NOT NULL,
                dit_tflops_per_second REAL NOT NULL,
                psnr_rgb_db REAL NOT NULL,
                ssim_rgb REAL NOT NULL,
                lpips REAL NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO threshold_summary VALUES (
                :threshold, :threshold_label, :prompt_count,
                :inference_speedup_ratio_of_sums,
                :fixed_module_normalized_speedup_ratio_of_sums,
                :pipeline_seconds_mean, :text_encoding_seconds_mean,
                :denoising_core_seconds_mean, :vae_decode_seconds_mean,
                :pipeline_other_seconds_mean, :dit_cuda_speedup_ratio_of_sums,
                :reuse_fraction, :estimated_dit_tflops_per_video_mean,
                :estimated_achieved_dit_tflops_per_second, :psnr_rgb_db,
                :ssim_rgb, :lpips_alex_v0_1_spatial
            )
            """,
            rows,
        )
        connection.execute(
            """
            CREATE TABLE target_bracket (
                status TEXT NOT NULL,
                threshold REAL NOT NULL,
                full_inference_speedup REAL NOT NULL,
                absolute_error REAL NOT NULL,
                psnr_rgb_db REAL,
                interpretation TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO target_bracket VALUES "
            "(:status, :threshold, :full_inference_speedup, :absolute_error, "
            ":psnr_rgb_db, :interpretation)",
            target_rows,
        )
        connection.commit()
    finally:
        connection.close()


def source_payload(
    *, source_id: str, label: str, sql: str, table: str, generated_at: str
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": "analysis/report_source.sql",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": sql,
            "description": label,
            "executed_at": generated_at,
            "tables_used": [table],
            "filters": [
                "Wan2.1-T2V-1.3B",
                "fixed 10-prompt VBench200 subset",
                "832x480, 81 frames, 50 UniPC steps, seed 42, BF16",
                "GPU T5 and offload_model=False",
                "thresholds 0.060-0.100",
            ],
            "metric_definitions": [
                "Full-inference speedup is sum(baseline pipeline_generate_wall_seconds) divided by sum(candidate pipeline_generate_wall_seconds) over the same 10 prompts.",
                "Pipeline inference includes text encoding, denoising, VAE decode, and pipeline other; model loading, MP4 export, and metric evaluation are excluded.",
                "DiT TFLOPs are trace-weighted DiT-forward estimates from Calflops plus the dense FlashAttention correction; T5 and VAE FLOPs are not included.",
                "PSNR, SSIM, and LPIPS compare each candidate with the matched baseline video using the same prompt and seed over 81 aligned RGB frames.",
            ],
        },
    }


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    analysis_dir = result_root / "analysis"
    summary_csv = analysis_dir / "summary.csv"
    summary_json = analysis_dir / "summary.json"
    if not summary_csv.is_file() or not summary_json.is_file():
        raise FileNotFoundError("validated scan summary is missing")
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("scan analysis is not complete")
    validation = summary.get("validation", {})
    if validation.get("status") != "passed":
        raise ValueError("scan validation did not pass")
    rows = read_rows(summary_csv)
    if len(rows) != 7:
        raise ValueError(f"expected seven threshold rows, found {len(rows)}")
    target_rows = make_target_rows(summary, rows)
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()

    write_sqlite(analysis_dir / "report_source.sqlite", rows, target_rows)
    (analysis_dir / "report_source.sql").write_text(
        "-- Exact rows used by the technical report.\n"
        "SELECT * FROM threshold_summary ORDER BY threshold;\n"
        "SELECT * FROM target_bracket ORDER BY threshold;\n",
        encoding="utf-8",
    )

    condition_source = source_payload(
        source_id="condition_query",
        label="Validated low-threshold condition summary",
        sql="SELECT * FROM threshold_summary ORDER BY threshold",
        table="threshold_summary",
        generated_at=generated_at,
    )
    target_source = source_payload(
        source_id="target_query",
        label="Measured and interpolated 1.8x bracket",
        sql="SELECT * FROM target_bracket ORDER BY threshold",
        table="target_bracket",
        generated_at=generated_at,
    )
    manifest_sources = [
        {key: value for key, value in source.items() if key != "query"}
        for source in (condition_source, target_source)
    ]

    chart = {
        "id": "speedup_chart",
        "title": "Full-inference speedup across the low-threshold grid",
        "subtitle": "10 matched VBench200 prompts; measured ratio of summed inference time",
        "type": "bar",
        "intent": "comparison",
        "question": "Which measured threshold is closest to the 1.8x time target?",
        "rationale": "Seven deliberately selected discrete thresholds are better shown as exact category bars than as an underpowered continuous trend or scatter.",
        "dataset": "threshold_summary",
        "sourceId": "condition_query",
        "valueFormat": "number",
        "unit": "×",
        "layout": "wide",
        "encodings": {
            "x": {
                "field": "threshold_label",
                "type": "nominal",
                "label": "Threshold",
            },
            "y": {
                "field": "inference_speedup_ratio_of_sums",
                "type": "quantitative",
                "label": "Full-inference speedup",
                "unit": "×",
            },
            "tooltip": [
                {
                    "field": "inference_speedup_ratio_of_sums",
                    "type": "quantitative",
                    "label": "Full-inference speedup",
                    "unit": "×",
                },
                {
                    "field": "dit_cuda_speedup_ratio_of_sums",
                    "type": "quantitative",
                    "label": "DiT CUDA speedup",
                    "unit": "×",
                },
                {
                    "field": "psnr_rgb_db",
                    "type": "quantitative",
                    "label": "PSNR",
                    "unit": "dB",
                },
                {
                    "field": "reuse_fraction",
                    "type": "quantitative",
                    "label": "Reuse fraction",
                },
            ],
        },
    }

    target_table = {
        "id": "target_table",
        "title": "1.8x target bracket",
        "subtitle": "Observed endpoints and descriptive interpolation; the middle row is not a measured run",
        "dataset": "target_bracket",
        "sourceId": "target_query",
        "density": "spacious",
        "layout": "wide",
        "defaultSort": {"field": "threshold", "direction": "asc"},
        "columns": [
            {"field": "status", "label": "Status", "type": "text"},
            {"field": "threshold", "label": "Threshold", "format": "number"},
            {
                "field": "full_inference_speedup",
                "label": "Full-inference speedup",
                "format": "number",
                "unit": "×",
            },
            {
                "field": "absolute_error",
                "label": "Absolute error vs 1.8x",
                "format": "number",
                "unit": "×",
            },
            {
                "field": "psnr_rgb_db",
                "label": "PSNR",
                "format": "number",
                "unit": "dB",
            },
            {
                "field": "interpretation",
                "label": "Interpretation",
                "type": "text",
            },
        ],
    }
    condition_table = {
        "id": "condition_table",
        "title": "Complete measured scan",
        "subtitle": "All values aggregate the same 10 prompts; TFLOPs and TFLOP/s are DiT-only",
        "dataset": "threshold_summary",
        "sourceId": "condition_query",
        "density": "compact",
        "layout": "wide",
        "defaultSort": {"field": "threshold", "direction": "asc"},
        "columns": [
            {"field": "threshold", "label": "Threshold", "format": "number"},
            {
                "field": "pipeline_seconds_mean",
                "label": "Inference",
                "format": "number",
                "unit": "s/video",
            },
            {
                "field": "inference_speedup_ratio_of_sums",
                "label": "Full speedup",
                "format": "number",
                "unit": "×",
            },
            {
                "field": "dit_cuda_speedup_ratio_of_sums",
                "label": "DiT speedup",
                "format": "number",
                "unit": "×",
            },
            {
                "field": "reuse_fraction",
                "label": "Reuse fraction",
                "format": "percent",
            },
            {
                "field": "estimated_dit_tflops_per_video_mean",
                "label": "DiT compute",
                "format": "number",
                "unit": "TFLOPs/video",
            },
            {
                "field": "estimated_achieved_dit_tflops_per_second",
                "label": "DiT throughput",
                "format": "number",
                "unit": "TFLOP/s",
            },
            {
                "field": "psnr_rgb_db",
                "label": "PSNR",
                "format": "number",
                "unit": "dB",
            },
            {"field": "ssim_rgb", "label": "SSIM", "format": "number"},
            {
                "field": "lpips_alex_v0_1_spatial",
                "label": "LPIPS",
                "format": "number",
            },
        ],
    }

    title = "TeaCache Wan2.1：1.8× 低阈值实测报告"
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "target_query",
            "body": (
                "## 技术摘要：1.8× 位于 threshold 0.075–0.080\n\n"
                "- **描述性插值为 0.07744。** 0.075 实测 1.7069×，0.080 实测 1.8980×；插值点本身没有直接生成视频。\n"
                "- **若只能选已测点，0.075 的绝对误差略小；若要求至少达到 1.8×，应选 0.080。** 两点 PSNR 分别为 21.235 dB 和 21.094 dB。\n"
                "- **不建议使用 0.085 或 0.090。** 它们与 0.080 同为约 1.90×/50% reuse 平台，但 PSNR 降至 20.829 dB，没有获得可辨认的速度收益。"
            ),
        },
        {
            "id": "speedup_finding",
            "type": "markdown",
            "sourceId": "condition_query",
            "body": (
                "## 0.075→0.080 是 1.8× 附近唯一明显的速度台阶\n\n"
                "threshold 从 0.075 增至 0.080 时，完整推理从 150.55 s/video 降到 135.40 s/video，speedup 增加 0.1911×；reuse 从 44% 增至 50%。之后 0.080–0.090 基本保持同一速度平台。"
            ),
        },
        {"id": "speedup", "type": "chart", "chartId": "speedup_chart"},
        {
            "id": "target_intro",
            "type": "markdown",
            "sourceId": "target_query",
            "body": (
                "## 实测选择与插值建议必须分开使用\n\n"
                "0.07744 只是在两个聚合时间点之间的线性描述。TeaCache cache 决策会随 prompt 和 timestep 离散变化，所以不能把该数值称为已验证 threshold。"
            ),
        },
        {"id": "targets", "type": "table", "tableId": "target_table"},
        {
            "id": "tradeoff_finding",
            "type": "markdown",
            "sourceId": "condition_query",
            "body": (
                "## 0.080 提供当前最合理的“至少 1.8×”质量折中\n\n"
                "0.080 的 DiT CUDA speedup 为 1.9989×，DiT compute 从 baseline 的 28,499.25 降到 14,250.72 TFLOPs/video；完整 pipeline 因 VAE 等固定开销只达到 1.8980×。从 0.075 到 0.080，PSNR 仅下降 0.141 dB；提高到 0.100 虽达到 1.9730×，PSNR 则进一步降到 20.425 dB。"
            ),
        },
        {"id": "conditions", "type": "table", "tableId": "condition_table"},
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## 数据范围与指标定义\n\n"
                "实验使用 Wan2.1-T2V-1.3B、固定 10 条 VBench200 prompt、832×480、81 帧、50-step UniPC、CFG 5、shift 5、seed 42、BF16；T5/DiT/VAE 均在 GPU，`offload_model=False`。完整推理时间包含 text encoding、denoising、VAE decode 和 pipeline other，排除模型加载、MP4 导出与质量评测。PSNR/SSIM/LPIPS 使用相同 prompt/seed 的 baseline 视频逐帧配对。"
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "body": (
                "## 实验设计与计算口径\n\n"
                "每个 threshold 在相同 10 条 prompt 上各运行一次，headline speedup 用 10 条 baseline 推理时间总和除以候选时间总和。DiT TFLOPs 由 Calflops 加 FlashAttention 解析补偿得到，并按每条实际 cache block trace 加权；它不包含 T5/VAE FLOPs。TFLOP/s 在各点约 115.8–116.0，说明时间下降主要来自计算量减少，而不是 GPU 吞吐漂移。"
            ),
        },
        {
            "id": "robustness",
            "type": "markdown",
            "sourceId": "condition_query",
            "body": (
                "## 完整性通过，但插值仍需直接验证\n\n"
                "验证覆盖 80 个视频/timing trace、70 个质量 pair 和 5,670 对帧；每条视频均核对 100 次 DiT 调用、full/reuse reconciliation 与模块时间闭合。限制包括固定 10-prompt 子集、每点单 seed/单次 timed run，以及 0.080→0.085 出现 0.0019× 的轻微非单调计时噪声。"
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 推荐使用方式\n\n"
                "1. 若目标是**至少 1.8×**且必须使用现有实测点，采用 **threshold=0.080**。\n"
                "2. 若目标是**最小化与 1.8× 的绝对误差**且允许低于目标，现有最近点是 **0.075**。\n"
                "3. 若要得到更精确的单一 threshold，下一次只需补测 **0.0775**；在直接实测前，将 0.07744 标为插值而不是最终测量值。\n"
                "4. 避免 0.085/0.090：相对 0.080 没有速度收益，质量更差。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 尚待确认的问题\n\n"
                "0.0775 在 10 条 prompt 上会落入 44%、50% 还是混合 reuse schedule；扩展到完整 VBench200 后，prompt 分布变化是否会让聚合 1.8× threshold 偏移；以及不同 GPU/并发条件下约 0.08 的固定模块时间是否保持稳定。"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Measured GPU-T5/no-offload VBench10 calibration of TeaCache thresholds near 1.8x full-inference speedup.",
            "generatedAt": generated_at,
            "charts": [chart],
            "tables": [target_table, condition_table],
            "sources": manifest_sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "threshold_summary": rows,
                "target_bracket": target_rows,
            },
        },
        "sources": [condition_source, target_source],
    }
    (analysis_dir / "artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "artifact": str(analysis_dir / "artifact.json")}))


if __name__ == "__main__":
    main()
