#!/usr/bin/env python3
"""Build a protocol-aware TeaCache literature comparison report."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULT_ROOT = Path(
    "/mnt/hdd/xiongyuxiang/tmp/exp/teacache_wan21_vbench10_low_speed_no_offload_scan"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    return parser.parse_args()


def read_summary_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in source.items()})
    return rows


def local_source(source_id: str, label: str, sql: str, generated_at: str) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": "analysis/paper_comparison_source.sql",
        "query": {
            "engine": "embedded_snapshot",
            "language": "sql",
            "sql": sql,
            "description": label,
            "executed_at": generated_at,
            "filters": [
                "Wan2.1-T2V-1.3B",
                "50 denoising steps",
                "reference-based RGB PSNR",
            ],
        },
    }


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    analysis_dir = result_root / "analysis"
    summary_path = analysis_dir / "summary.json"
    rows_path = analysis_dir / "summary.csv"
    if not summary_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError("validated low-threshold scan summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("validation", {}).get("status") != "passed":
        raise ValueError("low-threshold scan is not validated and complete")
    rows = read_summary_rows(rows_path)
    by_threshold = {round(row["threshold"], 6): row for row in rows}
    local_008 = by_threshold[0.08]
    baseline = summary["baseline"]
    baseline_tflops = baseline["estimated_dit_tflops_per_video"]["mean"]
    baseline_pipeline = baseline["pipeline_generate_wall_seconds"]["mean"]
    baseline_dit_cuda = baseline["dit_cuda_seconds"]["mean"]
    fixed_share = (baseline_pipeline - baseline_dit_cuda) / baseline_pipeline
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()

    matched_compute = [
        {
            "source": "本次实测 · threshold 0.08",
            "group": "本地",
            "threshold": "0.08",
            "equivalent_nfe": 25,
            "compute_reduction": baseline_tflops / local_008["estimated_dit_tflops_per_video_mean"],
            "reported_speedup": local_008["inference_speedup_ratio_of_sums"],
            "speed_scope": "完整推理墙钟",
            "psnr_db": local_008["psnr_rgb_db"],
            "ssim": local_008["ssim_rgb"],
            "lpips": local_008["lpips_alex_v0_1_spatial"],
            "frames": "81",
            "prompts": "10",
            "comparison": "比 SeaCache 高 0.254 dB；位于公开区间内",
        },
        {
            "source": "SeaCache TeaCache rerun",
            "group": "论文",
            "threshold": "0.09",
            "equivalent_nfe": 25,
            "compute_reduction": 2.0,
            "reported_speedup": 176.3 / 86.6,
            "speed_scope": "论文 latency",
            "psnr_db": 20.84,
            "ssim": 0.721,
            "lpips": 0.171,
            "frames": "65",
            "prompts": "944",
            "comparison": "25 NFE 公开低端；TFLOPs 8214→4107",
        },
        {
            "source": "BAG TeaCache rerun",
            "group": "论文",
            "threshold": "0.09",
            "equivalent_nfe": 25,
            "compute_reduction": 2.0,
            "reported_speedup": 1.98,
            "speed_scope": "论文墙钟",
            "psnr_db": 23.79,
            "ssim": 0.846,
            "lpips": 0.111,
            "frames": "65",
            "prompts": "100",
            "comparison": "同 NFE；比本地高 2.696 dB",
        },
        {
            "source": "SenCache TeaCache-fast rerun",
            "group": "论文",
            "threshold": "未报告",
            "equivalent_nfe": 25,
            "compute_reduction": 2.0,
            "reported_speedup": 2.0,
            "speed_scope": "由 NFE 得到的名义值",
            "psnr_db": 25.0661,
            "ssim": 0.8697,
            "lpips": 0.0966,
            "frames": "81",
            "prompts": "full VBench",
            "comparison": "同 NFE；比本地高 3.972 dB",
        },
    ]

    threshold_checks = [
        {
            "threshold": "0.08",
            "source": "本次实测",
            "reported_speedup": local_008["inference_speedup_ratio_of_sums"],
            "speed_scope": "完整推理墙钟",
            "psnr_db": local_008["psnr_rgb_db"],
            "frames": "81",
            "prompts": "10",
            "note": "本地新 GPU-T5/no-offload 扫描",
        },
        {
            "threshold": "0.08",
            "source": "EasyCache family",
            "reported_speedup": 2.0,
            "speed_scope": "DiT-module scope",
            "psnr_db": 22.57,
            "frames": "81",
            "prompts": "未统一报告",
            "note": "相同数值被 ScalingCache/VDE 重复；按一个证据家族计",
        },
        {
            "threshold": "0.08",
            "source": "MagCache",
            "reported_speedup": 2.14,
            "speed_scope": "论文 latency",
            "psnr_db": 18.14,
            "frames": "81",
            "prompts": "950",
            "note": "本地值位于 18.14–22.57 dB 之间",
        },
        {
            "threshold": "0.15",
            "source": "本地 no-offload 复测",
            "reported_speedup": 2.534717398518575,
            "speed_scope": "完整推理墙钟",
            "psnr_db": 18.636205696798672,
            "frames": "81",
            "prompts": "10",
            "note": "18 NFE；DiT compute reduction 2.7774×",
        },
        {
            "threshold": "0.15",
            "source": "SeaCache TeaCache rerun",
            "reported_speedup": 176.3 / 63.6,
            "speed_scope": "论文 latency",
            "psnr_db": 18.88,
            "frames": "65",
            "prompts": "944",
            "note": "18 NFE；本地仅低 0.244 dB",
        },
    ]

    near_two_speed = [
        {"source": "本次实测", "speedup": local_008["inference_speedup_ratio_of_sums"], "psnr_db": local_008["psnr_rgb_db"], "threshold": "0.08", "frames": "81", "protocol": "10 prompts；完整推理；25 NFE"},
        {"source": "NaviCache", "speedup": 1.77, "psnr_db": 22.79, "threshold": "—", "frames": "81", "protocol": "VBench conservative point"},
        {"source": "BudCache", "speedup": 1.89, "psnr_db": 21.52, "threshold": "—", "frames": "未说明", "protocol": "latency 189/100"},
        {"source": "TACache", "speedup": 1.91, "psnr_db": 21.74, "threshold": "—", "frames": "81", "protocol": "end-to-end；full VBench"},
        {"source": "BAG TeaCache", "speedup": 1.98, "psnr_db": 23.79, "threshold": "swept", "frames": "65", "protocol": "25 NFE；100 prompts"},
        {"source": "ProfilingDiT family", "speedup": 2.00, "psnr_db": 16.17, "threshold": "—", "frames": "81", "protocol": "该 tuple 被 ERTACache/GCache 重复"},
        {"source": "ASDSV", "speedup": 2.00, "psnr_db": 14.20, "threshold": "0.20", "frames": "81", "protocol": "A800；official defaults"},
        {"source": "EasyCache family", "speedup": 2.00, "psnr_db": 22.57, "threshold": "0.08", "frames": "81", "protocol": "DiT module；被 ScalingCache/VDE 重复"},
        {"source": "SeaCache TeaCache", "speedup": 176.3 / 86.6, "psnr_db": 20.84, "threshold": "0.09", "frames": "65", "protocol": "944 VBench prompts"},
        {"source": "SenCache TeaCache-fast", "speedup": 2.00, "psnr_db": 25.0661, "threshold": "—", "frames": "81", "protocol": "25 NFE 名义 2×；无 wall-clock"},
        {"source": "MagCache", "speedup": 2.14, "psnr_db": 18.14, "threshold": "0.08", "frames": "81", "protocol": "950 samples；seed 0"},
    ]

    timing_consistency = []
    for threshold in (0.075, 0.08, 0.10):
        row = by_threshold[round(threshold, 6)]
        compute_reduction = baseline_tflops / row["estimated_dit_tflops_per_video_mean"]
        dit_speedup = row["dit_cuda_speedup_ratio_of_sums"]
        amdahl_full = 1.0 / (fixed_share + (1.0 - fixed_share) / dit_speedup)
        timing_consistency.append(
            {
                "threshold": threshold,
                "compute_reduction": compute_reduction,
                "dit_cuda_speedup": dit_speedup,
                "dit_vs_compute_gap_percent": (dit_speedup / compute_reduction - 1.0) * 100.0,
                "observed_full_speedup": row["inference_speedup_ratio_of_sums"],
                "amdahl_expected_full_speedup": amdahl_full,
                "full_vs_amdahl_gap_percent": (row["inference_speedup_ratio_of_sums"] / amdahl_full - 1.0) * 100.0,
            }
        )

    sql_path = analysis_dir / "paper_comparison_source.sql"
    sql_path.write_text(
        "-- Exact snapshot tables embedded in paper_comparison_artifact.json.\n"
        "SELECT * FROM matched_compute ORDER BY psnr_db;\n"
        "SELECT * FROM threshold_checks ORDER BY threshold, source;\n"
        "SELECT * FROM near_two_speed ORDER BY speedup, source;\n"
        "SELECT * FROM timing_consistency ORDER BY threshold;\n",
        encoding="utf-8",
    )

    matched_source = local_source(
        "matched_compute_query",
        "25-NFE compute-matched TeaCache synthesis",
        "SELECT * FROM matched_compute ORDER BY psnr_db",
        generated_at,
    )
    threshold_source = local_source(
        "threshold_query",
        "Matched-threshold TeaCache synthesis",
        "SELECT * FROM threshold_checks ORDER BY threshold, source",
        generated_at,
    )
    speed_source = local_source(
        "near_two_query",
        "Approximately 2x TeaCache literature synthesis",
        "SELECT * FROM near_two_speed ORDER BY speedup, source",
        generated_at,
    )
    timing_source = local_source(
        "timing_query",
        "Local compute/time consistency checks",
        "SELECT * FROM timing_consistency ORDER BY threshold",
        generated_at,
    )
    query_sources = [matched_source, threshold_source, speed_source, timing_source]

    chart = {
        "id": "matched_compute_chart",
        "title": "TeaCache PSNR at 25 equivalent NFE",
        "subtitle": "Wan2.1-T2V-1.3B, 50-step baseline; protocols differ in prompts and frame count",
        "type": "bar",
        "intent": "comparison",
        "question": "Does the local 25-NFE point fall outside published TeaCache results?",
        "rationale": "Four exact compute-matched observations form a categorical comparison; bars are more honest than a fitted relationship.",
        "dataset": "matched_compute",
        "sourceId": "matched_compute_query",
        "valueFormat": "number",
        "unit": "dB",
        "layout": "wide",
        "encodings": {
            "x": {"field": "source", "type": "nominal", "label": "Source"},
            "y": {"field": "psnr_db", "type": "quantitative", "label": "PSNR", "unit": "dB"},
            "tooltip": [
                {"field": "threshold", "type": "nominal", "label": "Threshold"},
                {"field": "reported_speedup", "type": "quantitative", "label": "Reported speedup", "unit": "×"},
                {"field": "frames", "type": "nominal", "label": "Frames"},
                {"field": "prompts", "type": "nominal", "label": "Prompts"},
                {"field": "speed_scope", "type": "nominal", "label": "Speed scope"},
            ],
        },
    }

    def table(table_id: str, title: str, subtitle: str, dataset: str, source_id: str, columns: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": table_id,
            "title": title,
            "subtitle": subtitle,
            "dataset": dataset,
            "sourceId": source_id,
            "density": "spacious" if len(columns) <= 7 else "compact",
            "layout": "wide",
            "columns": columns,
        }

    matched_table = table(
        "matched_compute_table",
        "25-NFE exact comparison",
        "Absolute PSNR remains protocol-dependent even after compute normalization",
        "matched_compute",
        "matched_compute_query",
        [
            {"field": "source", "label": "Source", "type": "text"},
            {"field": "threshold", "label": "Threshold", "type": "text"},
            {"field": "reported_speedup", "label": "Reported speedup", "format": "number", "unit": "×"},
            {"field": "speed_scope", "label": "Speed scope", "type": "text"},
            {"field": "psnr_db", "label": "PSNR", "format": "number", "unit": "dB"},
            {"field": "ssim", "label": "SSIM", "format": "number"},
            {"field": "lpips", "label": "LPIPS", "format": "number"},
            {"field": "frames", "label": "Frames", "type": "text"},
            {"field": "prompts", "label": "Prompts", "type": "text"},
            {"field": "comparison", "label": "Interpretation", "type": "text"},
        ],
    )
    threshold_table = table(
        "threshold_table",
        "Matched-threshold cross-check",
        "Threshold 0.08 and 0.15; speed definitions and evaluation protocols are retained",
        "threshold_checks",
        "threshold_query",
        [
            {"field": "threshold", "label": "Threshold", "type": "text"},
            {"field": "source", "label": "Source", "type": "text"},
            {"field": "reported_speedup", "label": "Reported speedup", "format": "number", "unit": "×"},
            {"field": "speed_scope", "label": "Speed scope", "type": "text"},
            {"field": "psnr_db", "label": "PSNR", "format": "number", "unit": "dB"},
            {"field": "frames", "label": "Frames", "type": "text"},
            {"field": "prompts", "label": "Prompts", "type": "text"},
            {"field": "note", "label": "Interpretation", "type": "text"},
        ],
    )
    timing_table = table(
        "timing_table",
        "Local compute/time consistency",
        "Compute reduction should track DiT CUDA speedup; full speedup also includes fixed modules",
        "timing_consistency",
        "timing_query",
        [
            {"field": "threshold", "label": "Threshold", "format": "number"},
            {"field": "compute_reduction", "label": "DiT compute reduction", "format": "number", "unit": "×"},
            {"field": "dit_cuda_speedup", "label": "DiT CUDA speedup", "format": "number", "unit": "×"},
            {"field": "dit_vs_compute_gap_percent", "label": "DiT vs compute gap", "format": "number", "unit": "%"},
            {"field": "observed_full_speedup", "label": "Observed full speedup", "format": "number", "unit": "×"},
            {"field": "amdahl_expected_full_speedup", "label": "Amdahl expectation", "format": "number", "unit": "×"},
            {"field": "full_vs_amdahl_gap_percent", "label": "Full vs Amdahl gap", "format": "number", "unit": "%"},
        ],
    )
    speed_table = table(
        "near_two_table",
        "Approximately 2x reported TeaCache rows",
        "Wan2.1-T2V-1.3B, 50 steps; this is a weak comparison because latency boundaries differ",
        "near_two_speed",
        "near_two_query",
        [
            {"field": "source", "label": "Source", "type": "text"},
            {"field": "speedup", "label": "Reported speedup", "format": "number", "unit": "×"},
            {"field": "psnr_db", "label": "PSNR", "format": "number", "unit": "dB"},
            {"field": "threshold", "label": "Threshold", "type": "text"},
            {"field": "frames", "label": "Frames", "type": "text"},
            {"field": "protocol", "label": "Protocol note", "type": "text"},
        ],
    )

    title = "TeaCache Wan2.1：本次扫描与既有论文对照"
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}\n\n**审计截止：2026-08-30。**"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## 结论：结果正常，没有发现系统性低估或计时错误\n\n"
                "**最可信的同计算量比较支持“正常”。** 本次 threshold 0.08 恰好约为 25 equivalent NFE，PSNR 为 **21.094 dB**；SeaCache、BAG、SenCache 对 TeaCache 的 25-NFE 报告范围为 **20.84–25.066 dB**，本地位于区间内，并且比 SeaCache 高 **0.254 dB**。\n\n"
                "**同 threshold 比较也支持“正常”。** 新 no-offload threshold 0.15 的 PSNR 为 **18.636 dB**，SeaCache 为 **18.88 dB**，只差 **−0.244 dB**。threshold 0.08 时，本地 21.094 dB 也位于 MagCache 18.14 与 EasyCache family 22.57 dB 之间。\n\n"
                "**计时链路与计算量高度一致。** threshold 0.075/0.08/0.10 的 DiT CUDA speedup 与 DiT TFLOPs reduction 相差仅 **0.02%–0.17%**；完整推理 speedup 比 DiT 低，幅度与 T5/VAE/other 固定开销的 Amdahl 预测相差不到 **1.1%**。这不符合“计时口径发生明显错误”的特征。"
            ),
        },
        {
            "id": "compute_finding",
            "type": "markdown",
            "body": (
                "## 25 NFE 下，本次 PSNR 位于论文公开区间内\n\n"
                "图中四点都使用 50-step Wan2.1-T2V-1.3B，并统一到 25 次等效完整 DiT 评估。PSNR 仍横跨 4.226 dB，说明 prompt、帧数、seed、reference 构造与指标实现会显著移动绝对值。本地结果靠近 SeaCache，而不是一个跨论文离群点。"
            ),
        },
        {"id": "matched_compute_chart_block", "type": "chart", "chartId": "matched_compute_chart", "layout": "full"},
        {"id": "matched_compute_table_block", "type": "table", "tableId": "matched_compute_table"},
        {
            "id": "threshold_finding",
            "type": "markdown",
            "body": (
                "## 相同 threshold 的交叉验证没有显示 PSNR 整体偏低\n\n"
                "threshold 0.15 是最强交叉验证，因为本地与 SeaCache 不仅 threshold 相同，计算缩减也都是约 2.778×/18 NFE；两者 PSNR 仅差 0.244 dB。threshold 0.08 的论文值分布更宽，但本地仍处于已报告范围内。"
            ),
        },
        {"id": "threshold_table_block", "type": "table", "tableId": "threshold_table"},
        {
            "id": "timing_finding",
            "type": "markdown",
            "sourceId": "timing_query",
            "body": (
                "## 计时差异符合固定模块开销，而不是异常\n\n"
                "如果计时实现有明显偏差，DiT CUDA 时间通常不会在三个 threshold 上都与 trace-weighted TFLOPs reduction 对齐。当前两者最大相对差只有 0.174%。把 baseline 中 DiT CUDA 之外的 4.38% 时间视为固定项后，Amdahl 预测的完整加速与实测最大只差 1.09%。因此 0.08 的 2.00× DiT 加速落成 1.898× 完整推理加速是预期现象。"
            ),
        },
        {"id": "timing_table_block", "type": "table", "tableId": "timing_table"},
        {
            "id": "speed_finding",
            "type": "markdown",
            "body": (
                "## 只按“约 2×”看，本次处于正常区间的中下部\n\n"
                "现有 Wan2.1-T2V-1.3B、50-step TeaCache 报告在约 1.77–2.14× 时从 14.20 到 25.066 dB；本次 21.094 dB 不低于该范围，也接近这些代表性公开值的中位区域。但这一比较把完整 pipeline、DiT-module latency 和 NFE 名义 speedup 混在一起，只能作为弱证据，不能替代 matched-compute 比较。"
            ),
        },
        {"id": "speed_table_block", "type": "table", "tableId": "near_two_table"},
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## 范围、数据与指标定义\n\n"
                "本地实验为 Wan2.1-T2V-1.3B、固定 10 条 VBench200 prompt、832×480、81 帧、50-step UniPC、CFG 5、shift 5、seed 42、BF16；T5/DiT/VAE 均在 GPU，`offload_model=False`。完整推理时间包含 text encoding、denoising、VAE decode 和 pipeline other，排除模型加载、MP4 导出与指标计算。PSNR/SSIM/LPIPS 将相同 prompt/seed 的 TeaCache 视频与 50-step baseline 视频逐帧配对。"
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "body": (
                "## 比较方法\n\n"
                "证据优先级依次为：相同 equivalent NFE/compute fraction、相同 threshold、相近报告 speedup。绝对 TFLOPs 不跨论文直接比较，因为帧数与 profiler 口径不同；统一使用 candidate DiT compute 除以各自 50-step baseline compute，再换算 equivalent NFE。重复出现的完全相同 TeaCache tuple 按一个证据家族处理，模型、任务或 step 数不一致的论文不进入主判断。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 限制与稳健性\n\n"
                "本地只有 10 prompts、单 seed、每点一次 timed run，因此不能给出跨数据集置信区间；SeaCache 使用 944 prompts/65 帧，BAG 使用 100 prompts/65 帧，SenCache 使用 full VBench/81 帧，绝对 PSNR 不应被解释成同一总体均值。当前判断是“没有系统性偏差证据”，不是证明所有协议下都能复现任一论文的绝对数值。"
            ),
        },
        {
            "id": "recommendations",
            "type": "markdown",
            "body": (
                "## 报告口径建议\n\n"
                "论文方法比较以 **equivalent NFE / DiT TFLOPs reduction + PSNR/SSIM/LPIPS** 为主；部署收益另报 **完整推理墙钟 speedup**，并同时给出 DiT CUDA speedup 和分模块时间。不要从另一篇论文的 wall-clock speedup 反推我们的 threshold，也不要把 SenCache 的 NFE 名义 2× 当成实测完整推理 2×。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 尚不能由现有证据回答的问题\n\n"
                "完整 VBench200 上本地 25-NFE PSNR 的均值和方差是否仍靠近 SeaCache；各论文 reference 视频是否完全共享相同 seed、采样器与视频编码路径；以及在统一硬件与统一 timer 边界下，不同 TeaCache rerun 的 wall-clock 是否会收敛。"
            ),
        },
    ]

    manifest_sources = [{key: value for key, value in source.items() if key != "query"} for source in query_sources]
    manifest_sources.extend(
        [
            {"id": "seacache", "label": "SeaCache (CVPR 2026)", "href": "https://arxiv.org/html/2602.18993v2"},
            {"id": "bag", "label": "BAG", "href": "https://arxiv.org/html/2608.09231v2"},
            {"id": "sencache", "label": "SenCache (CVPR 2026)", "href": "https://arxiv.org/html/2602.24208v1"},
            {"id": "profilingdit", "label": "ProfilingDiT (ICCV 2025)", "href": "https://arxiv.org/html/2504.03140v2"},
            {"id": "easycache", "label": "EasyCache", "href": "https://arxiv.org/abs/2507.02860"},
            {"id": "magcache", "label": "MagCache (NeurIPS 2025)", "href": "https://proceedings.neurips.cc/paper_files/paper/2025/file/311207bb626e36a8f1d3eb92aa67af22-Paper-Conference.pdf"},
        ]
    )
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Protocol-aware diagnostic of the new GPU-T5/no-offload TeaCache scan against prior Wan2.1-T2V-1.3B reports.",
            "generatedAt": generated_at,
            "charts": [chart],
            "tables": [matched_table, threshold_table, timing_table, speed_table],
            "sources": manifest_sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "matched_compute": matched_compute,
                "threshold_checks": threshold_checks,
                "timing_consistency": timing_consistency,
                "near_two_speed": near_two_speed,
            },
        },
        "sources": query_sources
        + [
            {"id": "seacache", "label": "SeaCache (CVPR 2026)", "href": "https://arxiv.org/html/2602.18993v2"},
            {"id": "bag", "label": "BAG", "href": "https://arxiv.org/html/2608.09231v2"},
            {"id": "sencache", "label": "SenCache (CVPR 2026)", "href": "https://arxiv.org/html/2602.24208v1"},
            {"id": "profilingdit", "label": "ProfilingDiT (ICCV 2025)", "href": "https://arxiv.org/html/2504.03140v2"},
            {"id": "easycache", "label": "EasyCache", "href": "https://arxiv.org/abs/2507.02860"},
            {"id": "magcache", "label": "MagCache (NeurIPS 2025)", "href": "https://proceedings.neurips.cc/paper_files/paper/2025/file/311207bb626e36a8f1d3eb92aa67af22-Paper-Conference.pdf"},
        ],
    }
    output = analysis_dir / "paper_comparison_artifact.json"
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "artifact": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
