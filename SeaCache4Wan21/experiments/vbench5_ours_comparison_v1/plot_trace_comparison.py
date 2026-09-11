"""Export exact, source-backed 50-step SeaCache/SEA7 action matrices."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, Rectangle

DEFAULT = Path("/mnt/hdd/xiongyuxiang/tmp/exp/seacache_wan21_vbench5_ours_comparison_4gpu_v1/analysis/trace_psnr_audit")
TARGETS = (1.8, 2.4, 3.0)
PROMPTS = ("001", "016", "056", "135", "159")
METHODS = ("SeaCache", "SEA7")
BLUE, BORDER, TEXT = "#3B6FD4", "#CAD1DA", "#222B38"


def build(root):
    source = root / "trace_steps.csv"
    rows = list(csv.DictReader(source.open()))
    audit = json.loads((root / "AUDIT.json").read_text())
    assert audit["status"] == "pass"
    assert len(rows) == 3000
    groups = {}
    for row in rows:
        key = (float(row["target"]), row["sample_id"][-3:], row["method"], row["branch"])
        groups.setdefault(key, []).append(row)
        reuse = int(row["reuse"])
        assert reuse in (0, 1)
        assert int(row["executed_blocks"]) == (0 if reuse else 30)
        assert row["action"] == ("reuse" if reuse else "recompute")
    assert set(groups) == {(t, p, m, b) for t in TARGETS for p in PROMPTS
                           for m in METHODS for b in ("cond", "uncond")}
    for values in groups.values():
        values.sort(key=lambda r: int(r["step"]))
        assert [int(r["step"]) for r in values] == list(range(50))
        assert sum(int(r["reuse"]) for r in values) == int(values[0]["skip_count"])
    for t in TARGETS:
        for p in PROMPTS:
            for m in METHODS:
                assert [r["reuse"] for r in groups[t, p, m, "cond"]] == [
                    r["reuse"] for r in groups[t, p, m, "uncond"]]
    comparison = root.parent / "comparison.csv"
    summary = {(float(r["target"]), r["label"]): r
               for r in csv.DictReader(comparison.open())}
    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 11,
                         "text.color": TEXT, "axes.labelcolor": TEXT,
                         "svg.fonttype": "path"})
    fig, axes = plt.subplots(3, 1, figsize=(20, 14))
    fig.subplots_adjust(left=.13, right=.88, top=.84, bottom=.085, hspace=.52)
    fig.text(.045, .96, "SeaCache 与 SEA7：50 步逐 prompt 动作对比", fontsize=23, weight="bold")
    fig.text(.045, .927, "Wan2.1-1.3B · 5 条 VBench200 子集 prompt · seed=42 · 每格为一个采样步（0–49）", fontsize=13)
    fig.legend(handles=[Patch(facecolor=BLUE, edgecolor=BLUE, label="复用 / 跳过 DiT blocks"),
                        Patch(facecolor="white", edgecolor=BORDER, label="重新计算")],
               loc="upper left", bbox_to_anchor=(.04, .91), frameon=False, ncol=2, fontsize=12)
    for ax, target in zip(axes, TARGETS):
        sea, ours = summary[target, "SeaCache"], summary[target, "SEA7"]
        sample = groups[target, "001", "SeaCache", "cond"][0]
        title = (f"名义 {target:.1f}×   |   SeaCache τ={float(sample['threshold']):.6f} / SEA7 K={ours['k']}"
                 f"   |   实测 {float(sea['speedup']):.3f}× / {float(ours['speedup']):.3f}×")
        ax.set_title(title, loc="left", fontsize=13, pad=29, weight="bold")
        ys, labels = [], []
        for pi, prompt in enumerate(PROMPTS):
            for mi, method in enumerate(METHODS):
                y = pi * 2.45 + mi
                ys.append(y)
                labels.append(f"{prompt}   {method}")
                values = groups[target, prompt, method, "cond"]
                for step, row in enumerate(values):
                    reuse = int(row["reuse"])
                    ax.add_patch(Rectangle((step-.46, y-.40), .92, .80,
                                           facecolor=BLUE if reuse else "white",
                                           edgecolor=BLUE if reuse else BORDER, linewidth=.6))
                ax.text(50.1, y, f"{values[0]['skip_count']:>2}/50    {float(values[0]['psnr']):.2f}",
                        va="center", fontsize=10)
        ax.text(50.1, -.9, "复用步数  PSNR/dB", fontsize=10)
        ax.set_xlim(-.6, 49.6)
        ax.set_ylim(11.5, -.65)
        ax.set_yticks(ys, labels, fontsize=11)
        ax.set_xticks(range(50), [str(i) for i in range(50)], fontsize=8)
        ax.xaxis.tick_top()
        ax.tick_params(axis="both", length=0, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.045, .040, "注：两种方法各自的 cond / uncond 动作逐步一致，图中无损合并；原始 3,000 条分支动作保留。", fontsize=11)
    fig.text(.045, .018, "按名义档位并列，不代表严格等速（尤其 2.4× 档）；PSNR 为每视频 81 帧均值。动作差异本身不构成质量差异的因果证明。", fontsize=11)
    output = root / "figures"
    output.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"trace_comparison_50_steps.{suffix}", dpi=160, facecolor="white")
    plt.close(fig)
    qa = {"status": "pass", "raw_rows": len(rows), "plotted_cells": 1500,
          "panels": 3, "rows_per_panel": 10, "steps": list(range(50)),
          "cfg_action_equality": True, "executed_blocks_parity": True,
          "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (source, comparison, root / "AUDIT.json", Path(__file__))}}
    (output / "plot_validation.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT)
    build(parser.parse_args().audit_root)
