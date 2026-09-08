"""Data-dependent DiCache schedules require measured parameter scans."""
from __future__ import annotations
import itertools
import math
from official import method_args


def grid_conditions(thresholds, retentions):
    conditions, seen = [], set()
    for threshold, retention in itertools.product(thresholds, retentions):
        method_args(threshold, retention)
        if (threshold, retention) in seen:
            continue
        seen.add((threshold, retention))
        conditions.append(dict(id=f"dicache_{len(conditions):03d}", mode="dicache",
                               threshold=threshold, retention_ratio=retention))
    if not conditions:
        raise ValueError("empty parameter grid")
    # Do not borrow MagCache's precomputed schedule deduplication: DiCache gates
    # depend on live shallow features, so two parameters cannot be merged a priori.
    return conditions


def select_targets(performance, targets, tolerance):
    if (not targets or len(set(targets)) != len(targets)
            or any(not math.isfinite(x) or x <= 1 for x in targets)
            or not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError("invalid target speeds or tolerance")
    if not performance or any(not math.isfinite(v["speedup"]) or v["speedup"] <= 0 for v in performance.values()):
        raise ValueError("invalid measured speedups")
    rows = []
    for target in targets:
        name, value = min(performance.items(), key=lambda kv: (abs(kv[1]["speedup"]-target), kv[0]))
        error = abs(value["speedup"] - target)
        rows.append(dict(target_speedup=target, condition_id=name, speedup=value["speedup"],
                         absolute_error=error, tolerance=tolerance,
                         status="matched" if error <= tolerance else "unmatched",
                         method=value["method"]))
    return dict(metric="ratio_of_summed_complete_generate_wall_seconds", targets=rows,
                note="Measured nearest candidates only; unmatched targets are not claimed achieved.")
