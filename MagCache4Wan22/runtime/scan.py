"""Plan official parameter grids; select targets only from measured wall times."""
from __future__ import annotations
import ast
import copy
import itertools
import math
from decimal import Decimal
from common import PROJECT
from official import load_official, method_args


def official_schedule(threshold, k, retention_ratio):
    # Execute the verbatim decision statements of the pinned upstream function.
    # This is a CPU planning tool, never a replacement for the inference forward.
    core = load_official(sinusoidal_embedding=lambda *a: None)
    tree = ast.parse((PROJECT / "vendor/magcache_generate.py").read_text())
    forward = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "magcache_forward")
    start = next(i for i, n in enumerate(forward.body)
                 if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "use_magcache")
    stop = next(i for i, n in enumerate(forward.body[start:], start)
                if isinstance(n, ast.If) and ast.unparse(n.test) == "skip_forward")
    body = ast.Module(body=copy.deepcopy(forward.body[start:stop]), type_ignores=[])
    code = compile(ast.fix_missing_locations(body), "<official-magcache-gate>", "exec")
    class Planner:
        pass
    model = Planner()
    options = method_args(threshold, k, retention_ratio)
    split = int((core.get_timesteps(shift=12, num_inference_steps=50) >= 875).sum())
    core.init_magcache(model, core.ratios, options, split_steps=split, mode="t2v")
    actions = []
    for i in range(100):
        # Original scalar tensor counter and independent branch accumulators.
        model.__class__.cnt.fill_(i)
        namespace = dict(core.namespace, self=model, x=0)
        exec(code, namespace)
        actions.append("reuse" if namespace["skip_forward"] else "recompute")
    if "reuse" in actions[:2]:
        raise ValueError("retention must protect both initial CFG calls before official cache reuse")
    return actions


def grid_conditions(thresholds, ks, retentions, deduplicate=True):
    conditions, by_schedule = [], {}
    for threshold, k, retention in itertools.product(thresholds, ks, retentions):
        options = method_args(threshold, k, retention)
        params = dict(threshold=options.magcache_thresh, K=options.magcache_K,
                      retention_ratio=options.retention_ratio)
        actions = official_schedule(threshold, k, retention)
        key = tuple(actions)
        if deduplicate and key in by_schedule:
            by_schedule[key]["equivalent_parameters"].append(params)
            continue
        condition = dict(id=f"magcache_{len(conditions):03d}", mode="magcache", **params,
                         schedule=actions, planned_full_calls=actions.count("recompute"),
                         planned_reuse_calls=actions.count("reuse"), equivalent_parameters=[])
        conditions.append(condition)
        by_schedule[key] = condition
    return conditions


def select_targets(performance, targets=(1.8, 2.4, 3.0), tolerance=.1):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("target tolerance must be positive and finite")
    if not performance:
        raise ValueError("target selection requires measured performance")
    if any(not math.isfinite(p["speedup"]) or p["speedup"] <= 0 for p in performance.values()):
        raise ValueError("invalid measured speedup")
    result = []
    for target in targets:
        if not math.isfinite(target) or target <= 1:
            raise ValueError("targets must be finite and greater than one")
        name, value = min(performance.items(), key=lambda item: (abs(item[1]["speedup"] - target), item[0]))
        delta = value["speedup"] - target
        result.append(dict(target=target, tolerance=tolerance,
                           status="hit" if abs(delta) <= tolerance + 1e-12 else "unmet",
                           nearest_condition=name, measured_speedup=value["speedup"],
                           delta=delta, parameters=value["method"]))
    return dict(metric="ratio_of_summed_complete_generate_wall_seconds", targets=result,
                note="Nearest measured grid point; unmet targets require further measurements. "
                     "Equivalent schedules are aliases, not separately timed configurations.")


def preset_conditions(presets, targets=None, tolerance=None, deduplicate=True):
    """Hold R/K fixed per target; expand decimal E grids without rounding drift."""
    if presets.get("schema") != "magcache4wan22_target_presets_v1":
        raise ValueError("unsupported target preset schema")
    relative = presets["relative_tolerance"]
    if not math.isfinite(relative) or not 0 < relative < 1:
        raise ValueError("relative tolerance must be between zero and one")
    rows = presets["targets"]
    available = [r["target_speedup"] for r in rows]
    requested = available if targets is None else targets
    if (not requested or len(set(available)) != len(available)
            or len(set(requested)) != len(requested) or not set(requested) <= set(available)):
        raise ValueError("requested targets must be distinct and present in presets")
    conditions, groups = [], []
    for row in rows:
        target = row["target_speedup"]
        if target not in requested:
            continue
        allowed_error = target * relative if tolerance is None else tolerance
        select_targets({"validation": dict(speedup=1., method={})}, [target], allowed_error)
        grid = row["threshold_range"]
        start, stop, step = (Decimal(str(grid[key])) for key in ("start", "stop", "step"))
        center = Decimal(str(row["recommended_threshold"]))
        if (not all(v.is_finite() for v in (start, stop, step, center))
                or start < 0 or stop < start or step <= 0 or not start <= center <= stop):
            raise ValueError("invalid preset threshold range")
        intervals = (stop - start) / step
        if intervals != intervals.to_integral_value() or intervals > 10000:
            raise ValueError("threshold range must end on the grid and contain at most 10001 points")
        if (center - start) % step:
            raise ValueError("recommended threshold must lie on the grid")
        values = [start + i * step for i in range(int(intervals) + 1)]
        # Keep the recommended E as the measured representative of its plateau.
        thresholds = [float(center)] + [float(v) for v in values if v != center]
        candidates = grid_conditions(thresholds, [row["K"]], [row["retention_ratio"]], deduplicate)
        prefix = "target_" + str(target).replace(".", "p")
        for i, candidate in enumerate(candidates):
            candidate["id"] = f"{prefix}_{i:03d}"
        groups.append(dict(target=target, tolerance=allowed_error,
                           tolerance_kind="relative" if tolerance is None else "absolute",
                           relative_tolerance=relative if tolerance is None else None,
                           K=row["K"], retention_ratio=row["retention_ratio"],
                           recommended_threshold=float(center), threshold_range=grid,
                           grid_point_count=len(thresholds),
                           condition_ids=[c["id"] for c in candidates]))
        conditions.extend(candidates)
    return conditions, groups


def select_preset_targets(performance, groups):
    """A target can only select measured points from its own fixed R/K group."""
    selected = []
    for group in groups:
        pool = {name: performance[name] for name in group["condition_ids"]}
        for value in pool.values():
            if any(value["method"][key] != group[key] for key in ("K", "retention_ratio")):
                raise ValueError("measured R/K differs from target preset")
        item = select_targets(pool, [group["target"]], group["tolerance"])["targets"][0]
        item.update(tolerance_kind=group["tolerance_kind"],
                    relative_tolerance=group["relative_tolerance"],
                    recommended_threshold=group["recommended_threshold"],
                    condition_ids=group["condition_ids"])
        selected.append(item)
    return dict(metric="ratio_of_summed_complete_generate_wall_seconds", targets=selected,
                note="Each target selects within its fixed R/K group. Exact speed is not guaranteed: "
                     "E creates discrete schedules and measured latency fluctuates. "
                     "Equivalent schedules are aliases, not independently timed configurations.")
