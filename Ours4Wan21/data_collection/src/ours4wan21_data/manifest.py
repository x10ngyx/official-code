#!/usr/bin/env python3
"""Build randomized-path and fixed-threshold manifests for Wan2.1 data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_PLAN = "ours4wan21_random_threshold_plan_v1"
SCHEMA_RUNNABLE = "ours4wan21_random_threshold_runnable_v1"
SCHEMA_SEACACHE = "ours4wan21_seacache_threshold_manifest_v1"
SCHEMA_CANDIDATE_COMPLETE = "ours4wan21_candidate_complete_v3"
MAPPING_SCHEMA = "ours4wan21_speed_threshold_mapping_v1"
SEACACHE_THRESHOLD_CONFIG_SCHEMA = "ours4wan21_seacache_threshold_grid_v1"
NUM_STEPS = 50
NUM_SOURCE_PROMPTS = 5000
NUM_SELECTED_PROMPTS = 3000
CANDIDATES_PER_PROMPT = 3
NUM_CANDIDATES = NUM_SELECTED_PROMPTS * CANDIDATES_PER_PROMPT
SEACACHE_NUM_SELECTED_PROMPTS = 1000
SEACACHE_CANDIDATES_PER_PROMPT = 3
SEACACHE_NUM_CANDIDATES = (
    SEACACHE_NUM_SELECTED_PROMPTS * SEACACHE_CANDIDATES_PER_PROMPT
)
SEACACHE_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70)
NUM_SHARDS = 4
FORCED_RECOMPUTE_STEPS = frozenset({0, NUM_STEPS - 1})
ELIGIBLE_STEPS = tuple(i for i in range(NUM_STEPS) if i not in FORCED_RECOMPUTE_STEPS)
TARGET_SPEEDUP_MIN = 1.5
TARGET_SPEEDUP_MAX = 3.5
Q_MIN = 0.2
Q_MAX = 1.0
INTERIOR_COUNTS = tuple(range(6))
MIN_INTERIOR_STEP = 4
MAX_INTERIOR_STEP = 45
MIN_INTERIOR_SEPARATION = 5
MAX_NORMALIZED_STEP_DELTA = 0.12
MEAN_TOLERANCE_FRACTION = 0.20
DEFAULT_SEED = 20260722
DEFAULT_PROMPT_SELECTION_SEED = 2026073001
EXPECTED_PROMPT_POOL_SHA256 = (
    "fb5d5d73f86b84d10d8e55154b789ac8549c74e90f33c1d4d2a02d67a5cde3e5"
)
PROTOCOL = {
    "model": "Wan2.1-T2V-1.3B",
    "task": "t2v-1.3B",
    "size_wh": [832, 480],
    "frame_num": 81,
    "fps": 16,
    "sample_steps": 50,
    "sample_solver": "unipc",
    "sample_shift": 5.0,
    "cfg": 5.0,
    "seed": 42,
    "parameter_dtype": "bfloat16",
    "offload_model": False,
    "t5_cpu": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"expected object at {path}:{number}")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in materialized),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "release_index", "trajectory_id", "prompt_rank", "sample_id", "split",
        "candidate_index_for_prompt", "shard_index", "target_speedup", "q",
        "mean_threshold", "calibration_status", "prompt",
    ]
    ordered = [key for key in preferred if key in fields] + [
        key for key in fields if key not in preferred
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def _normalized_prompt(row: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
    prompt = " ".join(str(row.get("prompt") or row.get("text") or "").split())
    video = str(row.get("video") or "").strip()
    if not sample_id or not prompt or not video:
        raise ValueError("prompt pool rows require unique id, non-empty text, and video")
    return {**row, "sample_id": sample_id, "prompt": prompt}


def load_prompt_pool(path: Path) -> list[dict[str, Any]]:
    observed_hash = sha256(path)
    if observed_hash != EXPECTED_PROMPT_POOL_SHA256:
        raise ValueError(
            "OpenVid balanced-5000 checksum mismatch: "
            f"expected {EXPECTED_PROMPT_POOL_SHA256}, got {observed_hash}"
        )
    rows = [_normalized_prompt(row) for row in read_jsonl(path)]
    if len(rows) != NUM_SOURCE_PROMPTS:
        raise ValueError(f"expected {NUM_SOURCE_PROMPTS} prompts, got {len(rows)}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("source prompt IDs are not unique")
    if len({row["video"] for row in rows}) != len(rows):
        raise ValueError("source video IDs are not unique")
    if len({row["prompt"].casefold() for row in rows}) != len(rows):
        raise ValueError("source prompt texts are not unique")
    if len({str(row.get("part") or "") for row in rows}) != 98:
        raise ValueError("source prompt pool must retain all 98 OpenVidHD parts")
    return rows


def smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def interpolate_knots(steps: list[int], values: list[float]) -> list[float]:
    if len(steps) != len(values) or steps[0] != 0 or steps[-1] != NUM_STEPS - 1:
        raise ValueError("invalid control points")
    result = [0.0] * NUM_STEPS
    for left in range(len(steps) - 1):
        start, end = steps[left], steps[left + 1]
        if end <= start:
            raise ValueError("control-point steps must increase")
        for step in range(start, end + 1):
            unit = (step - start) / float(end - start)
            result[step] = values[left] + smoothstep(unit) * (
                values[left + 1] - values[left]
            )
    return result


def sample_interior_steps(rng: random.Random, count: int) -> list[int]:
    legal = list(range(MIN_INTERIOR_STEP, MAX_INTERIOR_STEP + 1))
    for _ in range(512):
        selected = sorted(rng.sample(legal, count))
        if all(b - a >= MIN_INTERIOR_SEPARATION for a, b in zip(selected, selected[1:])):
            return selected
    raise RuntimeError(f"could not sample {count} separated control points")


def sample_residual_path(seed: int, count: int) -> tuple[list[float], dict[str, Any]]:
    rng = random.Random(seed)
    interior = sample_interior_steps(rng, count)
    steps = [0, *interior, NUM_STEPS - 1]
    values = [rng.uniform(-1.0, 1.0) for _ in steps]
    raw = interpolate_knots(steps, values)
    residual_mean = sum(raw) / len(raw)
    centered = [value - residual_mean for value in raw]
    peak = max(abs(value) for value in centered)
    if peak <= 1e-12:
        raise RuntimeError("degenerate residual path")
    residual = [value / peak for value in centered]
    return residual, {
        "interior_reference_point_count": count,
        "interior_reference_steps": interior,
        "knot_steps": steps,
        "raw_knots": values,
        "reference_position_distribution": "uniform_legal_steps_conditioned_min_spacing_5",
        "reference_value_distribution": "independent_uniform_-1_1",
        "residual_mean_before_centering": residual_mean,
        "residual_peak_before_normalization": peak,
    }


def build_plan(
    prompt_pool: Path,
    seed: int,
    prompt_selection_seed: int = DEFAULT_PROMPT_SELECTION_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = load_prompt_pool(prompt_pool)
    selector = random.Random(prompt_selection_seed)
    selected = selector.sample(pool, NUM_SELECTED_PROMPTS)
    rows: list[dict[str, Any]] = []
    for prompt_rank, prompt in enumerate(selected):
        split = "train" if prompt_rank < 2400 else ("val" if prompt_rank < 2700 else "test")
        for local_index in range(CANDIDATES_PER_PROMPT):
            release_index = len(rows)
            target_seed = seed + 3_000_001 * (release_index + 1)
            q_seed = seed + 2_100_003 * (release_index + 1)
            reference_count_seed = seed + 4_000_003 * (release_index + 1)
            schedule_seed = seed + 1_000_003 * (release_index + 1)
            target_rng = random.Random(target_seed)
            target_random_unit = target_rng.random()
            target_speedup = TARGET_SPEEDUP_MIN + (
                TARGET_SPEEDUP_MAX - TARGET_SPEEDUP_MIN
            ) * target_random_unit
            q_rng = random.Random(q_seed)
            q_random_unit = q_rng.random()
            q = Q_MIN + (Q_MAX - Q_MIN) * q_random_unit
            reference_count_random_unit = random.Random(reference_count_seed).random()
            reference_count = min(
                int(reference_count_random_unit * len(INTERIOR_COUNTS)),
                len(INTERIOR_COUNTS) - 1,
            )
            residual, provenance = sample_residual_path(schedule_seed, reference_count)
            rows.append({
                "schema": SCHEMA_PLAN,
                "release_index": release_index,
                "trajectory_id": f"{prompt['sample_id']}__rt{release_index:04d}",
                "prompt_rank": prompt_rank,
                "prompt_selection_rank": prompt_rank,
                "sample_id": prompt["sample_id"],
                "prompt": prompt["prompt"],
                "split": split,
                "part": prompt.get("part", ""),
                "content_group": prompt.get("content_group", ""),
                "length_group": prompt.get("length_group", ""),
                "motion_group": prompt.get("motion_group", ""),
                "topic_tag": prompt.get("topic_tag", ""),
                "candidate_index_for_prompt": local_index,
                "shard_index": release_index % NUM_SHARDS,
                "num_shards": NUM_SHARDS,
                "target_speedup": target_speedup,
                "target_speedup_seed": target_seed,
                "target_speedup_random_unit": target_random_unit,
                "target_speedup_distribution": "continuous_uniform_[1.5,3.5]",
                "q": q,
                "q_seed": q_seed,
                "q_random_unit": q_random_unit,
                "q_distribution": "continuous_uniform_[0.2,1.0]",
                "reference_point_count_seed": reference_count_seed,
                "reference_point_count_random_unit": reference_count_random_unit,
                "reference_point_count_distribution": "discrete_uniform_integer_0_through_5",
                "schedule_seed": schedule_seed,
                "manifest_seed": seed,
                "prompt_selection_seed": prompt_selection_seed,
                "prompt_selection_distribution": "uniform_without_replacement_3000_of_5000",
                "normalized_residual_path": residual,
                "forced_recompute_steps": sorted(FORCED_RECOMPUTE_STEPS),
                "policy_family": "random_continuous_seacache_threshold",
                "calibration_status": "pending",
                "mean_threshold": None,
                "threshold_path": None,
                "protocol": PROTOCOL,
                **provenance,
            })
    validate_plan(rows)
    summary = {
        "schema": SCHEMA_PLAN,
        "prompt_pool": str(prompt_pool.resolve()),
        "prompt_pool_sha256": sha256(prompt_pool),
        "selection_distribution": "uniform_without_replacement_3000_of_5000",
        "prompt_selection_seed": prompt_selection_seed,
        "seed": seed,
        "source_prompt_count": len(pool),
        "selected_prompt_count": len(selected),
        "selected_part_count": len({str(row.get("part") or "") for row in selected}),
        "candidate_count": len(rows),
        "candidates_per_prompt": CANDIDATES_PER_PROMPT,
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "prompt_split_counts": {"train": 2400, "val": 300, "test": 300},
        "shard_counts": dict(Counter(row["shard_index"] for row in rows)),
        "target_speedup_distribution": "continuous_uniform_[1.5,3.5]",
        "q_distribution": "continuous_uniform_[0.2,1.0]",
        "calibration_status": "pending",
        "candidate_runnable": False,
        "protocol": PROTOCOL,
    }
    return rows, summary


def load_seacache_threshold_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SEACACHE_THRESHOLD_CONFIG_SCHEMA:
        raise ValueError("invalid SeaCache threshold-grid config schema")
    thresholds = tuple(float(value) for value in payload.get("thresholds", []))
    if thresholds != SEACACHE_THRESHOLDS:
        raise ValueError(
            "SeaCache threshold grid must exactly match the frozen local Wan2.2 list: "
            f"{list(SEACACHE_THRESHOLDS)}"
        )
    if int(payload.get("thresholds_per_prompt", -1)) != SEACACHE_CANDIDATES_PER_PROMPT:
        raise ValueError("SeaCache config must select exactly three thresholds per prompt")
    if payload.get("sampling_without_replacement") is not True:
        raise ValueError("SeaCache thresholds must be sampled without replacement per prompt")
    source_files = payload.get("source_files")
    if (
        not isinstance(source_files, dict)
        or set(source_files)
        != {"README.md", "launch_queued_4gpu.sh", "queue_then_run_shard.sh"}
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in source_files.values()
        )
    ):
        raise ValueError("SeaCache threshold config requires frozen Wan2.2 source hashes")
    return payload


def build_seacache_manifest(
    prompt_pool: Path,
    threshold_config: Path,
    seed: int = DEFAULT_SEED,
    prompt_selection_seed: int = DEFAULT_PROMPT_SELECTION_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze 1,000 prompts and three fixed SeaCache thresholds per prompt."""

    pool = load_prompt_pool(prompt_pool)
    config = load_seacache_threshold_config(threshold_config)
    selector = random.Random(prompt_selection_seed)
    selected = selector.sample(pool, SEACACHE_NUM_SELECTED_PROMPTS)
    rows: list[dict[str, Any]] = []
    config_hash = sha256(threshold_config)
    for prompt_rank, prompt in enumerate(selected):
        split = "train" if prompt_rank < 800 else ("val" if prompt_rank < 900 else "test")
        threshold_seed = seed + 5_000_003 * (prompt_rank + 1)
        sampled = random.Random(threshold_seed).sample(list(SEACACHE_THRESHOLDS), 3)
        for local_index, threshold in enumerate(sampled):
            release_index = len(rows)
            rows.append({
                "schema": SCHEMA_SEACACHE,
                "release_index": release_index,
                "trajectory_id": f"{prompt['sample_id']}__sc{release_index:04d}",
                "prompt_rank": prompt_rank,
                "prompt_selection_rank": prompt_rank,
                "sample_id": prompt["sample_id"],
                "prompt": prompt["prompt"],
                "split": split,
                "part": prompt.get("part", ""),
                "content_group": prompt.get("content_group", ""),
                "length_group": prompt.get("length_group", ""),
                "motion_group": prompt.get("motion_group", ""),
                "topic_tag": prompt.get("topic_tag", ""),
                "candidate_index_for_prompt": local_index,
                "shard_index": release_index % NUM_SHARDS,
                "num_shards": NUM_SHARDS,
                "target_speedup": None,
                "q": None,
                "fixed_threshold": threshold,
                "mean_threshold": threshold,
                "threshold_min": threshold,
                "threshold_max": threshold,
                "threshold_path": [threshold] * NUM_STEPS,
                "threshold_grid": list(SEACACHE_THRESHOLDS),
                "threshold_grid_config": str(threshold_config.resolve()),
                "threshold_grid_config_sha256": config_hash,
                "threshold_selection_seed": threshold_seed,
                "threshold_selection_distribution": (
                    "uniform_without_replacement_3_of_9_per_prompt"
                ),
                "manifest_seed": seed,
                "prompt_selection_seed": prompt_selection_seed,
                "prompt_selection_distribution": "uniform_without_replacement_1000_of_5000",
                "forced_recompute_steps": sorted(FORCED_RECOMPUTE_STEPS),
                "policy_family": "fixed_seacache_threshold",
                "calibration_status": "not_applicable_fixed_threshold",
                "protocol": PROTOCOL,
            })
    validate_seacache_manifest(rows)
    summary = {
        "schema": SCHEMA_SEACACHE,
        "prompt_pool": str(prompt_pool.resolve()),
        "prompt_pool_sha256": sha256(prompt_pool),
        "threshold_grid_config": str(threshold_config.resolve()),
        "threshold_grid_config_sha256": config_hash,
        "threshold_grid_source": config.get("source_reference"),
        "threshold_grid_source_files": config.get("source_files"),
        "thresholds": list(SEACACHE_THRESHOLDS),
        "threshold_sampling": "uniform_without_replacement_3_of_9_per_prompt",
        "selection_distribution": "uniform_without_replacement_1000_of_5000",
        "prompt_selection_seed": prompt_selection_seed,
        "seed": seed,
        "source_prompt_count": len(pool),
        "selected_prompt_count": len(selected),
        "selected_part_count": len({str(row.get("part") or "") for row in selected}),
        "candidate_count": len(rows),
        "candidates_per_prompt": SEACACHE_CANDIDATES_PER_PROMPT,
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "prompt_split_counts": {"train": 800, "val": 100, "test": 100},
        "shard_counts": dict(Counter(row["shard_index"] for row in rows)),
        "candidate_runnable": True,
        "protocol": PROTOCOL,
    }
    return rows, summary


def validate_seacache_manifest(rows: list[dict[str, Any]]) -> None:
    if len(rows) != SEACACHE_NUM_CANDIDATES:
        raise ValueError(
            f"expected {SEACACHE_NUM_CANDIDATES} SeaCache rows, got {len(rows)}"
        )
    if [row.get("release_index") for row in rows] != list(range(SEACACHE_NUM_CANDIDATES)):
        raise ValueError("SeaCache release_index must be contiguous")
    if len({row.get("trajectory_id") for row in rows}) != SEACACHE_NUM_CANDIDATES:
        raise ValueError("SeaCache trajectory IDs are not unique")
    prompt_counts = Counter(str(row.get("sample_id")) for row in rows)
    if (
        len(prompt_counts) != SEACACHE_NUM_SELECTED_PROMPTS
        or set(prompt_counts.values()) != {SEACACHE_CANDIDATES_PER_PROMPT}
    ):
        raise ValueError("SeaCache manifest must contain three rows for each of 1,000 prompts")
    if Counter(int(row["shard_index"]) for row in rows) != Counter({i: 750 for i in range(4)}):
        raise ValueError("SeaCache candidate shards must be exactly 750 each")
    if Counter(str(row["split"]) for row in rows) != Counter(train=2400, val=300, test=300):
        raise ValueError("SeaCache candidate split must be 2400/300/300")
    if len({str(row.get("part") or "") for row in rows}) != 98:
        raise ValueError("selected SeaCache prompts must retain all 98 OpenVidHD parts")
    config_hashes = {str(row.get("threshold_grid_config_sha256") or "") for row in rows}
    if len(config_hashes) != 1 or len(next(iter(config_hashes))) != 64:
        raise ValueError("SeaCache threshold-grid config checksum is missing or inconsistent")
    for prompt_rank in range(SEACACHE_NUM_SELECTED_PROMPTS):
        start = prompt_rank * SEACACHE_CANDIDATES_PER_PROMPT
        group = rows[start:start + SEACACHE_CANDIDATES_PER_PROMPT]
        if any(int(row.get("prompt_rank", -1)) != prompt_rank for row in group):
            raise ValueError("SeaCache prompt rows must be contiguous in selection order")
        if [int(row.get("candidate_index_for_prompt", -1)) for row in group] != [0, 1, 2]:
            raise ValueError("SeaCache candidate indices must be 0,1,2 inside each prompt")
        expected_seed = int(group[0].get("manifest_seed", -1)) + 5_000_003 * (
            prompt_rank + 1
        )
        expected = random.Random(expected_seed).sample(list(SEACACHE_THRESHOLDS), 3)
        observed = [float(row.get("fixed_threshold")) for row in group]
        if observed != expected or len(set(observed)) != 3:
            raise ValueError("SeaCache threshold sample is not reproducible/distinct")
        if any(int(row.get("threshold_selection_seed", -1)) != expected_seed for row in group):
            raise ValueError("SeaCache threshold-selection seed mismatch")
    for row in rows:
        if row.get("schema") != SCHEMA_SEACACHE:
            raise ValueError("SeaCache manifest schema mismatch")
        if row.get("policy_family") != "fixed_seacache_threshold":
            raise ValueError("SeaCache policy family mismatch")
        if row.get("calibration_status") != "not_applicable_fixed_threshold":
            raise ValueError("fixed SeaCache rows must not claim calibrated random thresholds")
        if row.get("target_speedup") is not None or row.get("q") is not None:
            raise ValueError("fixed SeaCache rows must not invent target speedup or q")
        threshold = float(row["fixed_threshold"])
        if threshold not in SEACACHE_THRESHOLDS:
            raise ValueError("fixed SeaCache threshold is outside the frozen Wan2.2 list")
        if row.get("threshold_grid") != list(SEACACHE_THRESHOLDS):
            raise ValueError("SeaCache row threshold grid mismatch")
        if not math.isclose(float(row["mean_threshold"]), threshold, abs_tol=0.0):
            raise ValueError("fixed SeaCache mean threshold mismatch")
        if row.get("threshold_path") != [threshold] * NUM_STEPS:
            raise ValueError("fixed SeaCache row must use one constant threshold for 50 steps")
        if row.get("protocol") != PROTOCOL:
            raise ValueError("SeaCache protocol mismatch")
        release_index = int(row["release_index"])
        if int(row["shard_index"]) != release_index % NUM_SHARDS:
            raise ValueError("SeaCache release-index shard derivation mismatch")
        if row["trajectory_id"] != f"{row['sample_id']}__sc{release_index:04d}":
            raise ValueError("SeaCache trajectory identity derivation mismatch")
        if row.get("forced_recompute_steps") != sorted(FORCED_RECOMPUTE_STEPS):
            raise ValueError("SeaCache forced recompute steps mismatch")


def manifest_contract(rows: list[dict[str, Any]]) -> dict[str, int | str | bool]:
    if not rows:
        raise ValueError("manifest is empty")
    schema = rows[0].get("schema")
    if schema in {SCHEMA_PLAN, SCHEMA_RUNNABLE}:
        return {
            "schema": str(schema),
            "selected_prompt_count": NUM_SELECTED_PROMPTS,
            "candidate_count": NUM_CANDIDATES,
            "baselines_per_shard": NUM_SELECTED_PROMPTS // NUM_SHARDS,
            "candidates_per_shard": NUM_CANDIDATES // NUM_SHARDS,
            "candidate_runnable": schema == SCHEMA_RUNNABLE,
        }
    if schema == SCHEMA_SEACACHE:
        return {
            "schema": str(schema),
            "selected_prompt_count": SEACACHE_NUM_SELECTED_PROMPTS,
            "candidate_count": SEACACHE_NUM_CANDIDATES,
            "baselines_per_shard": SEACACHE_NUM_SELECTED_PROMPTS // NUM_SHARDS,
            "candidates_per_shard": SEACACHE_NUM_CANDIDATES // NUM_SHARDS,
            "candidate_runnable": True,
        }
    raise ValueError("unknown manifest schema")


def validate_candidate_manifest(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("manifest is empty")
    if rows[0].get("schema") == SCHEMA_RUNNABLE:
        validate_runnable(rows)
    elif rows[0].get("schema") == SCHEMA_SEACACHE:
        validate_seacache_manifest(rows)
    else:
        raise ValueError("candidate collection requires a runnable candidate manifest")


def validate_plan(rows: list[dict[str, Any]]) -> None:
    if len(rows) != NUM_CANDIDATES:
        raise ValueError(f"expected {NUM_CANDIDATES} plan rows, got {len(rows)}")
    if [row.get("release_index") for row in rows] != list(range(NUM_CANDIDATES)):
        raise ValueError("release_index must be contiguous")
    if len({row.get("trajectory_id") for row in rows}) != NUM_CANDIDATES:
        raise ValueError("trajectory IDs are not unique")
    prompt_counts = Counter(str(row.get("sample_id")) for row in rows)
    if len(prompt_counts) != NUM_SELECTED_PROMPTS or set(prompt_counts.values()) != {3}:
        raise ValueError("plan must contain exactly three trajectories per prompt")
    if Counter(int(row["shard_index"]) for row in rows) != Counter({i: 2250 for i in range(4)}):
        raise ValueError("plan shards must be exactly 2250 each")
    if Counter(str(row["split"]) for row in rows) != Counter(train=7200, val=900, test=900):
        raise ValueError("candidate split must be 7200/900/900")
    if len({str(row.get("part") or "") for row in rows}) != 98:
        raise ValueError("selected prompts must retain all 98 OpenVidHD parts")
    for prompt_rank in range(NUM_SELECTED_PROMPTS):
        group = rows[
            prompt_rank * CANDIDATES_PER_PROMPT:(prompt_rank + 1) * CANDIDATES_PER_PROMPT
        ]
        if any(int(row.get("prompt_rank", -1)) != prompt_rank for row in group):
            raise ValueError("prompt rows must remain contiguous in selection order")
        if [int(row.get("candidate_index_for_prompt", -1)) for row in group] != [0, 1, 2]:
            raise ValueError("candidate indices must be 0,1,2 inside each prompt")
    for row in rows:
        if row.get("schema") != SCHEMA_PLAN or row.get("calibration_status") != "pending":
            raise ValueError("plan row schema/calibration mismatch")
        if row.get("mean_threshold") is not None or row.get("threshold_path") is not None:
            raise ValueError("pending plans must not contain thresholds")
        target = float(row["target_speedup"])
        q = float(row["q"])
        residual = row.get("normalized_residual_path")
        if not TARGET_SPEEDUP_MIN <= target <= TARGET_SPEEDUP_MAX:
            raise ValueError("target speedup outside uniform domain")
        if not Q_MIN <= q <= Q_MAX:
            raise ValueError("q outside uniform domain")
        if not isinstance(residual, list) or len(residual) != NUM_STEPS:
            raise ValueError("invalid normalized residual path")
        residual_mean = sum(float(value) for value in residual) / len(residual)
        if abs(residual_mean) > 1e-12 or abs(max(abs(float(v)) for v in residual) - 1.0) > 1e-12:
            raise ValueError("residual path is not all-step-mean-zero/peak-one")
        if row.get("protocol") != PROTOCOL:
            raise ValueError("plan protocol mismatch")
        release_index = int(row["release_index"])
        manifest_seed = int(row.get("manifest_seed", -1))
        expected_seeds = {
            "target_speedup_seed": manifest_seed + 3_000_001 * (release_index + 1),
            "q_seed": manifest_seed + 2_100_003 * (release_index + 1),
            "reference_point_count_seed": manifest_seed + 4_000_003 * (release_index + 1),
            "schedule_seed": manifest_seed + 1_000_003 * (release_index + 1),
        }
        if any(int(row.get(key, -1)) != value for key, value in expected_seeds.items()):
            raise ValueError("manifest seed derivation mismatch")
        if int(row["shard_index"]) != release_index % NUM_SHARDS:
            raise ValueError("release-index shard derivation mismatch")
        if row["trajectory_id"] != f"{row['sample_id']}__rt{release_index:04d}":
            raise ValueError("trajectory identity derivation mismatch")
        if row.get("forced_recompute_steps") != sorted(FORCED_RECOMPUTE_STEPS):
            raise ValueError("forced recompute steps mismatch")
        target_unit = float(row.get("target_speedup_random_unit", -1.0))
        q_unit = float(row.get("q_random_unit", -1.0))
        reference_unit = float(row.get("reference_point_count_random_unit", -1.0))
        if not math.isclose(
            target_unit,
            random.Random(expected_seeds["target_speedup_seed"]).random(),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("target seed does not reproduce random unit")
        if not math.isclose(
            q_unit,
            random.Random(expected_seeds["q_seed"]).random(),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("q seed does not reproduce random unit")
        if not math.isclose(
            reference_unit,
            random.Random(expected_seeds["reference_point_count_seed"]).random(),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("reference-count seed does not reproduce random unit")
        if not 0.0 <= target_unit < 1.0 or not math.isclose(
            target,
            TARGET_SPEEDUP_MIN + (TARGET_SPEEDUP_MAX - TARGET_SPEEDUP_MIN) * target_unit,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("target random unit/provenance mismatch")
        if not 0.0 <= q_unit < 1.0 or not math.isclose(
            q, Q_MIN + (Q_MAX - Q_MIN) * q_unit, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError("q random unit/provenance mismatch")
        expected_count = min(int(reference_unit * len(INTERIOR_COUNTS)), len(INTERIOR_COUNTS) - 1)
        if not 0.0 <= reference_unit < 1.0 or int(row["interior_reference_point_count"]) != expected_count:
            raise ValueError("reference-count random unit/provenance mismatch")
        expected_residual, expected_provenance = sample_residual_path(
            expected_seeds["schedule_seed"], expected_count
        )
        if residual != expected_residual or any(
            row.get(key) != value for key, value in expected_provenance.items()
        ):
            raise ValueError("schedule seed does not reproduce residual-path provenance")


def _load_calibration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != MAPPING_SCHEMA:
        raise ValueError("invalid speed-threshold mapping schema")
    if payload.get("calibration_status") != "calibrated":
        raise ValueError(
            "speed-threshold mapping is not calibrated; runnable candidate materialization is blocked"
        )
    if payload.get("model") != PROTOCOL["model"]:
        raise ValueError("calibration model does not match Wan2.1-T2V-1.3B")
    if payload.get("protocol") != {key: value for key, value in PROTOCOL.items() if key != "model"}:
        raise ValueError("calibration protocol does not match the frozen Wan2.1 protocol")
    if payload.get("target_speedup_domain") != [TARGET_SPEEDUP_MIN, TARGET_SPEEDUP_MAX]:
        raise ValueError("calibration target domain must be exactly [1.5,3.5]")
    fit_source_value = payload.get("fit_source")
    fit_source_hash = payload.get("fit_source_sha256")
    if not isinstance(fit_source_value, str) or not fit_source_value.strip():
        raise ValueError("calibrated mapping requires a concrete fit_source")
    fit_source = Path(fit_source_value).expanduser().resolve(strict=True)
    if not isinstance(fit_source_hash, str) or sha256(fit_source) != fit_source_hash:
        raise ValueError("calibration fit_source checksum mismatch")
    bounds = payload.get("threshold_bounds")
    mapping = payload.get("mapping")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("calibrated mapping requires threshold_bounds=[min,max]")
    lower, upper = map(float, bounds)
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0.0 < lower < upper:
        raise ValueError("threshold bounds must be finite and positive")
    if not isinstance(mapping, dict) or mapping.get("kind") != "monotone_piecewise_linear":
        raise ValueError("mapping.kind must be monotone_piecewise_linear")
    speeds = [float(v) for v in mapping.get("speedups", [])]
    thresholds = [float(v) for v in mapping.get("mean_thresholds", [])]
    if len(speeds) != len(thresholds) or len(speeds) < 2:
        raise ValueError("mapping knots must have matching length >=2")
    if any(b <= a for a, b in zip(speeds, speeds[1:])):
        raise ValueError("mapping speedups must strictly increase")
    if any(not math.isfinite(value) for value in (*speeds, *thresholds)):
        raise ValueError("mapping knots must be finite")
    if any(b < a for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError("mapping mean thresholds must be nondecreasing")
    if speeds[0] > TARGET_SPEEDUP_MIN or speeds[-1] < TARGET_SPEEDUP_MAX:
        raise ValueError("mapping knots must cover [1.5,3.5] without extrapolation")
    if any(not lower <= value <= upper for value in thresholds):
        raise ValueError("mapping thresholds lie outside declared bounds")
    return payload


def _interpolate(target: float, speeds: list[float], thresholds: list[float]) -> float:
    for index in range(len(speeds) - 1):
        if speeds[index] <= target <= speeds[index + 1]:
            unit = (target - speeds[index]) / (speeds[index + 1] - speeds[index])
            return thresholds[index] + unit * (thresholds[index + 1] - thresholds[index])
    raise ValueError(f"target {target} requires forbidden extrapolation")


def materialize_threshold_path(
    residual: list[float], center: float, q: float, lower: float, upper: float
) -> tuple[list[float], dict[str, Any]]:
    threshold_range = upper - lower
    up_room, down_room = upper - center, center - lower
    offsets = [q * (up_room * max(v, 0.0) + down_room * min(v, 0.0)) for v in residual]
    maximum = max(abs(offsets[i + 1] - offsets[i]) / threshold_range for i in range(NUM_STEPS - 1))
    adjacent_scale = 1.0 if maximum <= MAX_NORMALIZED_STEP_DELTA else MAX_NORMALIZED_STEP_DELTA / maximum
    raw = [center + adjacent_scale * value for value in offsets]
    clipped = [min(upper, max(lower, value)) for value in raw]
    mean_before = sum(clipped[s] for s in ELIGIBLE_STEPS) / len(ELIGIBLE_STEPS)
    requested_translation = center - mean_before
    translation_min = lower - min(clipped)
    translation_max = upper - max(clipped)
    applied_translation = min(translation_max, max(translation_min, requested_translation))
    translated = [value + applied_translation for value in clipped]
    translated_mean = sum(translated[s] for s in ELIGIBLE_STEPS) / len(ELIGIBLE_STEPS)
    nearest = min(center - lower, upper - center)
    tolerance = MEAN_TOLERANCE_FRACTION * nearest
    error = translated_mean - center
    mean_scale = 1.0 if abs(error) <= tolerance or abs(error) <= 1e-15 else tolerance / abs(error)
    final = [
        min(upper, max(lower, center + mean_scale * (value - center)))
        for value in translated
    ]
    final_mean = sum(final[s] for s in ELIGIBLE_STEPS) / len(ELIGIBLE_STEPS)
    final_delta = max(abs(final[i + 1] - final[i]) / threshold_range for i in range(NUM_STEPS - 1))
    if any(not lower - 1e-12 <= value <= upper + 1e-12 for value in final):
        raise AssertionError("materialized threshold escaped bounds")
    if final_delta > MAX_NORMALIZED_STEP_DELTA + 1e-12:
        raise AssertionError("materialized threshold violated adjacent cap")
    if abs(final_mean - center) > tolerance + 1e-12:
        raise AssertionError("materialized threshold violated mean tolerance")
    return final, {
        "threshold_min": lower,
        "threshold_max": upper,
        "max_normalized_step_delta": MAX_NORMALIZED_STEP_DELTA,
        "max_normalized_adjacent_delta_before_cap": maximum,
        "adjacent_cap_scale": adjacent_scale,
        "eligible_mean_before_translation": mean_before,
        "requested_translation": requested_translation,
        "applied_translation": applied_translation,
        "mean_tolerance": tolerance,
        "mean_scale": mean_scale,
        "eligible_mean_final": final_mean,
        "eligible_mean_final_error": final_mean - center,
        "max_normalized_adjacent_delta_final": final_delta,
    }


def materialize(plan_path: Path, calibration_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = read_jsonl(plan_path)
    validate_plan(plan)
    calibration = _load_calibration(calibration_path)
    bounds = [float(v) for v in calibration["threshold_bounds"]]
    speeds = [float(v) for v in calibration["mapping"]["speedups"]]
    thresholds = [float(v) for v in calibration["mapping"]["mean_thresholds"]]
    result: list[dict[str, Any]] = []
    for row in plan:
        center = _interpolate(float(row["target_speedup"]), speeds, thresholds)
        path, provenance = materialize_threshold_path(
            [float(v) for v in row["normalized_residual_path"]],
            center,
            float(row["q"]),
            bounds[0],
            bounds[1],
        )
        result.append({
            **row,
            "schema": SCHEMA_RUNNABLE,
            "calibration_status": "calibrated",
            "mean_threshold": center,
            "threshold_path": path,
            "calibration_file": str(calibration_path.resolve()),
            "calibration_sha256": sha256(calibration_path),
            "calibration_fit_source": calibration.get("fit_source"),
            "calibration_fit_source_sha256": calibration.get("fit_source_sha256"),
            **provenance,
        })
    validate_runnable(result)
    summary = {
        "schema": SCHEMA_RUNNABLE,
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256(plan_path),
        "calibration": str(calibration_path.resolve()),
        "calibration_sha256": sha256(calibration_path),
        "calibration_status": "calibrated",
        "candidate_runnable": True,
        "candidate_count": len(result),
        "selected_prompt_count": NUM_SELECTED_PROMPTS,
        "shard_counts": dict(Counter(row["shard_index"] for row in result)),
        "protocol": PROTOCOL,
    }
    return result, summary


def validate_runnable(rows: list[dict[str, Any]]) -> None:
    if len(rows) != NUM_CANDIDATES:
        raise ValueError("runnable manifest row count mismatch")
    pending_view = [
        {
            **row,
            "schema": SCHEMA_PLAN,
            "calibration_status": "pending",
            "mean_threshold": None,
            "threshold_path": None,
        }
        for row in rows
    ]
    validate_plan(pending_view)
    for row in rows:
        if row.get("schema") != SCHEMA_RUNNABLE or row.get("calibration_status") != "calibrated":
            raise ValueError("runnable schema/calibration mismatch")
        path = row.get("threshold_path")
        if not isinstance(path, list) or len(path) != NUM_STEPS:
            raise ValueError("runnable row lacks 50-step threshold path")
        lower, upper = float(row["threshold_min"]), float(row["threshold_max"])
        if any(not lower - 1e-12 <= float(value) <= upper + 1e-12 for value in path):
            raise ValueError("runnable threshold outside bounds")
        if not lower - 1e-12 <= float(row["mean_threshold"]) <= upper + 1e-12:
            raise ValueError("runnable mean threshold outside bounds")


def _write_bundle(output: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_jsonl(output, rows)
    write_csv(output.with_suffix(".csv"), rows)
    summary_path = output.with_name(output.stem + "_summary.json")
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--prompt-pool", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    plan_parser.add_argument(
        "--prompt-selection-seed", type=int, default=DEFAULT_PROMPT_SELECTION_SEED
    )
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--plan", type=Path, required=True)
    materialize_parser.add_argument("--calibration", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    seacache_parser = subparsers.add_parser("seacache-plan")
    seacache_parser.add_argument("--prompt-pool", type=Path, required=True)
    seacache_parser.add_argument("--threshold-config", type=Path, required=True)
    seacache_parser.add_argument("--output", type=Path, required=True)
    seacache_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    seacache_parser.add_argument(
        "--prompt-selection-seed", type=int, default=DEFAULT_PROMPT_SELECTION_SEED
    )
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        rows, summary = build_plan(
            args.prompt_pool.resolve(strict=True), args.seed, args.prompt_selection_seed
        )
        _write_bundle(args.output, rows, summary)
    elif args.command == "materialize":
        rows, summary = materialize(args.plan.resolve(strict=True), args.calibration.resolve(strict=True))
        _write_bundle(args.output, rows, summary)
    elif args.command == "seacache-plan":
        rows, summary = build_seacache_manifest(
            args.prompt_pool.resolve(strict=True),
            args.threshold_config.resolve(strict=True),
            args.seed,
            args.prompt_selection_seed,
        )
        _write_bundle(args.output, rows, summary)
    else:
        rows = read_jsonl(args.manifest.resolve(strict=True))
        if rows and rows[0].get("schema") == SCHEMA_PLAN:
            validate_plan(rows)
        elif rows and rows[0].get("schema") == SCHEMA_RUNNABLE:
            validate_runnable(rows)
        elif rows and rows[0].get("schema") == SCHEMA_SEACACHE:
            validate_seacache_manifest(rows)
        else:
            raise ValueError("unknown or empty manifest")
        print(json.dumps({"status": "ok", "rows": len(rows), "schema": rows[0]["schema"]}, indent=2))
        return
    print(json.dumps({"status": "ok", "output": str(args.output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
