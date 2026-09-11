#!/usr/bin/env python3
"""Compute archive-level custom-input VBench scores for baseline/candidate sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .collector import (
    all_baselines_complete,
    baseline_paths,
    resolve_prompt_limit,
    unique_prompt_rows,
)
from .manifest import manifest_contract, read_jsonl, validate_candidate_manifest
from .paths import require_result_path
from .publisher import candidate_paths, load_completion


THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def ensure_link(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    if target.is_symlink():
        if target.resolve(strict=True) != source:
            raise ValueError(f"staging link target mismatch: {target}")
        return
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(Path(os.path.relpath(source, target.parent.resolve())))


def read_score(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("protocol") != "vbench_custom_input_raw_mean_v1"
        or payload.get("official_full_vbench_score") is not False
        or not isinstance(payload.get("vbench_score"), (int, float))
    ):
        raise ValueError(f"invalid custom VBench score: {path}")
    return payload


def quality_root_for(parent: Path, candidate_count: int, full_candidate_count: int) -> Path:
    if candidate_count == full_candidate_count:
        return parent / "quality"
    return parent / "quality" / "stages" / f"prefix_{candidate_count:09d}"


def evaluate(
    *,
    manifest: Path,
    parent: Path,
    runner: Path,
    vbench_python: Path,
    vbench_cache: Path,
    prompt_limit: int | None = None,
) -> dict[str, Any]:
    manifest = manifest.expanduser().resolve(strict=True)
    parent = require_result_path(parent)
    runner = runner.expanduser().resolve(strict=True)
    vbench_python = vbench_python.expanduser().resolve(strict=True)
    vbench_cache = vbench_cache.expanduser().resolve(strict=True)
    rows = read_jsonl(manifest)
    validate_candidate_manifest(rows)
    contract = manifest_contract(rows)
    full_prompt_count = int(contract["selected_prompt_count"])
    full_candidate_count = int(contract["candidate_count"])
    limit = resolve_prompt_limit(rows, prompt_limit)
    candidates_per_prompt = full_candidate_count // full_prompt_count
    expected_candidate_count = limit * candidates_per_prompt
    target_rows = rows[:expected_candidate_count]
    ready, missing = all_baselines_complete(rows, parent, limit)
    if not ready:
        raise RuntimeError(f"VBench requires all baselines; missing={len(missing)}")
    if any(load_completion(parent, row) is None for row in target_rows):
        raise RuntimeError(
            f"VBench requires all {expected_candidate_count} completed candidates"
        )

    quality_root = quality_root_for(parent, expected_candidate_count, full_candidate_count)
    comparison_path = quality_root / "vbench_summary.json"
    if comparison_path.is_file():
        payload = json.loads(comparison_path.read_text(encoding="utf-8"))
        read_score(Path(payload["baseline"]["path"]))
        read_score(Path(payload["candidate"]["path"]))
        return payload

    staging = quality_root / "vbench_staging"
    baseline_map: dict[str, str] = {}
    for row in unique_prompt_rows(rows)[:limit]:
        name = f"baseline_{row['sample_id']}.mp4"
        ensure_link(
            baseline_paths(parent, str(row["sample_id"]))["video"],
            staging / "baseline" / name,
        )
        baseline_map[name] = str(row["prompt"])
    candidate_map: dict[str, str] = {}
    for row in target_rows:
        name = f"candidate_{row['trajectory_id']}.mp4"
        ensure_link(
            candidate_paths(parent, int(row["shard_index"]), str(row["trajectory_id"]))[
                "video"
            ],
            staging / "candidate" / name,
        )
        candidate_map[name] = str(row["prompt"])
    baseline_map_path = quality_root / "baseline_prompt_map.json"
    candidate_map_path = quality_root / "candidate_prompt_map.json"
    if not baseline_map_path.exists():
        atomic_json(baseline_map_path, baseline_map)
    elif json.loads(baseline_map_path.read_text(encoding="utf-8")) != baseline_map:
        raise ValueError("existing baseline VBench prompt map differs")
    if not candidate_map_path.exists():
        atomic_json(candidate_map_path, candidate_map)
    elif json.loads(candidate_map_path.read_text(encoding="utf-8")) != candidate_map:
        raise ValueError("existing candidate VBench prompt map differs")

    environment = {**os.environ, **THREAD_ENV}
    environment["PYTHON_BIN"] = str(vbench_python)
    environment["VBENCH_CACHE_DIR"] = str(vbench_cache)
    for condition, prompt_map in (
        ("baseline", baseline_map_path),
        ("candidate", candidate_map_path),
    ):
        work = quality_root / f"vbench_{condition}"
        score_path = work / "vbench_custom_aggregate_scores.json"
        if not score_path.is_file():
            subprocess.run(
                [
                    "bash",
                    str(runner),
                    str(staging / condition),
                    str(work),
                    str(prompt_map),
                ],
                check=True,
                env=environment,
            )

    baseline_score_path = quality_root / "vbench_baseline" / "vbench_custom_aggregate_scores.json"
    candidate_score_path = quality_root / "vbench_candidate" / "vbench_custom_aggregate_scores.json"
    baseline_score = read_score(baseline_score_path)
    candidate_score = read_score(candidate_score_path)
    payload = {
        "schema": "ours4wan21_vbench_summary_v1",
        "protocol": "vbench_custom_input_raw_mean_v1",
        "warning": baseline_score["warning"],
        "baseline_video_count": len(baseline_map),
        "candidate_video_count": len(candidate_map),
        "prompt_limit": limit,
        "manifest_prompt_count": full_prompt_count,
        "manifest_candidate_count": full_candidate_count,
        "baseline": {
            "vbench_score": baseline_score["vbench_score"],
            "path": str(baseline_score_path.resolve()),
            "sha256": sha256(baseline_score_path),
        },
        "candidate": {
            "vbench_score": candidate_score["vbench_score"],
            "path": str(candidate_score_path.resolve()),
            "sha256": sha256(candidate_score_path),
        },
        "candidate_minus_baseline": (
            float(candidate_score["vbench_score"]) - float(baseline_score["vbench_score"])
        ),
    }
    atomic_json(comparison_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--vbench-python", type=Path, required=True)
    parser.add_argument("--vbench-cache", type=Path, required=True)
    parser.add_argument("--prompt-limit", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                manifest=args.manifest,
                parent=args.parent_root,
                runner=args.runner,
                vbench_python=args.vbench_python,
                vbench_cache=args.vbench_cache,
                prompt_limit=args.prompt_limit,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
