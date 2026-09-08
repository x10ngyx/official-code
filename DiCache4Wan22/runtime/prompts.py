"""Validated local Vbench200 or custom prompt tables."""
import json
from pathlib import Path
from common import REPOSITORY, read_json

def load_prompts(path, mode):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [r["sample_id"] for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("prompts must have unique sample IDs")
    for row in rows:
        name = row["sample_id"]
        if not name or Path(name).name != name or name in {".", ".."} or not row["prompt_en"].strip():
            raise ValueError("invalid sample ID or empty prompt")
    if mode == "standard":
        official = {r["sample_id"]: r for r in
                    (json.loads(line) for line in (REPOSITORY / "Vbench200/prompts.jsonl").read_text().splitlines())}
        dimensions = set()
        for row in rows:
            ref = official.get(row["sample_id"])
            if ref is None or ref["prompt_en"] != row["prompt_en"]:
                raise ValueError("standard VBench requires exact official Vbench200 ID/prompt pairs")
            dimensions.update(ref["dimension"])
        required = set(read_json(REPOSITORY / "VbenchEvaluation/dimensions.json")["dimensions"])
        if not required <= dimensions:
            raise ValueError(f"standard subset does not cover all 16 dimensions: {sorted(required - dimensions)}")
    return rows

