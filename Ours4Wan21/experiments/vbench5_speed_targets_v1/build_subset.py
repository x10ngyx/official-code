#!/usr/bin/env python3
"""Freeze a deterministic maximum-dimension-coverage five-prompt VBench200 subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ours4wan21.contracts import dump, sha256

CUSTOM_DIMENSIONS = (
    'subject_consistency', 'background_consistency', 'motion_smoothness',
    'dynamic_degree', 'aesthetic_quality', 'imaging_quality',
    'temporal_flickering', 'human_action', 'temporal_style',
    'overall_consistency',
)


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 200 or len({row['sample_id'] for row in rows}) != 200:
        raise ValueError('source must be the frozen 200-row VBench200 prompts.jsonl')
    if any(not row.get('dimension') or not row.get('prompt_en') for row in rows):
        raise ValueError('VBench200 row lacks prompt or dimension metadata')
    return rows


def select_maximum_coverage(rows: list[dict], count: int = 5) -> tuple[list[dict], list[str]]:
    """Exact bitmask DP; ties use lexicographically earliest sample-id tuple."""
    dimensions = sorted({dimension for row in rows for dimension in row['dimension']})
    bit = {name: 1 << index for index, name in enumerate(dimensions)}
    masks = [sum(bit[name] for name in row['dimension']) for row in rows]
    # (number selected, union mask) -> tuple of source row indices.
    states: dict[tuple[int, int], tuple[int, ...]] = {(0, 0): ()}
    for index, row_mask in enumerate(masks):
        updated = dict(states)
        for (used, union), chosen in states.items():
            if used == count:
                continue
            key = (used + 1, union | row_mask)
            candidate = (*chosen, index)
            if key not in updated or tuple(rows[i]['sample_id'] for i in candidate) < tuple(
                    rows[i]['sample_id'] for i in updated[key]):
                updated[key] = candidate
        states = updated
    candidates = [(mask.bit_count(), tuple(rows[i]['sample_id'] for i in chosen), chosen, mask)
                  for (used, mask), chosen in states.items() if used == count]
    if not candidates:
        raise ValueError('cannot select requested prompt count')
    coverage, _, chosen, mask = min(candidates, key=lambda item: (-item[0], item[1]))
    selected = [rows[index] for index in chosen]
    covered = [name for name in dimensions if mask & bit[name]]
    if len(selected) != count or len(covered) != coverage:
        raise AssertionError('subset selection invariant failed')
    return selected, covered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    official = PROJECT.parent
    parser.add_argument('--source-prompts', type=Path,
                        default=official / 'Vbench200/prompts.jsonl')
    parser.add_argument('--source-full-info', type=Path,
                        default=official / 'Vbench200/VBench200_full_info.json')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'subset')
    args = parser.parse_args()
    prompts_path = args.source_prompts.resolve(strict=True)
    full_info_path = args.source_full_info.resolve(strict=True)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'refusing to overwrite frozen subset: {out}')
    rows = load_rows(prompts_path)
    selected, covered = select_maximum_coverage(rows)
    all_dimensions = sorted({name for row in rows for name in row['dimension']})
    omitted = sorted(set(all_dimensions) - set(covered))
    full_info = json.loads(full_info_path.read_text())
    selected_prompts = {row['prompt_en'] for row in selected}
    subset_info = [row for row in full_info if row.get('prompt_en') in selected_prompts]
    if {row['prompt_en'] for row in subset_info} != selected_prompts:
        raise ValueError('selected prompts do not map exactly into VBench200 full-info')

    out.mkdir(parents=True)
    prompt_text = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                          for row in selected)
    (out / 'prompts.jsonl').write_text(prompt_text)
    dump(out / 'prompt_map.json', {f"{row['sample_id']}.mp4": row['prompt_en'] for row in selected})
    dump(out / 'subset_full_info.json', subset_info)
    manifest = dict(
        schema='ours4wan21_vbench200_five_prompt_subset_v1',
        status='frozen', prompt_count=5,
        source=dict(prompts=str(prompts_path), prompts_sha256=sha256(prompts_path),
                    full_info=str(full_info_path), full_info_sha256=sha256(full_info_path)),
        selection=dict(
            objective='maximize the number of distinct VBench metadata dimensions covered by five records',
            algorithm='exact dynamic programming over dimension bitmasks',
            tie_break='lexicographically earliest ordered sample_id tuple',
            quality_or_runtime_blind=True),
        selected_sample_ids=[row['sample_id'] for row in selected],
        selected_source_unique_indices_1based=[row['source_unique_index_1based'] for row in selected],
        covered_metadata_dimensions=covered,
        omitted_metadata_dimensions=omitted,
        custom_input_dimensions=list(CUSTOM_DIMENSIONS),
        score_scope='official VBench custom-input implementations; unweighted mean of ten raw dimensions',
        official_full_vbench_score=False,
        warning=('Five prompts cannot cover all 16 VBench dimensions. This is a small matched '
                 'diagnostic subset, not an official full VBench or VBench200 score.'))
    dump(out / 'selection_manifest.json', manifest)
    checksum_names = ('prompts.jsonl', 'prompt_map.json', 'subset_full_info.json',
                      'selection_manifest.json')
    (out / 'SHA256SUMS').write_text(''.join(
        f'{sha256(out / name)}  {name}\n' for name in checksum_names))
    (out / 'README.md').write_text(
        '# Frozen VBench200 five-prompt diagnostic subset\n\n'
        'The manifest records the exact maximum-coverage selection rule, source hashes and score limitation.\n')
    print(json.dumps(dict(selected=manifest['selected_sample_ids'], covered=covered,
                          omitted=omitted), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
