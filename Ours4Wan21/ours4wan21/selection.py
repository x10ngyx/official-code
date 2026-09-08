"""Freeze the existing 3000 completed random trajectories before cache building."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from .contracts import create_result, dump, sha256


def select_rows(rows, *, count=3000, strategy, seed=42):
    if count <= 0 or len(rows) < count:
        raise ValueError('not enough random trajectories for requested subset')
    rows = sorted(rows, key=lambda r: r['trajectory_id'])
    if len({r['trajectory_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate source trajectory')
    groups = defaultdict(list)
    for row in rows:
        if row['policy_family'] != 'random_continuous_seacache_threshold':
            raise ValueError('only randomized-path trajectories are eligible, not fixed SeaCache anchors')
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('requires the collection registered train/val/test split')
        groups[row['sample_id']].append(row)
    if any(len({r['split'] for r in group}) != 1 for group in groups.values()):
        raise ValueError('source prompt split leakage')
    rng = random.Random(seed)
    if strategy == 'all-completed':
        if len(rows) != count:
            raise ValueError(f'all-completed requires exactly {count} completed random trajectories; found {len(rows)}')
        selected = rows
    elif strategy == 'one-per-prompt':
        if len(groups) != count:
            raise ValueError('one-per-prompt requires exactly count distinct source prompts')
        selected = [rng.choice(groups[sid]) for sid in sorted(groups)]
    elif strategy == 'uniform-trajectories':
        selected = rng.sample(rows, count)
    else:
        raise ValueError('unknown sampling strategy')
    return sorted(selected, key=lambda r: r['trajectory_id'])


def completed_rows(root, rows, *, require_all=False):
    """Read only completed members of the original plan; never hide corrupt markers."""
    if not rows or len({r['trajectory_id'] for r in rows}) != len(rows):
        raise ValueError('empty manifest or duplicate source trajectory')
    paths, available = {}, []
    for row in rows:
        path = root / 'shards' / f"shard_{row['shard_index']:02d}" / 'candidates' / row['trajectory_id'] / 'CANDIDATE_COMPLETE.json'
        if not path.is_file():
            if require_all:
                raise ValueError('sampling a larger collection requires all source completions')
            continue
        complete = json.loads(path.read_text())
        if complete.get('schema') != 'ours4wan21_candidate_complete_v3':
            raise ValueError('invalid candidate completion marker')
        recorded = complete['trajectory_row']
        if any(recorded[k] != row[k] for k in ('trajectory_id','sample_id','split','policy_family','protocol')):
            raise ValueError('completion differs from immutable source manifest')
        paths[row['trajectory_id']] = path
        available.append(row)
    actual = set(root.glob('shards/shard_*/candidates/*/CANDIDATE_COMPLETE.json'))
    if actual != set(paths.values()):
        raise ValueError('completion markers exist outside their registered manifest paths')
    return available, paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection-root', type=Path, required=True)
    p.add_argument('--strategy', choices=('all-completed', 'one-per-prompt', 'uniform-trajectories'), default='all-completed')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    root = a.collection_root.resolve(strict=True)
    manifest_path = root / 'manifests/random_runnable.jsonl'
    rows = [json.loads(s) for s in manifest_path.read_text().splitlines() if s.strip()]
    available, paths = completed_rows(root, rows, require_all=a.strategy != 'all-completed')
    selected = select_rows(available, strategy=a.strategy)
    out = create_result(a.output_dir, '# Frozen random training dataset\n\nselection.json records the existing 3000 completed random trajectory IDs, original splits, relative completion paths and source SHA. all-completed keeps all of them without sampling; optional sampling modes use seed42. Both state modes must use this same selection.')
    dump(out / 'selection.json', dict(schema='ours4wan21_random_subset_v1',
        seed=None if a.strategy == 'all-completed' else 42,
        strategy=a.strategy, source_count=len(rows), completed_count=len(available), selected_count=len(selected),
        source_manifest_sha256=sha256(manifest_path),
        split_counts=dict(Counter(r['split'] for r in selected)),
        rows=[dict(trajectory_id=r['trajectory_id'], sample_id=r['sample_id'], split=r['split'],
            complete_path=str(paths[r['trajectory_id']].relative_to(root)),
            complete_sha256=sha256(paths[r['trajectory_id']])) for r in selected]))


def selected_paths(selection, root):
    value = json.loads(Path(selection).read_text())
    if value.get('schema') != 'ours4wan21_random_subset_v1' or value['selected_count'] != 3000 or len(value['rows']) != 3000:
        raise ValueError('expected frozen 3000-trajectory selection')
    root = Path(root).resolve(strict=True)
    if sha256(root / 'manifests/random_runnable.jsonl') != value['source_manifest_sha256']:
        raise ValueError('selection source manifest changed')
    paths, seen = [], set()
    for row in value['rows']:
        path = (root / row['complete_path']).resolve(strict=True)
        if not path.is_relative_to(root) or path in seen or sha256(path) != row['complete_sha256']:
            raise ValueError('selected completion path/hash/uniqueness mismatch')
        original = json.loads(path.read_text())['trajectory_row']
        if any(original[k] != row[k] for k in ('trajectory_id','sample_id','split')):
            raise ValueError('selected identity/split mismatch')
        if original['policy_family'] != 'random_continuous_seacache_threshold':
            raise ValueError('selected trajectory is not random')
        paths.append(path)
        seen.add(path)
    return paths


if __name__ == '__main__':
    main()
