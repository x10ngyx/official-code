"""Build scalar or feature-augmented transitions from completed collection traces.

Raw latents are opened only by prepare_features.py. One video is one 50-transition episode; CFG
branches must agree, as required by the local denoising-step policy contract.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .contracts import (FORCED, MODES, PROTOCOL, STEPS, create_result, dump,
                        observation, sha256, state_contract, state_names, latent_group)
from .local_iql import required_hard_budget_action


def episode(decisions, quality, mode):
    state_names(mode)
    if len(decisions) != 100 or not math.isfinite(quality):
        raise ValueError('requires 100 branch decisions and finite terminal RGB PSNR')
    actions, signals = [], []
    afters = {'cond': 0., 'uncond': 0.}
    for step in range(STEPS):
        pair = decisions[2 * step:2 * step + 2]
        values = []
        for offset, branch in enumerate(('cond', 'uncond')):
            row = pair[offset]
            if (row['step_index'], row['branch'], row['call_index']) != (step, branch, 2 * step + offset):
                raise ValueError('malformed CFG branch order')
            if row['action'] not in ('reuse', 'recompute') or row.get('execution') != row['action']:
                raise ValueError('missing or inconsistent executed action')
            action = int(row['action'] == 'reuse')
            if row.get('stored_feature') != 'sea_filtered':
                raise ValueError('raw or unspecified SEA boundary features are not accepted')
            before = float(row['accumulated_distance_before'])
            after = float(row['accumulated_distance_after'])
            if not math.isclose(before, afters[branch], abs_tol=2e-6, rel_tol=1e-6):
                raise ValueError('broken SEA history/reset')
            if step in FORCED:
                if action or after != 0.:
                    raise ValueError('native boundary must recompute and reset')
                d = total = 0.
            else:
                d = float(row['filtered_relative_l1'])
                total = float(row['accumulated_distance_with_current'])
                if not all(math.isfinite(x) and x >= 0 for x in (d, total, before, after)):
                    raise ValueError('invalid SEA distance')
                if not math.isclose(before + d, total, abs_tol=2e-6, rel_tol=1e-6):
                    raise ValueError('SEA accumulator must include current distance before action')
                if not math.isclose(after, total if action else 0., abs_tol=2e-6, rel_tol=1e-6):
                    raise ValueError('wrong post-action SEA reset')
            afters[branch] = after
            values.append((action, d, total))
        if values[0][0] != values[1][0] or any(
                not math.isclose(a, b, abs_tol=2e-6, rel_tol=1e-6)
                for a, b in zip(values[0][1:], values[1][1:])):
            raise ValueError('mixed CFG actions/signals cannot train a shared 50-step policy')
        actions.append(values[0][0])
        signals.append(values[0][1:])
    budget = sum(actions)
    states, masks = [], []
    used = consecutive = 0
    for step, (action, (d, total)) in enumerate(zip(actions, signals)):
        required, _ = required_hard_budget_action(step_index=step, used_skips=used,
            skip_budget=budget, num_steps=STEPS, forced_steps=FORCED)
        if required is not None and required != action:
            raise ValueError('recorded trajectory violates exact-K reachability')
        masks.append(float(required is None))
        states.append(observation(mode, step=step, budget=budget, used=used,
            consecutive=consecutive, cached_valid=step > 0, adjacent=d, accumulated=total))
        used += action
        consecutive = consecutive + 1 if action else 0
    states = torch.stack(states)
    # Matches the local trainer: terminal self-state, masked out by done=1.
    next_states = torch.cat((states[1:], states[-1:]))
    rewards, dones = torch.zeros(STEPS), torch.zeros(STEPS)
    rewards[-1], dones[-1] = quality, 1.
    return dict(state=states, next_state=next_states, action=torch.tensor(actions),
                reward=rewards, done=dones, actor_mask=torch.tensor(masks))


def load_completion(path):
    path = Path(path).resolve(strict=True)
    complete = json.loads(path.read_text())
    if complete.get('schema') != 'ours4wan21_candidate_complete_v3':
        raise ValueError('only completed Ours4Wan21 collection v3 candidates are accepted')
    row = complete['trajectory_row']
    trace_path = Path(row['trace_json']).resolve(strict=True)
    metrics_path = Path(row['video_metrics_json']).resolve(strict=True)
    trace = json.loads(trace_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    expected = dict(task='t2v-1.3B', sampling_steps=50, sample_solver='unipc',
                    shift=5., guide_scale=5., frame_num=81, size_wh=[832, 480])
    if any(trace.get(k) != v for k, v in expected.items()):
        raise ValueError('trace does not match frozen Wan21 sampling protocol')
    if trace['trajectory_id'] != row['trajectory_id'] or complete['trajectory_id'] != row['trajectory_id']:
        raise ValueError('trajectory identity mismatch')
    record = trace['manifest_record']
    if any(record[k] != row[k] for k in ('sample_id', 'split', 'trajectory_id')):
        raise ValueError('trace/manifest identity or split mismatch')
    collection_protocol = dict(model='Wan2.1-T2V-1.3B', task='t2v-1.3B',
        size_wh=[832, 480], frame_num=81, fps=16, sample_steps=50,
        sample_solver='unipc', sample_shift=5., cfg=5., seed=42,
        parameter_dtype='bfloat16', offload_model=False, t5_cpu=False)
    if record.get('protocol') != collection_protocol or row.get('protocol') != collection_protocol:
        raise ValueError('collection must use the frozen resident seed42 protocol')
    if (metrics.get('protocol_id'), metrics.get('frames'), metrics.get('width'), metrics.get('height')) != (
            'rgb_full_reference_v1', 81, 832, 480):
        raise ValueError('requires shared VideoMetrics RGB PSNR on all 81 frames')
    quality = float(metrics['metrics']['psnr_rgb_db']['mean'])
    if not math.isfinite(quality) or not math.isclose(quality, float(row['mean_psnr']), abs_tol=1e-6):
        raise ValueError('terminal PSNR disagrees with completed collection')
    for field, metric_field in (('candidate_video', 'candidate'), ('baseline_video', 'reference')):
        if Path(row[field]).resolve() != Path(metrics[metric_field]).resolve():
            raise ValueError('metric pair does not match trajectory videos')
    return row, trace['decisions'], quality, [path, trace_path, metrics_path]


def build(completions, mode, feature_cache=None):
    state_names(mode)
    group = latent_group(mode)
    if bool(group) != (feature_cache is not None):
        raise ValueError("feature modes require --feature-cache; scalar controls do not use it")
    if group:
        from .feature_cache import load_feature_index, load_features
        feature_index = load_feature_index(feature_cache)
    episodes, splits, ids, samples, sources = [], [], set(), {}, []
    for path in completions:
        row, decisions, quality, paths = load_completion(path)
        tid, sid = row['trajectory_id'], row['sample_id']
        split = 'evaluation' if row['split'] == 'val' else row['split']
        if tid in ids or split not in ('train', 'evaluation', 'test'):
            raise ValueError('duplicate trajectory or unsupported manifest split')
        if sid in samples and samples[sid] != split:
            raise ValueError('prompt leakage across train/evaluation')
        ids.add(tid)
        samples[sid] = split
        item = episode(decisions, quality, 'sea7' if group else mode)
        if group:
            features = load_features(feature_cache, row, paths[1], item['action'], index=feature_index)[group]
            item['state'] = torch.cat((features, item['state']), dim=1)
            item['next_state'] = torch.cat((item['state'][1:], item['state'][-1:]))
        episodes.append(item)
        splits.extend([split] * STEPS)
        sources.append(dict(trajectory_id=tid, sample_id=sid, split=split, source_split=row['split'],
                            files={str(p): sha256(p) for p in paths}))
    if not episodes or not {'train', 'evaluation'}.issubset(set(splits)):
        raise ValueError('requires nonempty registered train and evaluation splits')
    tensors = {key: torch.cat([e[key] for e in episodes]) for key in episodes[0]}
    train = torch.tensor([i for i, s in enumerate(splits) if s == 'train'])
    val = torch.tensor([i for i, s in enumerate(splits) if s == 'evaluation'])
    if any(not tensors['actor_mask'][idx].any() for idx in (train, val)):
        raise ValueError('both splits need discretionary actor decisions')
    test = torch.tensor([i for i, s in enumerate(splits) if s == 'test'], dtype=torch.long)
    return dict(tensors=tensors, train_indices=train, val_indices=val, test_indices=test,
                manifest=dict(schema='ours4wan21_training_data_v1', state=state_contract(mode),
                    protocol=PROTOCOL, trajectories=len(episodes), transitions=len(splits),
                    quality_mode='absolute', reward_timing='terminal_only',
                    skip_budget_source='realized logged reuse count per trajectory',
                    sources=sources, prompt_splits=samples,
                    **({'latent_feature_cache': dict(index_sha256=sha256(Path(feature_cache)/'index.json'),
                         contracts=feature_index['contracts'])} if group else {})))


class Transitions(Dataset):
    def __init__(self, tensors, indices):
        self.tensors, self.indices = tensors, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return {k: v[self.indices[index]] for k, v in self.tensors.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-root', type=Path, required=True)
    parser.add_argument('--state-mode', choices=MODES, required=True)
    parser.add_argument('--selection', type=Path, required=True,
                        help='selection.json from select_data.py; frozen 3000 random trajectories')
    parser.add_argument('--feature-cache', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    from .selection import selected_paths
    paths = selected_paths(args.selection, args.collection_root)
    if args.feature_cache:
        from .feature_cache import load_feature_index
        if load_feature_index(args.feature_cache)['selection_sha256'] != sha256(args.selection):
            raise ValueError('feature cache must use the same frozen 3000 selection')
    bundle = build(paths, args.state_mode, args.feature_cache)
    bundle['manifest']['selection_sha256'] = sha256(args.selection)
    bundle['manifest']['selection'] = json.loads(args.selection.read_text())
    out = create_result(args.output_dir, '# Scalar RL training data\n\ntransitions.pt contains small scalar tensors and registered splits; manifest.json records source hashes. No latents or model weights.')
    torch.save(bundle, out / 'transitions.pt')
    dump(out / 'manifest.json', bundle['manifest'])
    dump(out / 'COMPLETE.json', dict(status='complete', sha256=sha256(out / 'transitions.pt')))


if __name__ == '__main__':
    main()
