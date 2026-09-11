"""Train the local 400-epoch Exact-K/terminal-PSNR IQL configuration."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

from .contracts import (FORCED, MODEL_ROOT, MODES, PROJECT, PROTOCOL, TrainingConfig,
                        IQL_PROFILES, training_config, create_result, dump, observation, latent_group,
                        sha256, state_contract, state_names, under)
from .data import Transitions
from .local_iql import (IQLModelConfig, PolicyNet, QNet, ValueNet, _run_epoch,
                        apply_normalizer, compute_normalizer, required_hard_budget_action)


def validate_bundle(bundle, mode):
    manifest = bundle['manifest']
    if (manifest.get('schema') != 'ours4wan21_training_data_v1' or
            manifest.get('state') != state_contract(mode) or
            manifest.get('protocol') != PROTOCOL or
            manifest.get('quality_mode') != 'absolute' or
            manifest.get('reward_timing') != 'terminal_only'):
        raise ValueError('dataset contract mismatch')
    tensors = bundle['tensors']
    n, dim = tensors['state'].shape
    if n == 0 or n % 50 or dim != len(state_names(mode)):
        raise ValueError('invalid episode/state shape')
    if set(tensors) != {'state', 'next_state', 'action', 'reward', 'done', 'actor_mask'}:
        raise ValueError('unexpected transition fields')
    for key, value in tensors.items():
        expected = (n, dim) if key in ('state', 'next_state') else (n,)
        if tuple(value.shape) != expected or not torch.isfinite(value).all():
            raise ValueError(f'malformed or nonfinite tensor: {key}')
    for key in ('action', 'done', 'actor_mask'):
        if not ((tensors[key] == 0) | (tensors[key] == 1)).all():
            raise ValueError(f'{key} must be binary')
    expected_done = torch.zeros(n)
    expected_done[49::50] = 1
    if not torch.equal(tensors['done'], expected_done) or tensors['reward'][expected_done == 0].any():
        raise ValueError('rewards/dones must be terminal only')
    for start in range(0, n, 50):
        states = tensors['state'][start:start+50]
        actions = tensors['action'][start:start+50]
        budget = int(actions.sum())
        used = consecutive = 0
        previous_accumulated = 0.
        for step in range(50):
            scalar = states[step, -7:] if mode != 'scalar5' else states[step]
            d, total = (map(float, scalar[:2]) if mode != 'scalar5' else (0., 0.))
            expected = observation(mode, step=step, budget=budget, used=used,
                consecutive=consecutive, cached_valid=step > 0, adjacent=d, accumulated=total,
                latent=states[step, :-7] if latent_group(mode) else None)
            if not torch.allclose(states[step], expected, atol=1e-7, rtol=1e-6):
                raise ValueError('pre-action state counters or boundary sentinels disagree')
            if mode != 'scalar5' and step not in FORCED and abs(total - previous_accumulated - d) > 2e-6 * max(1., total):
                raise ValueError('cached SEA accumulation is inconsistent')
            required, _ = required_hard_budget_action(step_index=step, used_skips=used,
                skip_budget=budget, num_steps=50, forced_steps=FORCED)
            action = int(actions[step])
            if required is not None and action != required:
                raise ValueError('cached actions violate exact-K')
            if float(tensors['actor_mask'][start+step]) != float(required is None):
                raise ValueError('cached actor mask disagrees with exact-K constraints')
            used += action
            consecutive = consecutive + 1 if action else 0
            previous_accumulated = total if action else 0.
        if not torch.equal(tensors['next_state'][start:start+50], torch.cat((states[1:], states[-1:]))):
            raise ValueError('next_state does not match within-episode successor')
    train, val = bundle['train_indices'], bundle['val_indices']
    test = bundle.get('test_indices', torch.tensor([], dtype=torch.long))
    joined = torch.cat((train, val, test))
    if joined.dtype != torch.long or not torch.equal(joined.sort().values, torch.arange(n)):
        raise ValueError('split indices must partition transitions exactly')
    sources = manifest['sources']
    if len(sources) * 50 != n or len({s['trajectory_id'] for s in sources}) != len(sources):
        raise ValueError('source trajectory coverage mismatch')
    train_set = set(train.tolist())
    val_set = set(val.tolist())
    prompt_splits = {}
    for i, source in enumerate(sources):
        split = source['split']
        sid = source['sample_id']
        if split not in ('train', 'evaluation', 'test') or prompt_splits.get(sid, split) != split:
            raise ValueError('source prompt leakage')
        prompt_splits[sid] = split
        if any((j in train_set) != (split == 'train') for j in range(i * 50, (i + 1) * 50)):
            raise ValueError('split is not the registered full-trajectory split')
        if any((j in val_set) != (split == 'evaluation') for j in range(i * 50, (i + 1) * 50)):
            raise ValueError('test rows must not enter validation/model selection')
    if manifest['prompt_splits'] != prompt_splits:
        raise ValueError('prompt split manifest mismatch')
    for indices in (train, val):
        if not len(indices) or not tensors['actor_mask'][indices].any():
            raise ValueError('split has no discretionary actor decisions')


def make_networks(mode, config, device):
    mc = IQLModelConfig(len(state_names(mode)), config.hidden_dim, config.num_layers, config.dropout)
    nets = {key: cls(mc).to(device) for key, cls in (
        ('value_net', ValueNet), ('q1_net', QNet), ('q2_net', QNet),
        ('target_q1', QNet), ('target_q2', QNet), ('policy_net', PolicyNet))}
    for name in ('q1', 'q2'):
        nets['target_' + name].load_state_dict(nets[name + '_net'].state_dict())
    return mc, nets


def make_optimizers(nets, config):
    return tuple(torch.optim.AdamW(p, lr=config.lr, weight_decay=config.weight_decay) for p in (
        nets['value_net'].parameters(),
        list(nets['q1_net'].parameters()) + list(nets['q2_net'].parameters()),
        nets['policy_net'].parameters()))


def train(bundle, mode, out, weights, *, config=TrainingConfig(), device='cuda', smoke=False,
          iql_profile='baseline'):
    """Production settings must match the explicitly selected versioned IQL profile."""
    validate_bundle(bundle, mode)
    expected = training_config(iql_profile, seed=config.seed)
    if not smoke and config != expected:
        raise ValueError('production settings differ from the explicit IQL profile')
    selection = bundle['manifest'].get('selection', {})
    mixed = selection.get('schema') == 'ours4wan21_mixed_subset_v1'
    expected_count = 3500 if mixed else 3000
    if not smoke and (bundle['manifest'].get('trajectories') != expected_count or
                      selection.get('selected_count') != expected_count):
        raise ValueError('production requires frozen random3000 or explicit random3000+increase500')
    if not smoke and mixed and (selection.get('family_counts') != {
            'random_continuous_seacache_threshold':3000, 'linear_increase_seacache_threshold':500}
            or selection.get('split_counts') != {'train':2800, 'val':350, 'test':350}):
        raise ValueError('invalid mixed dataset composition')
    if config.epochs < 1:
        raise ValueError('epochs must be positive')
    weights = under(weights, MODEL_ROOT)
    if weights.exists():
        raise FileExistsError(weights)
    out = create_result(out, '# Ours4Wan21 IQL training\n\nconfig.json and dataset_manifest.json freeze the run; epoch_metrics.jsonl records complete train/validation passes. model_weights links to models/. TRAINING_COMPLETE.json is written only after success.')
    weights.mkdir(parents=True)
    (weights / 'README.md').write_text('# Ours4Wan21 policy and critics\n\ncheckpoints/ contains every epoch and optimizer/RNG states. best_model.pt minimizes validation actor loss; final_model.pt is the last epoch. See the linked result for configuration.\n')
    (out / 'model_weights').symlink_to(weights, target_is_directory=True)
    (weights / 'training_result').symlink_to(out, target_is_directory=True)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = torch.device(device)
    tensors = dict(bundle['tensors'])
    train_idx, val_idx = bundle['train_indices'], bundle['val_indices']
    normalizer = compute_normalizer(tensors['state'], train_idx.tolist())
    for key in ('state', 'next_state'):
        tensors[key] = apply_normalizer(tensors[key], normalizer)
    loaders = {name: DataLoader(Transitions(tensors, idx), batch_size=config.batch_size,
                    shuffle=name == 'train', num_workers=0, pin_memory=device.type == 'cuda')
               for name, idx in (('train', train_idx), ('val', val_idx))}
    mc, nets = make_networks(mode, config, device)
    optimizers = make_optimizers(nets, config)
    dump(out / 'config.json', dict(**asdict(config), device=str(device), state=state_contract(mode),
        smoke_only=smoke, iql_profile=iql_profile, objective='exact-K terminal absolute RGB PSNR',
        best_checkpoint_metric='validation pi_loss on discretionary rows',
        local_training_lock=json.loads((PROJECT / 'local_training_lock.json').read_text())))
    dump(out / 'dataset_manifest.json', bundle['manifest'])
    dump(out / 'split.json', dict(train_count=len(train_idx), val_count=len(val_idx),
                                  test_count=len(bundle.get('test_indices', [])),
                                  prompt_splits=bundle['manifest']['prompt_splits']))
    started, best, best_epoch = time.perf_counter(), float('inf'), None
    try:
        for epoch in range(1, config.epochs + 1):
            row = dict(epoch=epoch)
            for split in ('train', 'val'):
                row[split] = _run_epoch(loader=loaders[split], **nets, args=config,
                    device=device, optimizers=optimizers if split == 'train' else None)
            if not all(torch.isfinite(torch.tensor(v)) for s in ('train', 'val') for v in row[s].values()):
                raise RuntimeError('nonfinite training/validation metrics')
            row['elapsed_seconds'] = time.perf_counter() - started
            with (out / 'epoch_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            payload = {key: net.state_dict() for key, net in nets.items()}
            payload.update(schema='ours4wan21_iql_checkpoint_v1', epoch=epoch, model_config=asdict(mc),
                train_config=asdict(config), iql_profile=iql_profile, state=state_contract(mode), protocol=PROTOCOL,
                normalizer=normalizer, metrics=row, smoke_only=smoke,
                optimizer_states=[o.state_dict() for o in optimizers],
                torch_rng_state=torch.get_rng_state(),
                cuda_rng_states=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                dataset_manifest=bundle['manifest'])
            path = weights / 'checkpoints' / f'epoch_{epoch:03d}.pt'
            path.parent.mkdir(exist_ok=True)
            torch.save(payload, path)
            if row['val']['pi_loss'] < best:
                best, best_epoch = row['val']['pi_loss'], epoch
                link = weights / 'best_model.pt'
                if link.is_symlink():
                    link.unlink()
                link.symlink_to(path.relative_to(weights))
            print(json.dumps(row), flush=True)
        (weights / 'final_model.pt').symlink_to(path.relative_to(weights))
        dump(out / 'TRAINING_COMPLETE.json', dict(status='complete', epochs=config.epochs,
            best_epoch=best_epoch, smoke_only=smoke, final_sha256=sha256(path)))
    except BaseException as exc:
        dump(out / 'FAILED.json', dict(error=repr(exc)))
        raise
    return weights / 'final_model.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='directory produced by prepare_data.py')
    parser.add_argument('--state-mode', choices=MODES, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--training-seed', type=int, default=42,
                        help='training RNG seed only; frozen video generation seed remains 42')
    parser.add_argument('--iql-profile', choices=IQL_PROFILES, default='baseline',
                        help='explicit IQL preset; aggressive_v1 changes only tau/beta/weight_max')
    args = parser.parse_args()
    state_names(args.state_mode)
    if 'wan2.2' not in Path(sys.prefix).name.lower():
        raise ValueError('use conda environment wan2.2')
    complete = json.loads((args.dataset / 'COMPLETE.json').read_text())
    data_path = args.dataset / 'transitions.pt'
    if complete['status'] != 'complete' or sha256(data_path) != complete['sha256']:
        raise ValueError('dataset completion/hash mismatch')
    bundle = torch.load(data_path, map_location='cpu', weights_only=False)
    if args.device == 'cuda' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError('select one GPU with CUDA_VISIBLE_DEVICES')
    train(bundle, args.state_mode, args.output_dir, args.checkpoint_dir,
          config=training_config(args.iql_profile,seed=args.training_seed), device=args.device,
          iql_profile=args.iql_profile)


if __name__ == '__main__':
    main()
