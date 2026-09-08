"""One pass over candidate raw latents; all ten causal feature groups, no baseline."""
import argparse
import json
from pathlib import Path
import torch
from .contracts import create_result, dump, sha256, under, EXP_ROOT
from .latent_features import GROUPS, LatentFeatureHistory, contract
from .selection import selected_paths
from .data import load_completion, episode


def extract(records, actions, *, device='cpu', strict_shape=True):
    if len(records) != 50 or len(actions) != 50:
        raise ValueError('requires 50 input latents and executed actions')
    history = LatentFeatureHistory(GROUPS)
    values = {g: [] for g in GROUPS}
    provenance = []
    for step, record in enumerate(records):
        if record['step_index'] != step or record.get('model_stage') != 'single':
            raise ValueError('requires ordered single-stage Wan21 input latent records')
        path = Path(record['latent_path']).resolve(strict=True)
        latent = torch.load(path, map_location=device, weights_only=True)
        if (not isinstance(latent, torch.Tensor) or latent.dtype != torch.float16 or
                list(latent.shape) != record['latent_shape'] or
                record.get('latent_dtype') != 'torch.float16'):
            raise ValueError('candidate latent archive shape/dtype mismatch')
        if strict_shape and tuple(latent.shape) != (16,21,60,104):
            raise ValueError('requires Wan21 832x480/81-frame candidate input latent')
        features = history.observe(latent, step, record['sigma'])
        for group, value in features.items():
            values[group].append(value[0].cpu())
        history.commit(int(actions[step]))
        provenance.append(dict(step=step, path=str(path), sha256=sha256(path)))
    return {g: torch.stack(v) for g,v in values.items()}, provenance


def load_feature_index(root):
    root = Path(root)
    complete = json.loads((root/'COMPLETE.json').read_text())
    index_path = root/'index.json'
    if sha256(index_path) != complete['index_sha256']:
        raise ValueError('feature cache index hash mismatch')
    index = json.loads(index_path.read_text())
    if index['contracts'] != {g: contract(g) for g in GROUPS}:
        raise ValueError('feature definition mismatch')
    return index


def load_features(root, row, trace_path, actions, *, index=None):
    root = Path(root)
    index = load_feature_index(root) if index is None else index
    entry = index['rows'][row['trajectory_id']]
    path = root/entry['file']
    if not path.resolve().is_relative_to(root.resolve()) or sha256(path) != entry['sha256']:
        raise ValueError('feature trajectory hash/path mismatch')
    value = torch.load(path, map_location='cpu', weights_only=True)
    if (value['trajectory_id'] != row['trajectory_id'] or value['sample_id'] != row['sample_id'] or
            value['split'] != row['split'] or value['trace_sha256'] != sha256(trace_path) or
            not torch.equal(value['actions'], actions)):
        raise ValueError('feature cache is bound to another trajectory/trace/actions')
    for g, (dim, _) in GROUPS.items():
        x = value['features'][g]
        if x.shape != (50,dim) or not torch.isfinite(x).all() or x[0].any():
            raise ValueError('invalid feature tensor or initial history sentinel')
    return value['features']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection-root', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    paths = selected_paths(a.selection, a.collection_root)
    config = dict(schema='ours21_features_v1', selection_sha256=sha256(a.selection),
                  contracts={g:contract(g) for g in GROUPS})
    if a.resume:
        out = under(a.output_dir, EXP_ROOT)
        if json.loads((out/'run.json').read_text()) != config:
            raise ValueError('resume selection/feature contract changed')
        if (out/'COMPLETE.json').exists():
            raise FileExistsError('feature cache already complete')
    else:
        out = create_result(a.output_dir, '# Ten candidate feature cache\n\nrun.json freezes selection and definitions; rows/ stores small feature tensors with raw input hashes. index.json and COMPLETE.json finalize the cache. No raw latent copies or model weights. Resume interrupted extraction with --resume.')
        (out/'rows').mkdir()
        dump(out/'run.json', config)
    entries = {}
    for i, path in enumerate(paths):
        row, decisions, quality, sources = load_completion(path)
        actions = episode(decisions, quality, 'sea7')['action']
        target = out/'rows'/f'{i:04d}.pt'
        trace_hash = sha256(sources[1])
        if target.exists():
            value = torch.load(target, map_location='cpu', weights_only=True)
            if (value['trajectory_id'] != row['trajectory_id'] or value['trace_sha256'] != trace_hash or
                    not torch.equal(value['actions'], actions) or
                    any(sha256(r['path']) != r['sha256'] for r in value['raw_inputs'])):
                raise ValueError('resume source changed')
        else:
            trace = json.loads(sources[1].read_text())
            features, raw_inputs = extract(trace['step_records'], actions, device=a.device)
            value = dict(trajectory_id=row['trajectory_id'], sample_id=row['sample_id'],
                         split=row['split'], trace_sha256=trace_hash, actions=actions,
                         features=features, raw_inputs=raw_inputs)
            temporary = target.with_suffix('.tmp')
            torch.save(value, temporary)
            temporary.replace(target)
        entries[row['trajectory_id']] = dict(file=str(target.relative_to(out)),sha256=sha256(target))
        print(f'features {i+1}/{len(paths)} {row["trajectory_id"]}', flush=True)
    dump(out/'index.json', dict(**config, rows=entries))
    dump(out/'COMPLETE.json', dict(status='complete', trajectories=len(entries), index_sha256=sha256(out/'index.json')))
