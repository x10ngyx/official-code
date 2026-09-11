"""Extract compact features once; build matching train bundles for all ablations."""
import argparse
import json
from pathlib import Path
import sys
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ours4wan21.bloc_features import CompactHistory, contract, DIMS
from ours4wan21.contracts import create_result, dump, sha256, state_contract
from ours4wan21.selection import selected_paths
from ours4wan21.train import validate_bundle


def select_group(features, group):
    if group == 'a':
        return features[:, :64]
    if group == 'ab':
        return features[:, :128]
    if group == 'ac':
        return torch.cat((features[:, :64], features[:, 128:]), 1)
    if group == 'abc':
        return features
    raise ValueError('unknown group')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--collection-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sea7-cache', type=Path, required=True)
    p.add_argument('--phase', choices=('initialize', 'extract', 'finalize'), required=True)
    p.add_argument('--worker', type=int)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(1)
    paths = selected_paths(a.selection, a.collection_root)
    config = dict(selection_sha256=sha256(a.selection), workers=a.workers,
                  contracts={g:contract('bloc_'+g) for g in DIMS},
                  source_sha256=sha256(PROJECT/'ours4wan21/bloc_features.py'),
                  extractor_sha256=sha256(Path(__file__)),
                  sea7_cache=str(a.sea7_cache.resolve()),
                  sea7_cache_sha256=sha256(a.sea7_cache/'transitions.pt'))
    root = a.output_dir
    if a.phase == 'initialize':
        root = create_result(root, '# BLOC compact features\n\nrun.json freezes sources; rows/ '
            'contains 50x144 pre-action feature vectors and raw input hashes. workers/ records '
            'completed shards; COMPLETE.json seals the index after full coverage checks.')
        for name in ('rows','workers'):
            (root/name).mkdir()
        dump(root/'run.json',config)
        return
    if json.loads((root/'run.json').read_text()) != config:
        raise ValueError('frozen feature inputs changed')
    if a.phase == 'extract':
        if a.worker is None or not 0 <= a.worker < a.workers:
            raise ValueError('explicit worker required')
        if torch.cuda.device_count() != 1:
            raise ValueError('select one GPU')
        for i in range(a.worker, len(paths), a.workers):
            row = json.loads(paths[i].read_text())['trajectory_row']
            path = Path(row['trace_json'])
            trace = json.loads(path.read_text())
            actions = [int(trace['decisions'][2*t]['action']=='reuse') for t in range(50)]
            target = root/'rows'/f'{i:04d}.pt'
            if target.exists():
                raise FileExistsError('fresh extraction only; do not overwrite row')
            history, values, raw = CompactHistory('abc'), [], []
            for t, record in enumerate(trace['step_records']):
                if record['step_index'] != t or record['model_stage'] != 'single':
                    raise ValueError('unexpected input sequence/stage')
                source = Path(record['latent_path'])
                x = torch.load(source, map_location='cpu', weights_only=True)
                if x.dtype != torch.float16 or tuple(x.shape) != (16,21,60,104):
                    raise ValueError('invalid raw archive precision/shape')
                values.append(history.observe(x.cuda(), t, record['sigma'])[0].cpu())
                history.commit(actions[t])
                raw.append(dict(path=str(source),sha256=sha256(source)))
            features = torch.stack(values)
            if features.shape != (50,144) or features[0].any() or not torch.isfinite(features).all():
                raise ValueError('invalid feature row')
            tmp = target.with_suffix('.tmp')
            torch.save(dict(trajectory_id=row['trajectory_id'],sample_id=row['sample_id'],split=row['split'],
                            trace_sha256=sha256(path),actions=torch.tensor(actions),features=features,raw=raw),tmp)
            tmp.replace(target)
            print(f'worker={a.worker} trajectory={i} complete',flush=True)
        dump(root/'workers'/f'{a.worker}.json',dict(status='complete',worker=a.worker,
             rows=len(range(a.worker,len(paths),a.workers)),config_sha256=sha256(root/'run.json')))
        return
    for worker in range(a.workers):
        marker = json.loads((root/'workers'/f'{worker}.json').read_text())
        if marker != dict(status='complete',worker=worker,rows=len(range(worker,len(paths),a.workers)),config_sha256=sha256(root/'run.json')):
            raise ValueError('worker completion mismatch')
    baseline = torch.load(a.sea7_cache/'transitions.pt',map_location='cpu',weights_only=False)
    if json.loads((a.sea7_cache/'COMPLETE.json').read_text())['sha256'] != config['sea7_cache_sha256']:
        raise ValueError('baseline cache hash mismatch')
    validate_bundle(baseline,'sea7')
    if baseline['manifest']['selection_sha256'] != config['selection_sha256']:
        raise ValueError('baseline selection differs')
    features, entries = [], []
    for i,path in enumerate(paths):
        saved = torch.load(root/'rows'/f'{i:04d}.pt',map_location='cpu',weights_only=True)
        source = baseline['manifest']['sources'][i]
        row = json.loads(path.read_text())['trajectory_row']
        if (saved['trajectory_id'] != source['trajectory_id'] or saved['sample_id'] != source['sample_id']
                or saved['split'] != source['source_split'] or saved['trace_sha256'] != sha256(row['trace_json'])
                or not torch.equal(saved['actions'],baseline['tensors']['action'][50*i:50*(i+1)])):
            raise ValueError('feature and baseline trajectory binding mismatch')
        value=saved['features']
        if value.shape != (50,144) or not torch.isfinite(value).all() or value[0].any():
            raise ValueError('feature validity mismatch')
        features.append(value)
        entries.append(dict(trajectory_id=saved['trajectory_id'],file=f'rows/{i:04d}.pt',sha256=sha256(root/'rows'/f'{i:04d}.pt')))
    full = torch.cat(features)
    dump(root/'index.json',dict(config=config,rows=entries))
    for group in DIMS:
        mode='sea7_bloc_'+group
        dest=create_result(root.with_name(root.name+'_'+group+'_cache'), '# BLOC training bundle\n\n'
            'transitions.pt pairs compact features with the unchanged SEA7 transitions, rewards, actions and splits.')
        bundle=dict(baseline)
        bundle['tensors']=dict(baseline['tensors'])
        state=torch.cat((select_group(full,group),baseline['tensors']['state']),1)
        episodes=state.reshape(-1,50,state.shape[1])
        bundle['tensors']['state']=state
        bundle['tensors']['next_state']=torch.cat((episodes[:,1:],episodes[:,-1:]),1).flatten(0,1)
        bundle['manifest']=dict(baseline['manifest'],state=state_contract(mode),
                                bloc_feature_cache=dict(path=str(root),index_sha256=sha256(root/'index.json')))
        validate_bundle(bundle,mode)
        torch.save(bundle,dest/'transitions.pt')
        dump(dest/'manifest.json',bundle['manifest'])
        dump(dest/'COMPLETE.json',dict(status='complete',sha256=sha256(dest/'transitions.pt')))
    dump(root/'COMPLETE.json',dict(status='complete',trajectories=len(paths),index_sha256=sha256(root/'index.json')))


if __name__ == '__main__':
    main()
