"""Training curves and local post-300 validation-only checkpoint selection."""
import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from .contracts import MODEL_ROOT, TrainingConfig, create_result, dump, sha256, under
from .local_iql import IQLModelConfig, PolicyNet, QNet, apply_normalizer
from .train import validate_bundle


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def midpoint_iqr(values):
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    centers = (np.arange(len(ordered)) + .5) / len(ordered)
    return float(np.interp(.75, centers, ordered) - np.interp(.25, centers, ordered))


def rank_checkpoints(epochs, q_values, actions, logs):
    """Same full-validation, two-sided actor/D_Q criterion as local SEA7 run."""
    if len(epochs) < 3 or epochs != list(range(epochs[0], epochs[-1] + 1)):
        raise ValueError('need at least three consecutive checkpoints')
    if q_values.shape[:2] != actions.shape or q_values.shape[0] != len(epochs) or q_values.shape[2] != 2:
        raise ValueError('invalid Q/actor arrays')
    if not q_values.shape[1] or not np.isfinite(q_values).all():
        raise ValueError('nonfinite/empty checkpoint census')
    scales = [midpoint_iqr(q) for q in q_values]
    adjacent = []
    for i in range(1,len(epochs)):
        dq = float(np.abs(q_values[i].astype(float)-q_values[i-1]).mean())
        scale = (scales[i] + scales[i-1]) / 2
        adjacent.append(dict(from_epoch=epochs[i-1], to_epoch=epochs[i],
            actor_agreement=float((actions[i] == actions[i-1]).mean()),
            dq=dq, q_iqr_scale=scale, dq_over_q_iqr=dq/max(scale,1e-12)))
    by = {r['to_epoch']: r for r in adjacent}
    ranked = []
    for e in epochs[1:-1]:
        left, right = by[e], by[e+1]
        ranked.append(dict(epoch=e, val_pi_loss=logs[e]['val']['pi_loss'],
            val_q_loss=logs[e]['val']['q_loss'],
            min_actor=min(left['actor_agreement'],right['actor_agreement']),
            mean_actor=(left['actor_agreement']+right['actor_agreement'])/2,
            mean_dq=(left['dq']+right['dq'])/2,
            mean_normalized_dq=(left['dq_over_q_iqr']+right['dq_over_q_iqr'])/2))
    gate = .96 if any(r['min_actor'] >= .96 for r in ranked) else (
        .95 if any(r['min_actor'] >= .95 for r in ranked) else max(r['min_actor'] for r in ranked))
    eligible = sorted((r for r in ranked if r['min_actor'] >= gate),
        key=lambda r:(r['mean_normalized_dq'],r['mean_dq'],-r['min_actor'],-r['epoch']))
    return adjacent, ranked, dict(gate=gate, selected=eligible[0], eligible_sorted=eligible,
        criterion='two-sided actor agreement gate .96, fallback .95, fallback maximum observed; minimize mean D_Q/IQR then raw D_Q; prefer higher min actor agreement then later epoch',
        zero_iqr_guard='max(IQR,1e-12); exact local formula otherwise')


def metrics_report(out, rows):
    flat = [dict(epoch=r['epoch'], elapsed_seconds=r['elapsed_seconds'],
        **{f'{split}_{k}':v for split in ('train','val') for k,v in r[split].items()}) for r in rows]
    write_csv(out / 'training_metrics.csv', flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = ('v_loss','q_loss','pi_loss','advantage_mean','advantage_std',
               'actor_weight_mean','actor_weight_max','skip_rate_data','skip_rate_policy',
               'actor_examples','constraint_forced_examples','reward')
    fig, axes = plt.subplots(4,3,figsize=(14,12),constrained_layout=True)
    for ax,key in zip(axes.flat,metrics):
        for split in ('train','val'):
            ax.plot([r['epoch'] for r in rows],[r[split][key] for r in rows],label=split,lw=1)
        ax.set_title(key)
        ax.set_xlabel('epoch')
    axes.flat[0].legend()
    fig.savefig(out/'training_metrics.png',dpi=160)
    fig.savefig(out/'training_metrics.svg')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-result', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    a = p.parse_args()
    torch.set_num_threads(1)
    weights = under(a.checkpoint_dir, MODEL_ROOT)
    training = a.training_result.resolve(strict=True)
    completion = json.loads((training/'TRAINING_COMPLETE.json').read_text())
    config = json.loads((training/'config.json').read_text())
    if completion.get('status') != 'complete' or completion.get('smoke_only') or completion['epochs'] != 400:
        raise ValueError('post300 selection requires complete production 400-epoch training')
    if any(config[k] != v for k,v in asdict(TrainingConfig()).items()):
        raise ValueError('training configuration differs from local frozen settings')
    if (training/'model_weights').resolve() != weights.resolve():
        raise ValueError('checkpoint directory does not belong to training result')
    data_file = a.dataset/'transitions.pt'
    if sha256(data_file) != json.loads((a.dataset/'COMPLETE.json').read_text())['sha256']:
        raise ValueError('dataset hash mismatch')
    bundle = torch.load(data_file,map_location='cpu',weights_only=False)
    validate_bundle(bundle,config['state']['mode'])
    if bundle['manifest'] != json.loads((training/'dataset_manifest.json').read_text()):
        raise ValueError('analysis must use the exact training cache')
    rows = [json.loads(s) for s in (training/'epoch_metrics.jsonl').read_text().splitlines() if s.strip()]
    if [r['epoch'] for r in rows] != list(range(1,401)):
        raise ValueError('requires all 400 complete training/validation rows')
    if not all(math.isfinite(v) for r in rows for s in ('train','val') for v in r[s].values()):
        raise ValueError('nonfinite training metrics')
    paths = {e:weights/'checkpoints'/f'epoch_{e:03d}.pt' for e in range(1,401)}
    if not all(path.is_file() for path in paths.values()):
        raise ValueError('missing epoch checkpoint')
    if sha256(paths[400]) != completion['final_sha256']:
        raise ValueError('final checkpoint hash mismatch')
    out = create_result(a.output_dir, '# Offline training diagnostics and checkpoint selection\n\ntraining_metrics.csv/png/svg covers all epochs. post300_* files census all discretionary validation states; checkpoint_selection.json locks the selected source and SHA. No test or VBench data enter selection.')
    metrics_report(out,rows)
    first = torch.load(paths[300],map_location='cpu',weights_only=False)
    positions = bundle['val_indices']
    positions = positions[bundle['tensors']['actor_mask'][positions] > .5]
    states = apply_normalizer(bundle['tensors']['state'][positions],first['normalizer'])
    device = torch.device(a.device)
    if device.type=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise ValueError('select one CUDA device for analysis')
    mc = IQLModelConfig(**first['model_config'])
    nets = {k:cls(mc).to(device).eval() for k,cls in (
        ('q1_net',QNet),('q2_net',QNet),('policy_net',PolicyNet))}
    epochs=list(range(300,401))
    values=np.empty((101,len(positions),2),dtype=np.float32)
    actions=np.empty((101,len(positions)),dtype=np.uint8)
    hashes={}
    for index,e in enumerate(epochs):
        checkpoint=torch.load(paths[e],map_location='cpu',weights_only=False)
        if checkpoint['epoch'] != e or checkpoint['state'] != config['state'] or checkpoint['smoke_only']:
            raise ValueError('checkpoint contract mismatch')
        if not all(torch.equal(first['normalizer'][k],checkpoint['normalizer'][k]) for k in ('mean','std')):
            raise ValueError('normalizer changed across epochs')
        for key,net in nets.items():
            net.load_state_dict(checkpoint[key])
        with torch.inference_mode():
            for start in range(0,len(states),2048):
                stop=min(start+2048,len(states))
                x=states[start:stop].to(device)
                values[index,start:stop]=torch.minimum(nets['q1_net'](x),nets['q2_net'](x)).cpu().numpy()
                actions[index,start:stop]=nets['policy_net'](x).argmax(-1).cpu().numpy()
        hashes[str(e)]=sha256(paths[e])
        print(json.dumps(dict(checkpoint_epoch=e,validation_free_rows=len(states))),flush=True)
    adjacent, ranking, selection=rank_checkpoints(epochs,values,actions,{r['epoch']:r for r in rows})
    write_csv(out/'post300_adjacent_census.csv',adjacent)
    write_csv(out/'post300_checkpoint_ranking.csv',ranking)
    np.savez_compressed(out/'post300_validation_values.npz',epochs=epochs,
                       row_index=positions.numpy(),qmin=values,actor=actions)
    e=selection['selected']['epoch']
    dump(out/'checkpoint_selection.json',dict(status='selected',checkpoint=str(paths[e].resolve()),
        checkpoint_sha256=hashes[str(e)],checkpoint_epoch=e,**selection,
        validation_actor_free_population=len(states),selection_window='two-sided e301-e399; census e300-e400',
        checkpoint_sha256_by_epoch=hashes,
        dataset_sha256=sha256(data_file),metrics_sha256=sha256(training/'epoch_metrics.jsonl'),
        limitation='Offline relative stability only; not proof of rollout quality/convergence. No test/VBench-based selection.'))
    (out/'selected_model.pt').symlink_to(paths[e].resolve())
    best=min(rows,key=lambda r:r['val']['pi_loss'])['epoch']
    (out/'REPORT.md').write_text(f'''# Training diagnostics

Selected checkpoint: epoch {e}, using the local two-sided post300 actor/Q stability rule.
The global minimum validation actor-loss checkpoint is epoch {best}; final is epoch 400.
These are different choices. Use checkpoint_selection.json and its SHA for the planned VBench run.

training_metrics.png/svg and training_metrics.csv cover all 400 epochs: Q/V/actor loss,
advantage mean/std, actor weight mean/max, logged/greedy skip fraction, actor/forced counts,
terminal reward and elapsed time. Skip fractions here are teacher-forced state diagnostics,
not closed-loop generated speedup or video quality.

Q/V loss rises can reflect value-scale drift or fitting error; assess them with actor agreement
and normalized adjacent Q changes. Train V uses target Q while validation V uses online Q,
as in the original trainer. Actor loss only includes discretionary rows. Exact-K-forced rows
still train critics. Stable actor decisions do not prove a calibrated critic or better videos.

post300_adjacent_census.csv records actor agreement, mean absolute Q change and Q-IQR-normalized
change. post300_checkpoint_ranking.csv contains two-sided candidates; checkpoint_selection.json
records gate fallback and tie-breaking. All {len(states)} discretionary validation states are used.
Held-out test rows and all VBench results are excluded from selection.
''')
    dump(out/'COMPLETE.json',dict(status='complete',selected_epoch=e))


if __name__=='__main__':
    main()
