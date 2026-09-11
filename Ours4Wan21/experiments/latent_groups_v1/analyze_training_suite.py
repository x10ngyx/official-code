#!/usr/bin/env python3
"""Fail-closed cross-group audit and descriptive readout for twelve 400-epoch runs."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ours4wan21.contracts import EXP_ROOT, MODEL_ROOT, TrainingConfig, create_result, dump, sha256

MODES = (
    'scalar5', 'sea7', 'sea7_dynamics_raw_sea128', 'sea7_cache_update192',
    'sea7_local_drift1024', 'sea7_spatial_gradient96', 'sea7_channel_geometry240',
    'sea7_distribution256', 'sea7_spectral_drift512', 'sea7_spectral_phase512',
    'sea7_spectral_shape576', 'sea7_spectral_dynamics1024',
)
SHORT = {
    'scalar5': 'Scalar5', 'sea7': 'SEA7',
    'sea7_dynamics_raw_sea128': 'Dynamics128', 'sea7_cache_update192': 'CacheUpdate192',
    'sea7_local_drift1024': 'LocalDrift1024', 'sea7_spatial_gradient96': 'SpatialGrad96',
    'sea7_channel_geometry240': 'ChannelGeom240', 'sea7_distribution256': 'Distribution256',
    'sea7_spectral_drift512': 'SpectralDrift512', 'sea7_spectral_phase512': 'SpectralPhase512',
    'sea7_spectral_shape576': 'SpectralShape576', 'sea7_spectral_dynamics1024': 'SpectralDynamics1024',
}
METRICS = ('v_loss', 'q_loss', 'pi_loss', 'advantage_mean', 'advantage_std',
           'actor_weight_mean', 'actor_weight_max', 'skip_rate_data',
           'skip_rate_policy', 'actor_examples', 'constraint_forced_examples', 'reward')


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def slope(values: list[float]) -> float:
    return float(np.polyfit(np.arange(len(values), dtype=np.float64), values, 1)[0])


def model_parameters(checkpoint: dict) -> tuple[int, int]:
    counts = {name: sum(value.numel() for value in checkpoint[name].values())
              for name in ('value_net', 'q1_net', 'q2_net', 'policy_net')}
    return counts['policy_net'], sum(counts.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-name', default='ours21_random3000_12groups_v1')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    orchestration = EXP_ROOT / f'{args.suite_name}_orchestration'
    complete = json.loads((orchestration / 'COMPLETE.json').read_text())
    if complete.get('status') != 'complete' or complete.get('modes') != 12 or complete.get('epochs_per_mode') != 400:
        raise ValueError('suite orchestration is not a complete 12x400 run')

    out = create_result(args.output_dir, '# Twelve-group 400-epoch training analysis\n\nThis directory contains the fail-closed cross-group audit, tidy 400-epoch metrics, descriptive comparisons, chart map and report source material. Training-only evidence does not establish rollout speed or video quality.')
    epoch_rows, summaries, selected_details = [], [], []
    reference_sources = reference_splits = reference_selection = reference_hparams = None
    per_mode_rows = {}
    selection_sha = complete['selection_sha256']
    for mode in MODES:
        training = EXP_ROOT / f'{args.suite_name}_{mode}_train'
        analysis = EXP_ROOT / f'{args.suite_name}_{mode}_analysis'
        cache = EXP_ROOT / f'{args.suite_name}_{mode}_cache'
        weights = MODEL_ROOT / f'{args.suite_name}_{mode}'
        marker = json.loads((training / 'TRAINING_COMPLETE.json').read_text())
        config = json.loads((training / 'config.json').read_text())
        manifest = json.loads((training / 'dataset_manifest.json').read_text())
        split = json.loads((training / 'split.json').read_text())
        selection = json.loads((analysis / 'checkpoint_selection.json').read_text())
        analysis_marker = json.loads((analysis / 'COMPLETE.json').read_text())
        cache_marker = json.loads((cache / 'COMPLETE.json').read_text())
        raw_rows = [json.loads(line) for line in (training / 'epoch_metrics.jsonl').read_text().splitlines() if line.strip()]
        if (marker.get('status') != 'complete' or marker.get('epochs') != 400 or marker.get('smoke_only') or
                analysis_marker.get('status') != 'complete' or cache_marker.get('status') != 'complete' or
                [row['epoch'] for row in raw_rows] != list(range(1, 401))):
            raise ValueError(f'incomplete production artifacts for {mode}')
        if any(config[key] != value for key, value in asdict(TrainingConfig()).items()):
            raise ValueError(f'frozen training hyperparameters changed for {mode}')
        if config['state']['mode'] != mode or manifest['state'] != config['state']:
            raise ValueError(f'state-mode contract mismatch for {mode}')
        if manifest.get('selection_sha256') != selection_sha or manifest.get('trajectories') != 3000:
            raise ValueError(f'selection or trajectory count mismatch for {mode}')
        if sha256(cache / 'transitions.pt') != cache_marker['sha256']:
            raise ValueError(f'cache hash mismatch for {mode}')
        if (training / 'model_weights').resolve() != weights.resolve():
            raise ValueError(f'checkpoint ownership mismatch for {mode}')
        if len(list((weights / 'checkpoints').glob('epoch_*.pt'))) != 400:
            raise ValueError(f'checkpoint count mismatch for {mode}')
        selected_epoch = int(selection['checkpoint_epoch'])
        selected_path = weights / 'checkpoints' / f'epoch_{selected_epoch:03d}.pt'
        if (sha256(selected_path) != selection['checkpoint_sha256'] or
                (analysis / 'selected_model.pt').resolve() != selected_path.resolve()):
            raise ValueError(f'selected checkpoint identity mismatch for {mode}')
        if not all(math.isfinite(value) for row in raw_rows for side in ('train', 'val')
                   for value in row[side].values()):
            raise ValueError(f'nonfinite epoch metrics for {mode}')
        hparams = {key: config[key] for key in asdict(TrainingConfig())}
        source_identity = [(source['trajectory_id'], source['sample_id'], source['split'], source['files'])
                           for source in manifest['sources']]
        if reference_sources is None:
            reference_sources, reference_splits = source_identity, split['prompt_splits']
            reference_selection, reference_hparams = manifest['selection'], hparams
        elif (source_identity != reference_sources or split['prompt_splits'] != reference_splits or
              manifest['selection'] != reference_selection or hparams != reference_hparams):
            raise ValueError(f'cross-group dataset/split/hyperparameter mismatch for {mode}')

        for row in raw_rows:
            epoch_rows.append(dict(mode=mode, label=SHORT[mode], kind='control' if mode in ('scalar5', 'sea7') else 'feature',
                input_dim=len(config['state']['names']), epoch=row['epoch'], elapsed_seconds=row['elapsed_seconds'],
                **{f'{side}_{metric}': row[side][metric] for side in ('train', 'val') for metric in METRICS}))
        per_mode_rows[mode] = raw_rows
        tail = raw_rows[-50:]
        best = min(raw_rows, key=lambda row: row['val']['pi_loss'])
        selected_row = raw_rows[selected_epoch - 1]
        selected_rank = selection['selected']
        adjacent = read_csv(analysis / 'post300_adjacent_census.csv')
        checkpoint = torch.load(selected_path, map_location='cpu', weights_only=False)
        policy_params, trainable_params = model_parameters(checkpoint)
        summary = dict(
            mode=mode, label=SHORT[mode], kind='control' if mode in ('scalar5', 'sea7') else 'feature',
            input_dim=len(config['state']['names']), policy_parameters=policy_params,
            trainable_parameters=trainable_params, selected_epoch=selected_epoch,
            selection_gate=float(selection['gate']), selected_min_actor_agreement=float(selected_rank['min_actor']),
            selected_mean_actor_agreement=float(selected_rank['mean_actor']),
            selected_mean_dq=float(selected_rank['mean_dq']),
            selected_mean_normalized_dq=float(selected_rank['mean_normalized_dq']),
            post300_adjacent_actor_agreement_median=float(np.median([float(row['actor_agreement']) for row in adjacent])),
            post300_adjacent_actor_agreement_min=min(float(row['actor_agreement']) for row in adjacent),
            best_val_pi_epoch=int(best['epoch']), best_val_pi_loss=float(best['val']['pi_loss']),
            selected_val_pi_loss=float(selected_row['val']['pi_loss']),
            selected_val_q_loss=float(selected_row['val']['q_loss']),
            selected_val_v_loss=float(selected_row['val']['v_loss']),
            final_val_pi_loss=float(raw_rows[-1]['val']['pi_loss']),
            final_val_q_loss=float(raw_rows[-1]['val']['q_loss']),
            final_val_v_loss=float(raw_rows[-1]['val']['v_loss']),
            tail50_val_pi_mean=float(np.mean([row['val']['pi_loss'] for row in tail])),
            tail50_val_pi_std=float(np.std([row['val']['pi_loss'] for row in tail])),
            tail50_val_q_mean=float(np.mean([row['val']['q_loss'] for row in tail])),
            tail50_val_q_slope_per_100_epochs=100 * slope([row['val']['q_loss'] for row in tail]),
            tail50_val_v_mean=float(np.mean([row['val']['v_loss'] for row in tail])),
            tail50_val_v_slope_per_100_epochs=100 * slope([row['val']['v_loss'] for row in tail]),
            tail50_val_skip_policy_mean=float(np.mean([row['val']['skip_rate_policy'] for row in tail])),
            tail50_val_skip_policy_std=float(np.std([row['val']['skip_rate_policy'] for row in tail])),
            logged_val_skip_rate=float(raw_rows[-1]['val']['skip_rate_data']),
            train_actor_weight_cap_epochs=sum(row['train']['actor_weight_max'] >= 20 - 1e-6 for row in raw_rows),
            val_actor_weight_cap_epochs=sum(row['val']['actor_weight_max'] >= 20 - 1e-6 for row in raw_rows),
            elapsed_hours=float(raw_rows[-1]['elapsed_seconds']) / 3600,
            train_transitions=int(split['train_count']), val_transitions=int(split['val_count']),
            test_transitions=int(split['test_count']), checkpoint_sha256=selection['checkpoint_sha256'])
        summaries.append(summary)
        selected_details.append(dict(**summary, selected_checkpoint=str(selected_path),
            training_result=str(training), analysis_result=str(analysis), cache=str(cache)))

    feature_modes = [mode for mode in MODES if mode not in ('scalar5', 'sea7')]
    aggregate_rows = []
    for epoch in range(1, 401):
        for series, modes in (('Scalar5', ['scalar5']), ('SEA7', ['sea7']), ('Feature median', feature_modes)):
            rows = [per_mode_rows[mode][epoch - 1] for mode in modes]
            values = {}
            for metric in ('pi_loss', 'q_loss', 'v_loss', 'skip_rate_policy'):
                metric_values = np.asarray([row['val'][metric] for row in rows], dtype=np.float64)
                values.update({f'val_{metric}': float(np.median(metric_values)),
                               f'val_{metric}_q25': float(np.quantile(metric_values, .25)),
                               f'val_{metric}_q75': float(np.quantile(metric_values, .75))})
            aggregate_rows.append(dict(epoch=epoch, series=series, group_count=len(modes), **values))

    write_csv(out / 'epoch_metrics_long.csv', epoch_rows)
    write_csv(out / 'epoch_aggregate.csv', aggregate_rows)
    write_csv(out / 'suite_summary.csv', summaries)
    dump(out / 'selected_checkpoints.json', dict(schema='ours21_12group_selected_checkpoints_v1', rows=selected_details))
    split_counts = reference_selection['split_counts']
    validation = dict(status='pass', assessment='Ready to share for training-only interpretation',
        suite=args.suite_name, groups=12, controls=2, feature_groups=10, epochs_per_group=400,
        total_epoch_records=len(epoch_rows), selection_sha256=selection_sha,
        trajectories=reference_selection['selected_count'], trajectory_split_counts=split_counts,
        prompt_counts={split: sum(value == ('evaluation' if split == 'val' else split)
            for value in reference_splits.values()) for split in ('train', 'val', 'test')},
        transition_split_counts=dict(train=summaries[0]['train_transitions'], val=summaries[0]['val_transitions'],
                                     test=summaries[0]['test_transitions']),
        checks=['400 finite train/validation rows per group', '400 checkpoints per group',
                'shared selection, source identities, prompt splits and frozen hyperparameters',
                'cache, final checkpoint and selected checkpoint hashes',
                'post300 two-sided checkpoint selection artifacts'],
        limitations=['single seed per state mode; no run-to-run uncertainty',
                     'different input dimensions imply different first-layer parameter counts',
                     'offline loss and logged-state skip diagnostics do not establish rollout speed or video quality',
                     'train V uses target Q while validation V uses online Q and the two curves are not a direct gap'])
    dump(out / 'VALIDATION.json', validation)
    chart_map = [
        dict(section='Loss dynamics', question='How did validation objectives evolve?', family='trend',
             chart='line', dataset='epoch_aggregate.csv', fields=['epoch', 'series', 'val_pi_loss', 'val_q_loss', 'val_v_loss'],
             claim='Controls and the across-feature median are shown without assigning a training-only winner.'),
        dict(section='Late-training comparison', question='Which groups had lower late validation actor loss?',
             family='comparison', chart='bar', dataset='suite_summary.csv', fields=['label', 'tail50_val_pi_mean'],
             claim='Descriptive lower-is-better comparison; not a rollout-quality ranking.'),
        dict(section='Checkpoint stability', question='Were selected policies stable on adjacent post300 checkpoints?',
             family='relationship', chart='scatter', dataset='suite_summary.csv',
             fields=['selected_mean_normalized_dq', 'selected_min_actor_agreement', 'label', 'kind'],
             claim='Selected checkpoints jointly expose actor agreement and critic drift.')]
    dump(out / 'chart_map.json', chart_map)
    ranked = sorted(summaries, key=lambda row: row['tail50_val_pi_mean'])
    (out / 'RESULTS.md').write_text(
        '# Twelve-group 400-epoch training readout\n\n'
        f'All 12 runs completed 400 epochs on the same {reference_selection["selected_count"]}-trajectory selection '
        f'(train/val/test {split_counts["train"]}/{split_counts["val"]}/{split_counts["test"]}). '
        'All epoch values were finite and all cache/checkpoint/selection hashes passed.\n\n'
        f'The lowest descriptive final-50 validation actor loss was {ranked[0]["label"]} '
        f'({ranked[0]["tail50_val_pi_mean"]:.6g}); SEA7 was '
        f'{next(row for row in summaries if row["mode"] == "sea7")["tail50_val_pi_mean"]:.6g}. '
        'This comparison is not a video-quality or deployment-speed result.\n\n'
        'Checkpoint selection uses only e300-e400 validation free states and the frozen two-sided actor/Q stability rule. '
        'Held-out test rows and VBench do not participate. Because every group has one seed and feature groups have different '
        'input dimensions, training-metric differences are descriptive rather than uncertainty-qualified causal effects.\n')
    dump(out / 'COMPLETE.json', dict(status='complete', validation_sha256=sha256(out / 'VALIDATION.json'),
        summary_sha256=sha256(out / 'suite_summary.csv'), epoch_metrics_sha256=sha256(out / 'epoch_metrics_long.csv')))


if __name__ == '__main__':
    main()
