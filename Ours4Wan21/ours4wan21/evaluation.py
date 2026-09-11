"""Shared video evaluation plus Ours-specific predictor overhead reporting."""
import argparse
import json
from pathlib import Path
import sys

from .contracts import OFFICIAL, PROTOCOL, dump
from .overhead import predictor_fields, aggregate
from .shared import benchmark


def add_overhead(directory, rows):
    result = []
    manifest = json.loads((directory / 'run.json').read_text())
    for row in rows:
        sid = row['sample_id']
        timing = json.loads((directory / 'timings' / f'{sid}.json').read_text())
        trace = json.loads((directory / 'traces' / f'{sid}.json').read_text())
        if timing.get('predictor', {}).get('profile') != manifest.get('predictor_flops_profile'):
            raise ValueError('predictor FLOPs profile differs from run manifest')
        feature = timing.get('latent_feature')
        if trace['state_contract'].get('latent_feature') is not None and feature is None:
            raise ValueError('missing feature extraction timing')
        result.append(dict(**row, **predictor_fields(timing, trace),
                           latent_feature_wall_seconds=feature['wall_seconds'] if feature else 0.))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('summarize', 'evaluate'))
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    args = parser.parse_args()
    if 'wan2.2' not in Path(sys.prefix).name.lower():
        raise ValueError('use conda environment wan2.2')
    shared = benchmark()
    if args.action == 'evaluate':
        return shared.main()
    if args.profile is None:
        parser.error('--profile is required for summarize')
    base, cand = shared.external(args.baseline_dir), shared.external(args.candidate_dir)
    target = cand / 'performance.json'
    if target.exists():
        raise FileExistsError(target)
    profile = json.loads(args.profile.read_text())
    if profile['input']['video_shape_fhw'] != [81,480,832]:
        raise ValueError('wrong FLOPs profile shape')
    bm, br = shared.collect(base, profile)
    cm, cr = shared.collect(cand, profile)
    if bm['method'] != 'baseline' or cm['method'] != 'ours':
        raise ValueError('requires native baseline and Ours candidate')
    for key in ('protocol', 'prompts', 'checkpoint_dir', 'gpu', 'gpu_uuid'):
        if bm[key] != cm[key]:
            raise ValueError('baseline/candidate mismatch: ' + key)
    cr = add_overhead(cand, cr)
    dump(target, dict(protocol=PROTOCOL, method='ours', baseline=br, candidate=cr,
        latency_speedup_ratio_of_sums=sum(r['generate_seconds'] for r in br)/sum(r['generate_seconds'] for r in cr),
        dit_tflops_speedup_ratio_of_sums=sum(r['dit_tflops'] for r in br)/sum(r['dit_tflops'] for r in cr),
        predictor_overhead=aggregate(cr),
        latent_feature_wall_seconds_total=sum(r["latent_feature_wall_seconds"] for r in cr),
        scope='Full generate latency; T5/DiT/VAE and predictor separately reported. Predictor timing is nested in DiT/generate, not additive. DiT TFLOPs exclude predictor (reported separately), SEA FFT, residual add and scheduler. Predictor estimator conventions are recorded per sample.'))
