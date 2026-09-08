#!/usr/bin/env python3
"""Reproducible CPU smoke: collection adapter -> trainer -> checkpoint -> controller.

All fixture data are synthetic and explicitly marked; this is not Wan video
training or GPU inference and must never supply a production checkpoint.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tests'))
from ours4wan21.contracts import (MODEL_ROOT, TrainingConfig, create_result, dump, sha256)
from ours4wan21.data import build, episode
from ours4wan21.policy import Policy
from ours4wan21.train import train
from ours4wan21.overhead import predictor_fields
from ours4wan21.analysis import metrics_report
from test_rl import ScriptPolicy, trajectory
import torch


def fixture(out):
    protocol = dict(model='Wan2.1-T2V-1.3B', task='t2v-1.3B', size_wh=[832,480],
        frame_num=81, fps=16, sample_steps=50, sample_solver='unipc', sample_shift=5.,
        cfg=5., seed=42, parameter_dtype='bfloat16', offload_model=False, t5_cpu=False)
    paths = []
    for i, (budget, quality, action) in enumerate(((20,21.,1),(30,23.,0),(25,19.,1),(15,24.,0))):
        directory = out / 'synthetic_collection' / 'shards/shard_00/candidates' / f't{i}'
        directory.mkdir(parents=True)
        c = trajectory(ScriptPolicy(action), budget)
        row = dict(trajectory_id=f't{i}', sample_id=f'p{i//2}',
            split='train' if i < 2 else 'evaluation', protocol=protocol,
            trace_json=str(directory/'trace.json'), video_metrics_json=str(directory/'metrics.json'),
            candidate_video=str(directory/'candidate.mp4'), baseline_video=str(directory/'baseline.mp4'),
            mean_psnr=quality)
        dump(directory / 'trace.json', dict(trajectory_id=f't{i}', manifest_record=row,
            task='t2v-1.3B', sampling_steps=50, sample_solver='unipc', shift=5.,
            guide_scale=5., frame_num=81, size_wh=[832,480], decisions=c.decisions))
        dump(directory / 'metrics.json', dict(protocol_id='rgb_full_reference_v1', frames=81,
            width=832, height=480, metrics={'psnr_rgb_db': {'mean':quality}},
            candidate=row['candidate_video'], reference=row['baseline_video'], synthetic=True))
        path = directory / 'CANDIDATE_COMPLETE.json'
        dump(path, dict(schema='ours4wan21_candidate_complete_v3', trajectory_id=f't{i}',
                        trajectory_row=row, synthetic=True))
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if Path(sys.prefix).name != 'wan2.2':
        raise ValueError('use wan2.2')
    torch.set_num_threads(1)
    out = create_result(args.output_dir, '# Ours4Wan21 CPU validation\n\nSynthetic collection metadata, two 2-epoch smoke runs and test output. No actual videos, GPU jobs or trained production policy. VALIDATION.json contains checks and source hashes.')
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(PROJECT/'tests'), '-v'],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out/'unit_tests.txt').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stdout)
    paths = fixture(out)
    results = []
    for mode in ('scalar5','sea7'):
        b = build(paths, mode)
        destination = out / f'{out.name}_{mode}'
        weights = MODEL_ROOT / out.name / mode
        checkpoint = train(b, mode, destination, weights,
            config=replace(TrainingConfig(), epochs=2), device='cpu', smoke=True)
        p = Policy(checkpoint, device='cpu', state_mode=mode, allow_smoke=True)
        c = trajectory(p, 25)
        overhead = predictor_fields(dict(predictor=p.overhead_summary(),pipeline_generate_wall_seconds=1.),c.summary())
        e = episode(c.decisions, 22., mode)
        assert torch.equal(e['state'], torch.tensor([d['state'] for d in c.decisions[::2]]))
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert ckpt['epoch'] == 2 and len(ckpt['optimizer_states']) == 3
        assert len(list((weights/'checkpoints').glob('epoch_*.pt'))) == 2
        rows = [json.loads(s) for s in (destination/'epoch_metrics.jsonl').read_text().splitlines()]
        plot_dir = destination/'smoke_diagnostic_curves'
        plot_dir.mkdir()
        (plot_dir/'README.md').write_text('# Synthetic two-epoch plotting check\n\nActual CPU smoke metrics only; not a 400-epoch research result.\n')
        metrics_report(plot_dir,rows)
        best = min(rows,key=lambda r:r['val']['pi_loss'])['epoch']
        assert (weights/'best_model.pt').resolve() == weights/'checkpoints'/f'epoch_{best:03d}.pt'
        assert ckpt['train_config'] == asdict(replace(TrainingConfig(), epochs=2))
        with torch.no_grad():
            x=e['state'][5]
            z=(x-p.normalizer['mean'])/p.normalizer['std']
            direct=p.net(z[None])[0].softmax(-1)
            action,prob=p.choose(x)
            assert action == int(direct.argmax()) and abs(prob-float(direct[1]))<1e-7
        try:
            Policy(checkpoint,device='cpu')
        except ValueError:
            pass
        else:
            raise AssertionError('smoke policy was accepted for production')
        try:
            Policy(checkpoint,device='cpu',state_mode='sea7' if mode=='scalar5' else 'scalar5',allow_smoke=True)
        except ValueError:
            pass
        else:
            raise AssertionError('cross-mode checkpoint was accepted')
        results.append(dict(mode=mode, epochs=2, transitions=200, train=100, evaluation=100,
                            checkpoint=str(checkpoint), sha256=sha256(checkpoint), exact_k=25,
                            state_parity='pass', best_epoch=best, checkpoint_mode_guard='pass',
                            predictor_overhead=overhead,diagnostic_plotting='pass'))
    dump(out/'VALIDATION.json',dict(status='pass', synthetic_only=True, gpu_used=False,
        runs=results, source_hashes={str(p.relative_to(PROJECT)):sha256(p)
            for p in sorted(PROJECT.rglob('*.py')) if 'experiment_results' not in p.parts and '__pycache__' not in p.parts}))
    print(json.dumps(dict(status='pass', output=str(out))))


if __name__ == '__main__':
    main()
