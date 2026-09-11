"""Audit frozen prompt splits and measure features on one train trajectory.

No generation, reward-based selection, policy training or video decoding.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys
import time

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ours4wan21.contracts import create_result, dump, sha256
from ours4wan21.latent_features import LatentFeatureHistory
from ours4wan21.local_iql import IQLModelConfig, PolicyNet, QNet, ValueNet
from ours4wan21.selection import selected_paths
from features import CompactHistory, DIMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--collection-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reuse-audit', type=Path,
                        help='reuse a completed metadata audit; still verify frozen completion hashes')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('select one CUDA GPU')
    paths = selected_paths(args.selection, args.collection_root)
    by_id, by_text = defaultdict(set), defaultdict(set)
    stages, split_count, latent_counts = Counter(), Counter(), Counter()
    selected = None
    for path in paths:
        row = json.loads(path.read_text())['trajectory_row']
        split = row['split']
        by_id[row['sample_id']].add(split)
        by_text[' '.join(row['prompt'].split()).casefold()].add(split)
        split_count[split] += 1
        trace = json.loads(Path(row['trace_json']).read_text())
        steps = trace['step_records']
        if len(steps) != 50 or len(trace['decisions']) != 100:
            raise ValueError('incomplete trace')
        for step, record in enumerate(steps):
            if (record['step_index'] != step or record['latent_dtype'] != 'torch.float16'
                    or record['latent_shape'] != [16, 21, 60, 104]):
                raise ValueError('input latent contract mismatch')
            if record['model_stage'] != 'single':
                raise ValueError('unexpected expert stage')
            p = Path(record['latent_path'])
            if not args.reuse_audit and (not p.is_file() or not p.stat().st_size):
                raise ValueError('missing input latent')
            stages[record['model_stage']] += 1
            latent_counts[split] += 1
        if selected is None and split == 'train':
            selected = (row, trace)
    if any(len(v) != 1 for v in (*by_id.values(), *by_text.values())):
        raise ValueError('prompt ID or normalized text crosses split')
    if split_count != Counter(train=2400, val=300, test=300):
        raise ValueError('unexpected frozen split')
    if args.reuse_audit:
        prior = json.loads(args.reuse_audit.read_text())
        if prior.get('status') != 'pass' or prior['selection_sha256'] != sha256(args.selection):
            raise ValueError('prior audit selection mismatch')
    out = create_result(args.output_dir, '# BLOC v2 input audit and extraction microbenchmark\n\n'
        'audit.json checks frozen selection, prompt text separation and all 150000 latent paths/metadata. '
        'benchmark.json contains one train-trajectory GPU extraction measurements, not rollout quality. '
        'Only the benchmark trajectory tensors are decoded and hashed; no full latent-content audit.')
    dump(out/'audit.json', dict(status='pass', selection_sha256=sha256(args.selection),
        trajectories=len(paths), prompt_ids=len(by_id), unique_normalized_texts=len(by_text),
        trajectories_by_split=dict(split_count), input_latents_by_split=dict(latent_counts),
        stages=dict(stages), prompt_id_cross_split=0, prompt_text_cross_split=0,
        validation_scope='all completion hashes and trace metadata; nonempty latent paths reused from prior audit if specified; no full tensor-content audit',
        reused_audit=None if not args.reuse_audit else dict(path=str(args.reuse_audit),sha256=sha256(args.reuse_audit)),
        missing_for_residual_features=['token residuals or predetermined residual summaries'],
        missing_for_exact_branching=['UniPC internal state', 'full precision latent', 'both CFG caches', 'RNG state']))
    row, trace = selected
    inputs, provenance = [], []
    for record in trace['step_records']:
        p = Path(record['latent_path'])
        x = torch.load(p, map_location='cpu', weights_only=True)
        if x.shape != (16,21,60,104) or x.dtype != torch.float16 or not torch.isfinite(x).all():
            raise ValueError('benchmark input invalid')
        inputs.append(x)
        provenance.append(dict(path=str(p), sha256=sha256(p)))
    actions = [int(trace['decisions'][2*t]['action']=='reuse') for t in range(50)]
    if any(trace['decisions'][2*t]['action'] != trace['decisions'][2*t+1]['action'] for t in range(50)):
        raise ValueError('mixed CFG actions')
    modes = ['dynamics128', *DIMS]
    samples = {m: [] for m in modes}
    for repeat in range(4):
        for mode in (modes if repeat % 2 == 0 else list(reversed(modes))):
            torch.cuda.empty_cache()
            make = lambda: (LatentFeatureHistory(['dynamics_raw_sea128']) if mode == 'dynamics128'
                            else CompactHistory(mode))
            history = make()
            # Identical startup input allocation. H2D disk/cache loading is outside timers.
            x = inputs[0].cuda().float()
            torch.cuda.synchronize()
            initial_allocated = torch.cuda.memory_allocated()
            initial_reserved = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            wall, cuda = [], []
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for step, cpu in enumerate(inputs):
                del x
                x = cpu.cuda().float()
                torch.cuda.synchronize()
                begun = time.perf_counter()
                start.record()
                value = history.observe(x, step, trace['step_records'][step]['sigma'])
                if isinstance(value, dict):
                    value = value['dynamics_raw_sea128']
                result = value.cpu()
                end.record()
                end.synchronize()
                wall.append(time.perf_counter()-begun)
                cuda.append(start.elapsed_time(end)/1000.)
                history.commit(actions[step])
            measurement = dict(repeat=repeat, wall_seconds=sum(wall), cuda_span_seconds=sum(cuda),
                per_step_wall_seconds=wall, per_step_cuda_span_seconds=cuda,
                peak_extra_allocated_bytes=max(0,torch.cuda.max_memory_allocated()-initial_allocated),
                peak_extra_reserved_bytes=max(0,torch.cuda.max_memory_reserved()-initial_reserved))
            if repeat:
                samples[mode].append(measurement)
            del history, value, result, x
            print(json.dumps(dict(mode=mode, repeat=repeat, wall_seconds=measurement['wall_seconds'])), flush=True)
    parameters = {}
    for mode, dim in [('sea7',7),('dynamics128',135),*((m,7+d) for m,d in DIMS.items())]:
        cfg = IQLModelConfig(input_dim=dim)
        counts = [sum(p.numel() for p in cls(cfg).parameters()) for cls in (PolicyNet,QNet,QNet,ValueNet)]
        parameters[mode] = dict(input_dim=dim, actor_parameters=counts[0], trainable_parameters=sum(counts))
    summary = {m:dict(median_extraction_wall_seconds_per50steps=statistics.median(v['wall_seconds'] for v in rows),
                    max_extra_allocated_bytes=max(v['peak_extra_allocated_bytes'] for v in rows))
               for m,rows in samples.items()}
    dump(out/'benchmark.json', dict(status='complete', trajectory_id=row['trajectory_id'], split='train',
         gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, inputs=provenance,
         repeats=3, warmup_repeats=1, samples=samples, summary=summary, parameters=parameters,
         scope='observe through CPU vector transfer; input H2D and action commit excluded; CUDA span includes host-induced gaps; isolated extraction, no resident DiT',
         caveat='single train trajectory; does not measure full-generation speed or PSNR; cached allocator and GPU clock may affect results',
         sources={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('features.py'),
                    PROJECT/'ours4wan21/bloc_features.py', PROJECT/'ours4wan21/latent_features.py']}))
    dump(out/'COMPLETE.json',dict(status='complete', files={p:sha256(out/p) for p in ('audit.json','benchmark.json')}))
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
