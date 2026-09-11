"""Freeze twenty VBench50 prompts and reuse their native baseline evidence."""
import json
import math
from pathlib import Path
import random
import subprocess
from fractions import Fraction

from .contracts import PROTOCOL, create_result, dump, sha256
from .online_common import OnlineConfig, prompts, read, seal, verified


def select_evaluation(rows, seed=42):
    if len(rows) != 50 or len({r['sample_id'] for r in rows}) != 50:
        raise ValueError('evaluation source must contain exactly 50 distinct prompts')
    return sorted(random.Random(seed).sample(sorted(rows, key=lambda r:r['sample_id']), 20),
                  key=lambda r:r['sample_id'])


def video_geometry(path):
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
        '-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate',
        '-of','json',str(path)],text=True))
    stream=probe['streams'][0]
    if ((stream['width'],stream['height'],int(stream['nb_read_frames']))!=(832,480,81) or
            Fraction(stream['r_frame_rate'])!=16):
        raise ValueError('reused baseline video violates resolution/frame-count/FPS protocol')
    return probe


def gpu_uuids(selectors):
    """Resolve physical selectors without initializing CUDA or loading models."""
    values = []
    for selector in selectors:
        selector = selector.strip()
        if selector.startswith('GPU-'):
            uuid = selector
        elif selector.isdigit():
            lines = subprocess.check_output(['nvidia-smi', '--id='+selector,
                '--query-gpu=uuid', '--format=csv,noheader,nounits'], text=True).splitlines()
            if len(lines) != 1 or not lines[0].strip().startswith('GPU-'):
                raise ValueError('cannot resolve physical GPU '+selector)
            uuid = lines[0].strip()
        else:
            raise ValueError('expected a physical GPU index or UUID')
        values.append(uuid)
    if not values or len(set(values)) != len(values):
        raise ValueError('provide distinct physical GPUs')
    return values


def load_evaluation_bundle(folder):
    folder = Path(folder).resolve()
    if not verified(folder):
        raise ValueError('evaluation bundle is incomplete')
    data = read(folder/'manifest.json')
    if data['schema'] != 'ours21_online_eval20_reference_v1' or data['protocol'] != PROTOCOL:
        raise ValueError('wrong evaluation reference protocol')
    if (data['targets'] != list(OnlineConfig().evaluation_targets) or
            data['skip_budgets'] != [23,29,35] or data['vbench_enabled'] is not False):
        raise ValueError('evaluation reference targets differ')
    if len(data['rows']) != 20 or len({r['sample_id'] for r in data['rows']}) != 20:
        raise ValueError('evaluation reference must contain twenty unique prompts')
    for path, digest in data['source_files'].items():
        if sha256(path) != digest:
            raise ValueError('evaluation source changed: '+path)
    for row in data['rows']:
        base = folder/'baselines'/row['sample_id']
        if not verified(base):
            raise ValueError('reused baseline is incomplete')
        generation = read(base/'generation.json')
        if (generation['gpu_uuid'] != row['baseline_gpu_uuid'] or
                generation['protocol'] != PROTOCOL or
                generation['identity']['job']['prompt'] != row['prompt']):
            raise ValueError('reused baseline identity differs')
    return data


def build_evaluation(args):
    source = Path(args.source_run).resolve(strict=True)
    out = Path(args.output_dir).resolve()
    if out.exists():
        data = load_evaluation_bundle(out)
        if data['source_run'] != str(source) or data['selection_seed'] != 42:
            raise ValueError('existing evaluation bundle has different inputs')
        print(out)
        return
    config = read(source/'config.json')
    complete = read(source/'COMPLETE.json')
    rows = prompts(source/'prompts/selected.jsonl')
    c = OnlineConfig()
    if (config['protocol'] != PROTOCOL or config['prompt_count'] != 50 or
            complete['status'] != 'complete' or complete['baseline_videos'] != 50 or
            sorted(config['selected_ids']) != sorted(r['sample_id'] for r in rows)):
        raise ValueError('requires a completed matching Wan21 VBench50 archive')
    if config['targets'] != list(c.evaluation_targets) or config['skip_budgets'] != [23,29,35]:
        raise ValueError('requires the existing 1.8/2.4/3.0 targets and K23/K29/K35')
    selected = select_evaluation(rows)
    sources = {}
    def record(path):
        path = Path(path).resolve(strict=True)
        sources[str(path)] = sha256(path)
        return path
    for name in ('config.json','COMPLETE.json','prompts/selected.jsonl'):
        record(source/name)
    profile = record(config['flops_profile'])
    if sources[str(profile)] != config['source_sha256'][str(profile)]:
        raise ValueError('source FLOPs profile changed')
    prepared = []
    for row in selected:
        sid = row['sample_id']
        shards = [key for key, ids in config['shard_ids'].items() if sid in ids]
        if len(shards) != 1:
            raise ValueError('baseline must have exactly one original GPU shard')
        base = source/'shards'/('gpu'+shards[0])/'baseline'
        manifest = read(record(base/'run.json'))
        marker = read(record(base/'COMPLETE.json'))
        components = read(record(base/'components.json'))
        matches = [x for x in manifest['prompts'] if x['sample_id'] == sid]
        if (manifest['protocol'] != PROTOCOL or manifest['method'] != 'baseline' or
                manifest['skip_budget'] is not None or manifest['policy_checkpoint'] is not None or
                manifest['flops_profile_sha256'] != sources[str(profile)] or
                marker['status'] != 'generation_complete' or
                marker['videos'] != len(manifest['prompts']) or
                len(matches) != 1 or matches[0]['prompt'] != row['prompt'] or
                not manifest['gpu_uuid'].startswith('GPU-')):
            raise ValueError('invalid native baseline identity: '+sid)
        measured = [x for x in components['rows'] if x['sample_id'] == sid]
        if len(measured) != 1:
            raise ValueError('missing or duplicate baseline measurement')
        measurement = {k:v for k,v in measured[0].items() if k != 'sample_id'}
        for key in ('generate_seconds','dit_tflops','t5_cuda_seconds','dit_cuda_seconds',
                    'vae_decode_cuda_seconds','estimated_t5_tflops_per_video',
                    'estimated_vae_decode_tflops_per_video'):
            if not math.isfinite(measurement[key]) or measurement[key] <= 0:
                raise ValueError('missing/nonfinite baseline component measurement')
        links = {name:record(base/sub/(sid+suffix)) for name,sub,suffix in (
            ('video.mp4','videos','.mp4'),('timing.json','timings','.json'),
            ('trace.json','traces','.json'))}
        timing, trace = read(links['timing.json']), read(links['trace.json'])
        if (timing['status'] != 'success' or timing['full_compute_forward_calls'] != 100 or
                timing['reuse_forward_calls'] != 0 or trace['step_reuse'] != 0 or
                trace['step_recompute'] != 50 or not math.isclose(measurement['generate_seconds'],
                    timing['pipeline_generate_wall_seconds'], rel_tol=1e-9)):
            raise ValueError('baseline timing/trace is inconsistent')
        probe=video_geometry(links['video.mp4'])
        prepared.append((dict(**row,baseline_gpu_uuid=manifest['gpu_uuid']),
                         measurement,links,manifest,base,probe))
    create_result(out, '# Online evaluation reference\n\nTwenty prompts sampled from the existing VBench50 with seed42; three targets, no VBench scoring. baselines/ contains small adapter metadata and symlinks to original video/timing/trace. manifest.json freezes source SHA and physical GPU pairing. No video generation.\n')
    (out/'baselines').mkdir()
    (out/'baselines/README.md').write_text('Native baseline adapters by prompt ID; large source files are symlinks. COMPLETE.json seals each adapter.\n')
    for row, measurement, links, manifest, base, probe in prepared:
        folder = out/'baselines'/row['sample_id'];folder.mkdir()
        (folder/'README.md').write_text('Reused VBench50 native baseline: video/timing/trace symlinks and original component measurements.\n')
        identity = dict(job=dict(kind='baseline',sample_id=row['sample_id'],prompt=row['prompt']),
                        source_baseline=str(base),protocol=PROTOCOL)
        dump(folder/'identity.json',identity)
        dump(folder/'generation.json',dict(identity=identity,protocol=PROTOCOL,
            gpu_uuid=row['baseline_gpu_uuid'],gpu_name=manifest['gpu'],
            wan_checkpoint=manifest['checkpoint_dir'],
            flops_profile_sha256=manifest['flops_profile_sha256'],
            measurement_scope='reused original complete pipeline.generate; excludes load/warmup/export'))
        dump(folder/'measurement.json',measurement)
        dump(folder/'ffprobe.json',probe)
        for name, path in links.items():(folder/name).symlink_to(path)
        seal(folder,['identity.json','generation.json','measurement.json','ffprobe.json',*links],identity=identity)
    data = dict(schema='ours21_online_eval20_reference_v1',protocol=PROTOCOL,
        source_run=str(source),selection_seed=42,
        selection_algorithm='random.Random(42).sample(sorted(source50, sample_id), 20); sort sample_id',
        targets=list(c.evaluation_targets),skip_budgets=[23,29,35],vbench_enabled=False,
        source_files=sources,rows=[x[0] for x in prepared])
    dump(out/'manifest.json',data)
    (out/'eval_prompts.jsonl').write_text(''.join(json.dumps(dict(sample_id=r['sample_id'],
        prompt=r['prompt']),ensure_ascii=False)+'\n' for r in data['rows']))
    seal(out,['manifest.json','eval_prompts.jsonl',
        *['baselines/'+r['sample_id']+'/COMPLETE.json' for r in data['rows']]],
        identity=dict(source_run=str(source),selection_seed=42))
    load_evaluation_bundle(out)
    print(out)
