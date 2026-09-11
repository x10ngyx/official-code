"""Pure-Python plan, provenance and resumable artifact contracts."""
import hashlib
import json
import os
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[2]
PACKAGES = PROJECT.parent
WORKSPACE = PACKAGES.parents[1]
EXP_ROOT = Path('/all/yiran07-disk3/huteng_data/exp')
PROTOCOL = dict(model='Wan2.1-T2V-1.3B', width=832, height=480, frames=81,
                fps=16, steps=50, solver='unipc', shift=5, cfg=5, seed=42,
                dit_compute_dtype='bfloat16', offload_model=False, t5_cpu=False,
                batch_size=1, prompt_extension=False, other_steps='recompute',
                cache='branch_local_transformer_stack_residual',
                last_step_skip_allowed=True, schema='single_skip_dataset_v2',
                latent_storage='50_pre_action_fp16_tensors', table_schema='trajectory_step_branch_v1')
COARSE = [2] + list(range(5, 51, 5))  # one-based execution order, not diffusion t
REMAINING = [s for s in range(2, 51) if s not in COARSE]


def steps(stage):
    return {'coarse': COARSE, 'remaining': REMAINING,
            'all': COARSE + REMAINING}[stage]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def prompts(path, expected=40):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    result = []
    for row in rows:
        sid = row.get('sample_id')
        text = row.get('prompt_en', row.get('prompt'))
        if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', sid):
            raise ValueError('sample_id must be a safe ASCII identifier')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('empty prompt')
        entry=dict(sample_id=sid,prompt=text)
        if 'split' in row:
            if row['split'] not in ['train','val','test','unassigned']:
                raise ValueError('split must be train/val/test/unassigned')
            entry['split']=row['split']
        result.append(entry)
    if len(result) != expected or len({r['sample_id'] for r in result}) != len(result):
        raise ValueError(f'require exactly {expected} distinct sample IDs')
    if len({r['prompt'] for r in result}) != len(result):
        raise ValueError('duplicate prompt text')
    return result


def job_plan(rows, stage):
    # Complete ALL baselines, then each step across ALL prompts.
    return [(0, row) for row in rows] + [(s, row) for s in steps(stage) for row in rows]


def cell_path(output, step, sid):
    if step == 0:
        return Path(output)/'shared_baselines'/sid
    return Path(output)/'shards/shard_00/candidates'/trajectory_id(step,sid)


def trajectory_id(step, sid):
    return f'{sid}__skip_{step:02d}' if step else f'{sid}__baseline'


def external(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(EXP_ROOT.resolve()) or path == EXP_ROOT.resolve():
        raise ValueError(f'output must be a subdirectory of {EXP_ROOT}')
    return path


def source_hashes():
    files = list((PROJECT/'experiments/single_skip_v1').glob('*.py'))
    for folder in ['SeaCache4Wan21', 'ComponentMetrics', 'VideoMetrics/video_metrics', 'VbenchEvaluation']:
        files += list((PACKAGES/folder).glob('*.py'))
    files += list((PACKAGES/'VbenchEvaluation').glob('*.sh'))
    files += list((PACKAGES/'VbenchEvaluation').glob('*.json'))
    files += [PACKAGES/'Wan21Benchmark/protocol.py',
              PACKAGES/'DiCache4Wan21/upstream_lock.json']
    return {str(p.relative_to(PACKAGES)): sha(p) for p in sorted(files)}


def completed(path, contract_hash, *, verify=True):
    path = Path(path)
    marker = path/'COMPLETE.json'
    if not marker.exists():
        return False
    info = read_json(marker)
    if info['contract_hash'] != contract_hash:
        raise ValueError(f'foreign cell contract: {path}')
    for name, expected in info['files'].items():
        f = path/name
        if not f.is_file() or (verify and sha(f) != expected):
            raise ValueError(f'corrupt completed artifact: {f}')
    return True


def validate_trace(trace, skip_step, blocks=30):
    calls = trace['decisions']
    if len(calls) != 100:
        raise ValueError('must have 100 ordered CFG calls')
    for i, row in enumerate(calls):
        expected_skip = skip_step != 0 and i//2 == skip_step-1
        if (row['step_index'], row['branch']) != (i//2, ('cond','uncond')[i%2]):
            raise ValueError('CFG branch or step order mismatch')
        expected = 'reuse' if expected_skip else 'recompute'
        if row['action'] != expected or row['execution'] != expected:
            raise ValueError('unexpected cache action')
        if expected_skip and row['cache_source_step_index'] != i//2-1:
            raise ValueError('cache age must be exactly one')


def validate_timing(timing, skip_step):
    if timing['status'] != 'success' or len(timing['calls']) != 100:
        raise ValueError('incomplete component timing')
    for i, row in enumerate(timing['calls']):
        expected = 0 if skip_step and i//2 == skip_step-1 else 30
        if row['blocks_executed'] != expected:
            raise ValueError('actual Transformer execution disagrees with intervention')
