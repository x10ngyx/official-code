"""Portable prompt-pool construction and measured Wan21 K calibration export."""
from pathlib import Path
import json
import math
import random

from .contracts import EXP_ROOT, PROTOCOL, create_result, dump, sha256, under
from .online_common import OnlineConfig, prompts, isolate_prompts, text_key, require_environment, read


def build_pool(args):
    require_environment()
    import torch
    data=Path(args.dataset)
    if data.is_dir():data=data/'transitions.pt'
    bundle=torch.load(data,map_location='cpu',weights_only=False)
    registry=prompts(args.prompt_registry);evaluation=prompts(args.eval_prompts)
    split=bundle['manifest']['prompt_splits']
    by_id={r['sample_id']:r['prompt'] for r in registry}
    if not set(split).issubset(by_id):raise ValueError('registry does not cover offline prompt IDs')
    excluded_ids={sid for sid,s in split.items() if s!='train'}|{r['sample_id'] for r in evaluation}
    excluded_texts={text_key(by_id[sid]) for sid in split if split[sid]!='train'}|{text_key(r['prompt']) for r in evaluation}
    pool=[r for r in registry if r['sample_id'] not in excluded_ids and text_key(r['prompt']) not in excluded_texts]
    config=OnlineConfig()
    if len(pool)<config.prompt_pool_size:
        raise ValueError("registry has fewer than 3000 eligible prompts; supply the full source registry")
    pool=random.Random(config.plan_seed).sample(pool,config.prompt_pool_size)
    isolate_prompts(pool,evaluation,registry,split)
    out=create_result(args.output_dir,'# Online prompt pool\n\ntrain_prompts.jsonl contains registered training and unseen prompts, excluding offline validation/test and evaluation20. sources.json seals inputs; review count before preparation.')
    (out/'train_prompts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pool))
    dump(out/'sources.json',dict(pool_size=len(pool),registry_count=len(registry),
        dataset_sha256=sha256(data),registry_sha256=sha256(args.prompt_registry),
        evaluation_sha256=sha256(args.eval_prompts),pool_sha256=sha256(out/'train_prompts.jsonl')))
    print(out/'train_prompts.jsonl')


def calibration_from_runs(baseline, candidates):
    base=Path(baseline);bm=read(base/'run.json')
    if bm['protocol']!=PROTOCOL or bm['method']!='baseline':raise ValueError('requires native resident Wan21 baseline')
    def rows(directory,manifest):
        c=read(directory/'COMPLETE.json')
        values=read(directory/'components.json')['rows']
        if c['videos']!=len(manifest['prompts']) or len(values)!=c['videos']:
            raise ValueError('incomplete calibration generation')
        if {x['sample_id'] for x in values}!={x['sample_id'] for x in manifest['prompts']}:
            raise ValueError('calibration prompt IDs differ')
        if any(not math.isfinite(x['generate_seconds']) or x['generate_seconds']<=0 for x in values):
            raise ValueError('invalid calibration timing')
        return values
    br=rows(base,bm);entries=[]
    for directory in map(Path,candidates):
        cm=read(directory/'run.json')
        for key in ('protocol','prompts','checkpoint_dir','gpu_uuid'):
            if bm[key]!=cm[key]:raise ValueError('calibration baseline/candidate mismatch: '+key)
        if cm['method']!='ours' or type(cm['skip_budget']) is not int:
            raise ValueError('calibration requires explicit-K Ours inference runs')
        cr=rows(directory,cm)
        entries.append(dict(skip_budget=cm['skip_budget'],
            calibrated_speedup=sum(x['generate_seconds'] for x in br)/sum(x['generate_seconds'] for x in cr),
            source=str(directory.resolve()),components_sha256=sha256(directory/'components.json')))
    entries.sort(key=lambda r:r['skip_budget'])
    if not entries or len({r['skip_budget'] for r in entries})!=len(entries):raise ValueError('empty or duplicate K samples')
    if any(not 0<=r['skip_budget']<=48 for r in entries):raise ValueError('invalid K')
    if any(a['calibrated_speedup']>b['calibrated_speedup'] for a,b in zip(entries,entries[1:])):
        raise ValueError('measured mapping is not monotone; inspect or repeat calibration, do not silently smooth')
    return dict(schema='ours4wan21_speed_to_k_v1',status='calibrated',protocol=PROTOCOL,
        forced_steps=[0,49],entries=entries,baseline=str(base.resolve()),
        baseline_components_sha256=sha256(base/'components.json'),
        note='Measured ratio of summed complete generate times; discrete nearest-K mapping, no interpolation. Timing depends on policy and GPU; targets are requests, not achieved speeds.')


def build_calibration(args):
    payload=calibration_from_runs(args.baseline_dir,args.candidate_dirs)
    out=create_result(args.output_dir,'# Measured Wan21 speedup to K\n\ncalibration.json records matched fixed-protocol timing evidence and discrete budgets. No theoretical Wan22 mapping is reused.')
    dump(out/'calibration.json',payload);print(out/'calibration.json')
