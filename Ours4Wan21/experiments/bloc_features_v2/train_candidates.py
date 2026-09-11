"""Equal 400-epoch training and validation-only checkpoint selection for BLOC."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from ours4wan21.contracts import EXP_ROOT, MODEL_ROOT, TrainingConfig, create_result, dump, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-root',type=Path,required=True)
    p.add_argument('--suite-name',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--groups',nargs='+',choices=('a','ab','ac','abc'),default=['a','ab','ac','abc'])
    a=p.parse_args()
    if a.seed<0 or len(set(a.groups))!=len(a.groups):
        raise ValueError('invalid seed/groups')
    if not (a.feature_root/'COMPLETE.json').is_file():
        raise ValueError('feature extraction and cache assembly must be complete before training')
    out=create_result(EXP_ROOT/(a.suite_name+'_training'), '# BLOC training orchestration\n\n'
        'config.json freezes common training settings, input cache hashes and code. logs/ contains '
        'per-group subprocess output. COMPLETE.json requires each full 400-epoch training and selection.')
    (out/'logs').mkdir()
    source_paths=[PROJECT/'ours4wan21'/name for name in
                  ('train.py','analysis.py','contracts.py','bloc_features.py','local_iql.py')]
    sources={str(p):sha256(p) for p in source_paths}
    config=dict(groups=a.groups,training_config=dict(asdict(TrainingConfig()),seed=a.seed),
                feature_root=str(a.feature_root),feature_complete_sha256=sha256(a.feature_root/'COMPLETE.json'),
                source_sha256=sources,selection_rule='same post300 two-sided actor/Q stability rule for every group; no test data',
                note='feature ranking requires subsequent validation closed-loop rollout; this marker is training completion only')
    dump(out/'config.json',config)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
             CUDA_DEVICE_ORDER='PCI_BUS_ID')
    def worker(job):
        gpu,group=job
        mode='sea7_bloc_'+group
        cache=a.feature_root.with_name(a.feature_root.name+'_'+group+'_cache')
        name=f'{a.suite_name}_{group}_seed{a.seed}'
        training,weights,analysis=EXP_ROOT/(name+'_train'),MODEL_ROOT/name,EXP_ROOT/(name+'_analysis')
        if any(sha256(path)!=digest for path,digest in sources.items()):
            raise ValueError('training source changed after freeze')
        cmds=[['train.py','--dataset',str(cache),'--state-mode',mode,'--output-dir',str(training),
               '--checkpoint-dir',str(weights),'--device','cuda','--training-seed',str(a.seed)],
              ['analyze_training.py','--dataset',str(cache),'--training-result',str(training),
               '--checkpoint-dir',str(weights),'--output-dir',str(analysis),'--device','cuda']]
        for label,cmd in zip(('train','analysis'),cmds):
            with (out/'logs'/f'{group}_{label}.log').open('w') as log:
                subprocess.run([sys.executable,*cmd],cwd=PROJECT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        marker=json.loads((training/'TRAINING_COMPLETE.json').read_text())
        if marker.get('epochs')!=400 or marker.get('smoke_only'):
            raise ValueError('incomplete production training')
        checkpoint=analysis/'selected_model.pt'
        return dict(group=group,mode=mode,seed=a.seed,training=str(training),analysis=str(analysis),
                    checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                    cache=str(cache),cache_sha256=sha256(cache/'transitions.pt'))
    with ThreadPoolExecutor(max_workers=min(4,len(a.groups))) as pool:
        rows=list(pool.map(worker,enumerate(a.groups)))
    if any(sha256(path)!=digest for path,digest in sources.items()):
        raise ValueError('training source changed during run')
    dump(out/'COMPLETE.json',dict(status='training_and_checkpoint_selection_complete',rows=rows,
                                 config_sha256=sha256(out/'config.json')))


if __name__=='__main__':
    main()
