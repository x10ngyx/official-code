"""Run one explicitly authorized aggressive Dynamics128 IQL experiment to completion."""
import argparse
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from ours4wan21.contracts import EXP_ROOT, MODEL_ROOT, create_result, dump, sha256, training_config

NAME='ours21_dynamics128_aggressive_iql_v1'
MODE='sea7_dynamics_raw_sea128'
CACHE=EXP_ROOT/'ours21_random3000_12groups_v1_sea7_dynamics_raw_sea128_cache'
BASELINE=EXP_ROOT/'ours21_random3000_12groups_v1_sea7_dynamics_raw_sea128_train'
ROOT=EXP_ROOT/NAME
TRAINING=EXP_ROOT/(NAME+'_train')
ANALYSIS=EXP_ROOT/(NAME+'_analysis')
WEIGHTS=MODEL_ROOT/NAME


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=3)
    args=parser.parse_args()
    if args.gpu not in (0,1,2,3):parser.error('select a physical GPU index 0–3')
    if 'wan2.2' not in Path(sys.prefix).name.lower():raise ValueError('use the Wan2.2 environment')
    info=subprocess.check_output(['nvidia-smi',f'--id={args.gpu}',
        '--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    uuid,memory,util=[s.strip() for s in info]
    if int(memory)>1500 or int(util)>10:raise RuntimeError('selected GPU is busy; do not displace another job')
    # This task owns one advisory per-GPU lock, in addition to the one-GPU visibility contract.
    lock=(EXP_ROOT/f'.ours21_gpu{args.gpu}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for path in (ROOT,TRAINING,ANALYSIS,WEIGHTS):
        if path.exists() or path.is_symlink():raise FileExistsError(f'fresh experiment required: {path}')
    marker=json.loads((CACHE/'COMPLETE.json').read_text())
    dataset_hash=sha256(CACHE/'transitions.pt')
    if marker['status']!='complete' or dataset_hash!=marker['sha256']:raise ValueError('cache hash mismatch')
    manifest=json.loads((CACHE/'manifest.json').read_text())
    if manifest!=json.loads((BASELINE/'dataset_manifest.json').read_text()):raise ValueError('not original Dynamics128 data')
    assert manifest['trajectories']==3000 and manifest['state']['mode']==MODE
    assert manifest['selection_sha256']=='f98653e3db6a8ac9b03567744bcd679810705628949c899bc4655653a2862208'
    sources={str(p):sha256(p) for p in [*sorted((PROJECT/'ours4wan21').glob('*.py')),
        PROJECT/'local_training_lock.json',*sorted(Path(__file__).parent.glob('*.py')),Path(__file__).with_name('README.md')]}
    config=dict(schema='ours21_aggressive_iql_experiment_v1',state_mode=MODE,
        iql_profile='aggressive_v1',training=asdict(training_config('aggressive_v1')),
        baseline_training=asdict(training_config()),fresh_random_initialization=True,
        dataset=str(CACHE),dataset_sha256=dataset_hash,selection_sha256=manifest['selection_sha256'],
        baseline_result=str(BASELINE),gpu=args.gpu,gpu_uuid=uuid,python=sys.executable,
        source_sha256=sources,scope='400 epochs, training/Q diagnostics, post300 stable checkpoint, frozen online VBench20 K23/K29/K35, no VBench score')
    create_result(ROOT,'# Dynamics128 aggressive IQL experiment\n\nconfig.json freezes the three-parameter comparison. training/ and analysis/ link to external result directories; cache/ reuses the original immutable cache. evaluation20/ contains the three-target frozen online20 test with no VBench score. logs/ stores process output; STATUS.json and COMPLETE.json describe actual completion.')
    for name,path in [('training',TRAINING),('analysis',ANALYSIS),('cache',CACHE),('model_weights',WEIGHTS)]:
        (ROOT/name).symlink_to(path.resolve(),target_is_directory=True)
    (ROOT/'logs').mkdir();(ROOT/'logs/README.md').write_text('# Process output\n\ntrain.log and analysis.log record the respective subprocesses. The root COMPLETE.json is the final completion contract.\n')
    dump(ROOT/'config.json',config)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
             MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    start=time.time()

    def execute(stage,command):
        dump(ROOT/'STATUS.json',dict(status='running',stage=stage,pid=os.getpid(),gpu=args.gpu,
            started_at=start,updated_at=time.time(),command=command))
        with (ROOT/'logs'/f'{stage}.log').open('x') as stream:
            subprocess.run(command,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

    try:
        execute('train',[sys.executable,'train.py','--dataset',str(CACHE),'--state-mode',MODE,
            '--output-dir',str(TRAINING),'--checkpoint-dir',str(WEIGHTS),'--device','cuda',
            '--training-seed','42','--iql-profile','aggressive_v1'])
        done=json.loads((TRAINING/'TRAINING_COMPLETE.json').read_text())
        rows=[json.loads(s) for s in (TRAINING/'epoch_metrics.jsonl').read_text().splitlines()]
        assert done['status']=='complete' and done['epochs']==400 and not done['smoke_only']
        assert [r['epoch'] for r in rows]==list(range(1,401))
        assert all(math.isfinite(v) for r in rows for s in ('train','val') for v in r[s].values())
        assert len(list((WEIGHTS/'checkpoints').glob('epoch_*.pt')))==400
        assert sha256(WEIGHTS/'checkpoints/epoch_400.pt')==done['final_sha256']
        actual=json.loads((TRAINING/'config.json').read_text())
        assert all(actual[k]==v for k,v in config['training'].items())
        assert actual['iql_profile']=='aggressive_v1' and not actual['smoke_only']
        assert json.loads((TRAINING/'dataset_manifest.json').read_text())==manifest
        execute('analysis',[sys.executable,'analyze_training.py','--training-result',str(TRAINING),
            '--dataset',str(CACHE),'--checkpoint-dir',str(WEIGHTS),'--output-dir',str(ANALYSIS),'--device','cuda'])
        selected=json.loads((ANALYSIS/'checkpoint_selection.json').read_text())
        assert selected['status']=='selected' and sha256(selected['checkpoint'])==selected['checkpoint_sha256']
        execute('extra_analysis',[sys.executable,str(Path(__file__).with_name('analyze_extra.py'))])
        # Training subprocess has exited; all-GPU evaluation owns its own idle-device checks.
        execute('evaluation20',[sys.executable,str(Path(__file__).with_name('evaluate20.py'))])
        evaluation=json.loads((ROOT/'evaluation20/COMPLETE.json').read_text())
        assert evaluation['status']=='complete' and evaluation['candidates']==60
        changed=[p for p,h in sources.items() if sha256(p)!=h]
        assert not changed,('source changed during experiment',changed)
        assert sha256(CACHE/'transitions.pt')==dataset_hash
        summary=dict(status='complete',epochs=400,checkpoints=400,elapsed_seconds=time.time()-start,
            training_seconds=rows[-1]['elapsed_seconds'],iql_profile='aggressive_v1',
            selected_epoch=selected['checkpoint_epoch'],selected_checkpoint=selected['checkpoint'],
            selected_checkpoint_sha256=selected['checkpoint_sha256'],final_sha256=done['final_sha256'],
            dataset_sha256=dataset_hash,final_train_metrics=rows[-1]['train'],final_val_metrics=rows[-1]['val'],
            baseline_weighted_losses_directly_comparable=False,
            evaluation20=evaluation,validation_scope='frozen online evaluation20; three fixed K targets; no VBench score')
        (ROOT/'READOUT.md').write_text(f'''# Dynamics128 aggressive IQL readout

Completed 400 epochs and 400 checkpoints with tau=.9, beta=3, weight_max=100.
All other training settings and the frozen Dynamics128 dataset match the original seed42 run.
Training elapsed: {rows[-1]['elapsed_seconds']:.1f} seconds.
Post300 validation-only selected epoch: {selected['checkpoint_epoch']}.

Final model: model_weights/final_model.pt. Selected model: analysis/selected_model.pt.
analysis/training_metrics.csv/png/svg and post300_* retain the complete diagnostics.
Changing expectile and actor weights changes the loss definitions: raw actor/V/Q losses are
not direct rankings against the baseline objective. Whether later skip concentration or video
quality is reported in evaluation20/RESULTS.md using the frozen online20 prompts and same-GPU baselines.
See analysis/extra/ for full-400-epoch mean_Q/Q-IQR and training-comparison figures.
''')
        with (ROOT/'READOUT.md').open('a') as report:
            report.write('\n'+(ANALYSIS/'extra/README.md').read_text())
            report.write('\n'+(ROOT/'evaluation20/RESULTS.md').read_text())
        dump(ROOT/'COMPLETE.json',summary);dump(ROOT/'STATUS.json',summary)
        print(json.dumps(summary),flush=True)
    except BaseException as exc:
        dump(ROOT/'FAILED.json',dict(error=repr(exc),time=time.time()))
        dump(ROOT/'STATUS.json',dict(status='failed',error=repr(exc),time=time.time()))
        raise


if __name__=='__main__':main()
