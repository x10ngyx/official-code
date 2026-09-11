"""Frozen two-feature/four-level experiment definitions and evidence checks."""
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from ours4wan21.contracts import EXP_ROOT,MODEL_ROOT,OFFICIAL,PROTOCOL,WORKSPACE,training_config,dump,sha256
from ours4wan21.online_reference import load_evaluation_bundle,gpu_uuids

NAME='ours21_iql_aggressiveness_2x4_v1'
ROOT=EXP_ROOT/NAME
LEGACY=EXP_ROOT/'ours21_dynamics128_aggressive_iql_v1'
REFERENCE=EXP_ROOT/'ours21_online_eval20_from_vbench50_random42_v1'
OLD=EXP_ROOT/'ours21_dynamics128_e391_vbench50_random42_4gpu_v1'
FEATURES={'sea7':'sea7','dynamics128':'sea7_dynamics_raw_sea128'}
LEVELS={'a1':'aggressive_a1_v1','a2':'aggressive_a2_v1','a3':'aggressive_v1','a4':'aggressive_a4_v1'}
TARGETS=[1.8,2.4,3.0];BUDGETS=[23,29,35]
FIELDS=('generate_seconds','dit_cuda_seconds','t5_cuda_seconds','vae_decode_cuda_seconds',
        'dit_tflops','estimated_t5_tflops_per_video','estimated_vae_decode_tflops_per_video',
        'predictor_tflops','predictor_network_cuda_seconds','predictor_decision_wall_seconds','latent_feature_wall_seconds')
METRICS=('psnr_rgb_db','ssim_rgb','lpips_alex_v0_1_spatial')

def read(p):return json.loads(Path(p).read_text())

def writecsv(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def group(feature,level):
    name=f'{feature}_{level}';mode=FEATURES[feature]
    prefix='ours21_random3000_12groups_v1_'+mode
    adopted=name=='dynamics128_a3'
    train_name='ours21_dynamics128_aggressive_iql_v1' if adopted else NAME+'_'+name
    return dict(name=name,feature=feature,level=level,mode=mode,profile=LEVELS[level],
        training_config=asdict(training_config(LEVELS[level])),adopted=adopted,
        cache=str(EXP_ROOT/(prefix+'_cache')),baseline=str(EXP_ROOT/(prefix+'_train')),
        baseline_analysis=str(EXP_ROOT/(prefix+'_analysis')),baseline_weights=str(MODEL_ROOT/prefix),
        training=str(EXP_ROOT/(train_name+'_train')),weights=str(MODEL_ROOT/train_name),
        analysis=str(EXP_ROOT/(NAME+'_'+name+'_analysis')))

def freeze_prompts(reference,uuids):
    assert reference['targets']==TARGETS and reference['skip_budgets']==BUDGETS
    rng=random.Random(42);selected=set()
    for uuid,n in zip(uuids,[3,3,2,2]):
        pool=[r['sample_id'] for r in reference['rows'] if r['baseline_gpu_uuid']==uuid]
        selected.update(rng.sample(pool,n))
    rows=[r for r in reference['rows'] if r['sample_id'] in selected]
    assert len(rows)==10 and len(selected)==10
    return rows

def environment(uuid=None):
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        TORCH_HOME=str(MODEL_ROOT/'torch-cache'),WAN22_PYTHON=sys.executable)
    if uuid is not None:env['CUDA_VISIBLE_DEVICES']=uuid
    return env

def verify_training(g):
    training=Path(g['training']);weights=Path(g['weights'])
    done=read(training/'TRAINING_COMPLETE.json');actual=read(training/'config.json')
    assert done['status']=='complete' and done['epochs']==400 and not done['smoke_only']
    assert actual['iql_profile']==g['profile'] and all(actual[k]==v for k,v in g['training_config'].items())
    rows=[json.loads(s) for s in (training/'epoch_metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows]==list(range(1,401))
    assert all(math.isfinite(v) for r in rows for split in ('train','val') for v in r[split].values())
    assert len(list((weights/'checkpoints').glob('epoch_*.pt')))==400
    assert sha256(weights/'checkpoints/epoch_400.pt')==done['final_sha256']
    assert read(training/'dataset_manifest.json')==read(Path(g['cache'])/'manifest.json')
    return done

def verify_sources(config):
    changed=[p for p,h in config['source_hashes'].items() if sha256(p)!=h]
    assert not changed,('frozen source changed',changed)

def audit_trace(trace,timing,k):
    assert trace['total_steps']==50 and len(trace['decisions'])==len(timing['calls'])==100
    paths=[]
    for offset,branch in enumerate(('cond','uncond')):
        ds=[d for d in trace['decisions'] if d['branch']==branch]
        assert [d['step_index'] for d in ds]==list(range(50))
        a=[int(d['action']=='reuse') for d in ds];assert sum(a)==k
        for i,v in enumerate(a):assert timing['calls'][2*i+offset]['blocks_executed']==(0 if v else 30)
        paths.append(a)
    assert paths[0]==paths[1] and trace['step_reuse']==k
    run=best=0
    for v in paths[0][25:]:run=run+1 if v else 0;best=max(best,run)
    return dict(late25=sum(paths[0][25:]),longest_late25=best,skip_path=''.join(map(str,paths[0])))
