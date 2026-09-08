"""Synthetic 12-mode CPU cache/train/checkpoint/live history validation; no Wan GPU job."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,sys,subprocess
from pathlib import Path
from dataclasses import replace
PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT),str(PROJECT/'tests'),str(PROJECT/'experiments/rl_validation_v1')]
from validate import fixture
from ours4wan21.contracts import create_result,dump,sha256,MODEL_ROOT,FEATURE_MODES,TrainingConfig
from ours4wan21.feature_cache import extract
from ours4wan21.latent_features import GROUPS,contract
from ours4wan21.data import build
from ours4wan21.train import train
from ours4wan21.policy import Policy
from ours4wan21.runtime import Controller,apply_policy
from ours4wan21.overhead import predictor_fields
from types import SimpleNamespace
from unittest.mock import patch
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    out=create_result(a.output_dir,'# Twelve-mode CPU synthetic validation\n\nunit_tests.txt contains regressions; synthetic_collection and features contain miniature fixtures; per-mode result links contain two-epoch smoke runs. Models are under workspace/models; VALIDATION.json records provenance. No production data or GPU timings.')
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(PROJECT/'tests'),'-v'],capture_output=True,text=True)
    (out/'unit_tests.txt').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError(tests.stderr)
    paths=fixture(out);torch.manual_seed(42);torch.set_num_threads(1)
    sigmas=torch.linspace(.99,.01,50);latents=torch.randn(50,16,2,4,4)*.2
    features_root=out/'features';features_root.mkdir();(features_root/'README.md').write_text('# Synthetic miniature features\n\nFour trajectories; full ten-group features. Not production training data.\n')
    entries={}
    for path in paths:
        completion=json.loads(path.read_text());row=completion['trajectory_row'];trace_path=Path(row['trace_json']);trace=json.loads(trace_path.read_text());records=[]
        for t in range(50):
            latent_path=path.parent/f'input_{t}.pt';torch.save(latents[t].half(),latent_path)
            records.append(dict(step_index=t,model_stage='single',sigma=float(sigmas[t]),latent_path=str(latent_path),latent_shape=list(latents[t].shape),latent_dtype='torch.float16'))
        trace['step_records']=records;dump(trace_path,trace)
        actions=torch.tensor([int(r['action']=='reuse') for r in trace['decisions'][::2]])
        features,raw=extract(records,actions,strict_shape=False)
        target=features_root/f'{row["trajectory_id"]}.pt'
        torch.save(dict(trajectory_id=row['trajectory_id'],sample_id=row['sample_id'],split=row['split'],trace_sha256=sha256(trace_path),actions=actions,features=features,raw_inputs=raw),target)
        entries[row['trajectory_id']]=dict(file=target.name,sha256=sha256(target))
    dump(features_root/'index.json',dict(contracts={g:contract(g) for g in GROUPS},rows=entries))
    dump(features_root/'COMPLETE.json',dict(index_sha256=sha256(features_root/'index.json')))
    results=[]
    for mode in ('scalar5','sea7',*FEATURE_MODES):
        b=build(paths,mode,features_root if mode in FEATURE_MODES else None)
        destination=out/f'{out.name}_{mode}'
        ckpt=train(b,mode,destination,MODEL_ROOT/out.name/mode,config=replace(TrainingConfig(),epochs=2),device='cpu',smoke=True)
        policy=Policy(ckpt,device='cpu',state_mode=mode,allow_smoke=True)
        # Exercise the actual wrapper that captures raw x BEFORE upstream forward.
        pipe=SimpleNamespace(model=SimpleNamespace())
        controller=apply_policy(pipe,policy,25);controller.set_scheduler_sigmas(sigmas)
        import ours4wan21.runtime as runtime
        def forward(model,x,**kw):
            t=kw['seacache_step_index'];branch=kw['seacache_branch'];feature=torch.arange(32).float().reshape(1,8,4)+1+t*.1
            reuse=controller.plan_step(branch=branch,step_index=t,num_steps=50,feature=feature,grid_size=torch.tensor([2,2,2]))
            if reuse:controller.reuse_residual(branch,t)
            else:controller.record_recompute(branch,t,torch.ones_like(feature))
            return reuse
        with patch.object(runtime.integration_functions(),'seacache_forward',forward):
            for t in range(50):
                for branch in ('cond','uncond'):
                    pipe.model.forward([latents[t]],seacache_branch=branch,seacache_step_index=t)
        trace=controller.summary();assert trace['step_reuse']==25
        assert controller.feature_overhead()['calls']==(50 if mode in FEATURE_MODES else 0)
        timing=dict(predictor=policy.overhead_summary(),pipeline_generate_wall_seconds=1.)
        overhead=predictor_fields(timing,trace)
        payload=torch.load(ckpt,map_location='cpu',weights_only=False)
        assert payload['model_config']['input_dim']==b['tensors']['state'].shape[1]
        from ours4wan21.local_iql import compute_normalizer
        expected=compute_normalizer(b['tensors']['state'],b['train_indices'].tolist())
        for key in expected:torch.testing.assert_close(expected[key],payload['normalizer'][key],atol=0,rtol=0)
        results.append(dict(mode=mode,input_dim=payload['model_config']['input_dim'],checkpoint=str(ckpt),sha256=sha256(ckpt),predictor_profile=policy.flops_profile,overhead=overhead))
    dump(out/'VALIDATION.json',dict(status='pass',groups=results,epochs_per_group=2,gpu_measured=False,synthetic=True,source_sha256={str(f.relative_to(PROJECT)):sha256(f) for f in sorted((PROJECT/'ours4wan21').glob('*.py'))}))

if __name__=='__main__':main()
