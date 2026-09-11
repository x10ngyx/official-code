import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'experiments/single_skip_v1'))
from common import (COARSE, REMAINING, PROTOCOL, cell_path, completed, digest,
                    job_plan, prompts, sha, validate_trace, validate_timing, write_json)


class PlanTests(unittest.TestCase):
    def test_coarse_then_step_major_completion(self):
        self.assertEqual(COARSE,[2,5,10,15,20,25,30,35,40,45,50])
        self.assertEqual(len(REMAINING),38)
        self.assertEqual(sorted(COARSE+REMAINING),list(range(2,51)))
        rows=[dict(sample_id=f'p{i}',prompt=f'prompt {i}') for i in range(40)]
        jobs=job_plan(rows,'all')
        self.assertEqual(len(jobs),2000)
        self.assertEqual([s for s,_ in jobs[:40]],[0]*40)
        self.assertEqual([s for s,_ in jobs[40:80]],[2]*40)
        self.assertEqual(jobs[480][0],3)
        self.assertEqual(len({(s,r['sample_id']) for s,r in jobs}),2000)

    def test_safe_prompts(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'p.jsonl'
            p.write_text(json.dumps(dict(sample_id='../bad',prompt='hello')))
            with self.assertRaises(ValueError):prompts(p,1)
            p.write_text(json.dumps(dict(sample_id='p1',prompt_en='hello')))
            self.assertEqual(prompts(p,1)[0]['prompt'],'hello')
            with self.assertRaises(ValueError):prompts(p,40)

    def test_resume_rejects_tampering_and_foreign_contract(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'data').write_text('good')
            write_json(p/'COMPLETE.json',dict(contract_hash='a',files={'data':sha(p/'data')}))
            self.assertTrue(completed(p,'a'))
            with self.assertRaises(ValueError):completed(p,'b')
            (p/'data').write_text('bad')
            with self.assertRaises(ValueError):completed(p,'a')


class ControllerTests(unittest.TestCase):
    def test_shared_forward_executes_single_skip_and_natural_suffix(self):
        import types
        import torch
        from runtime import SingleSkipController
        from wan21_integration import seacache_forward
        # Exercise the actual shared forward with tiny CPU DiT components.
        # The only mocked Wan symbol is the parameter-free sinusoidal embedding.
        module=types.ModuleType('wan.modules.model')
        module.sinusoidal_embedding_1d=lambda dim,t:t.float().reshape(-1,1).repeat(1,dim)
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.modulation=torch.zeros(1,6,2)
                self.norm1=torch.nn.Identity(); self.calls=0
            def forward(self,x,**kwargs):
                self.calls+=1
                return x+.01*torch.sin(x)+.001*kwargs['context'].mean()
        class Head(torch.nn.Module):
            def forward(self,x,e):return x+e[:,None,:]
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.model_type='t2v';self.dim=2;self.freq_dim=2;self.text_len=1
                self.patch_embedding=torch.nn.Conv3d(2,2,1,bias=False)
                with torch.no_grad():self.patch_embedding.weight.copy_(torch.eye(2).reshape(2,2,1,1,1))
                self.time_embedding=torch.nn.Identity();self.time_projection=torch.nn.Linear(2,12,bias=False)
                with torch.no_grad():self.time_projection.weight.zero_()
                self.text_embedding=torch.nn.Identity();self.freqs=torch.ones(1)
                self.blocks=torch.nn.ModuleList([Block() for _ in range(30)]);self.head=Head()
            def unpatchify(self,x,grid):return [x[0].transpose(0,1).reshape(2,1,2,2)]
        with patch.dict(sys.modules,{'wan.modules.model':module}), torch.no_grad():
            for skip in [0,2,50]:
                model=Model();ctrl=SingleSkipController(skip);model.seacache_controller=ctrl
                cached={};latent=torch.full((2,1,2,2),.1)
                for step in range(50):
                    outputs=[]
                    for j,branch in enumerate(['cond','uncond']):
                        context=torch.full((1,2),float(j+1))
                        tokens=latent.flatten(1).transpose(0,1).unsqueeze(0)
                        if skip and step==skip-1:
                            expected=tokens+cached[branch]
                        else:
                            expected=tokens.clone()
                            for _ in range(30):expected=expected+.01*torch.sin(expected)+.001*context.mean()
                            cached[branch]=expected-tokens
                        expected=expected+float(step)
                        result=seacache_forward(model,[latent],t=torch.tensor([float(step)]),context=[context],
                            seq_len=4,seacache_branch=branch,seacache_step_index=step,seacache_num_steps=50)[0]
                        self.assertTrue(torch.equal(result,expected[0].transpose(0,1).reshape_as(latent)))
                        outputs.append(result)
                    # Propagate intervention into later inputs; never restore baseline latents.
                    latent=latent-.001*(outputs[1]+5*(outputs[0]-outputs[1]))
                validate_trace(ctrl.summary(),skip)
                self.assertTrue(all(b.calls==(100 if skip==0 else 98) for b in model.blocks))

    def test_every_intervention_including_final_and_cfg_cache_age(self):
        import torch
        from runtime import SingleSkipController
        grid=torch.tensor([1,2,2])
        for skip in [0]+list(range(2,51)):
            ctrl=SingleSkipController(skip)
            ctrl.set_scheduler_sigmas(torch.linspace(1,0,51))
            for i in range(50):
                for j,branch in enumerate(['cond','uncond']):
                    feature=torch.full((1,4,2),1.+i+j)
                    reuse=ctrl.plan_step(branch=branch,step_index=i,num_steps=50,feature=feature,grid_size=grid)
                    if reuse:
                        self.assertTrue(torch.equal(ctrl.reuse_residual(branch,i),torch.full((1,4,2),100.*j+i-1)))
                    else:ctrl.record_recompute(branch,i,torch.full((1,4,2),100.*j+i))
            validate_trace(ctrl.summary(),skip)
            self.assertEqual(sum(r['action']=='reuse' for r in ctrl.decisions),0 if skip==0 else 2)
            if skip:self.assertIn(skip,ctrl.proxies)

    def test_wrong_age_or_cfg_order_rejected(self):
        import torch
        from runtime import SingleSkipController
        c=SingleSkipController(2); grid=torch.tensor([1,1,1]); f=torch.ones(1,1,2)
        with self.assertRaises(RuntimeError):c.plan_step(branch='uncond',step_index=0,num_steps=50,feature=f,grid_size=grid)
        for branch in ['cond','uncond']:
            c.plan_step(branch=branch,step_index=0,num_steps=50,feature=f,grid_size=grid)
            c.record_recompute(branch,0,f)
        c.cache_steps['cond']=-1
        with self.assertRaises(ValueError):c.plan_step(branch='cond',step_index=1,num_steps=50,feature=f,grid_size=grid)

    def test_prefix_hash_and_g1_causality(self):
        import torch
        from runtime import SingleSkipController, pool_g1
        a=torch.arange(16*5*10*12,dtype=torch.float32).reshape(16,5,10,12)/1000
        b=a+1
        baseline=SingleSkipController(0)
        baseline.observe_latent(a,0); baseline.observe_latent(b,1)
        candidate=SingleSkipController(2,baseline.features())
        candidate.observe_latent(a.clone(),0); candidate.observe_latent(b.clone(),1)
        self.assertEqual(candidate.g1[2].numel(),18432)
        self.assertTrue(torch.equal(candidate.g1[2],baseline.g1[2]))
        q,p=pool_g1(b),pool_g1(a)
        self.assertTrue(torch.equal(candidate.g1[2],torch.cat((q[0],p[0],p[0],q[1],p[1],p[1]))))
        bad=SingleSkipController(2,baseline.features())
        with self.assertRaises(ValueError):bad.observe_latent(a+.1,0)

    def test_timing_rejects_full_compute_at_target(self):
        calls=[dict(blocks_executed=30) for _ in range(100)]
        timing=dict(status='success',calls=calls)
        validate_timing(timing,0)
        with self.assertRaises(ValueError):validate_timing(timing,50)
        calls[98]['blocks_executed']=calls[99]['blocks_executed']=0
        validate_timing(timing,50)

    def test_mse_preserves_tiny_unencoded_signal(self):
        import numpy as np
        from collect import mse_frames
        a=np.full((2,3,12,12),.5,np.float32); b=a+np.float32(1e-5)
        values=mse_frames(a,b)
        self.assertGreater(values[0],0)
        self.assertAlmostEqual(values[0],float(np.float64(b[0,0,0,0]-.5)**2),places=18)
        self.assertEqual(mse_frames(a,a),[0.,0.])


if __name__=='__main__':unittest.main()
