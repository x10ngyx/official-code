"""Compare the complete imported forward against the untouched official forward."""
import ast
import copy
import sys
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
import torch.cuda.amp as amp

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from dicache import DiCacheConfig, DiCacheController, SOURCE
from wan21_integration import dicache_forward


def embedding(dim,t):
    return torch.stack([t*.001,torch.sin(t),torch.cos(t),t*.0001],dim=-1)


def official_forward():
    tree=ast.parse(SOURCE.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dicache_forward')
    env=dict(torch=torch,amp=amp,sinusoidal_embedding_1d=embedding)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),env)
    return env['dicache_forward']


class Patch(nn.Conv3d):
    def __init__(self,bf16):
        super().__init__(2,4,1,bias=False)
        self.bf16=bf16
    def forward(self,x):
        x=super().forward(x)
        return x.bfloat16() if self.bf16 else x


class Projection(nn.Module):
    def forward(self,x): return x.repeat(1,6)


class Block(nn.Module):
    def __init__(self,scale):
        super().__init__();self.scale=scale;self.calls=0
    def forward(self,x,*,e,context,**kwargs):
        self.calls+=1
        return x.float()+self.scale*torch.tanh(x.float()+e[:,0:1]*.03+context.mean(1,keepdim=True)*.04)


class Head(nn.Module):
    def forward(self,x,e): return (x.float()+e[:,None]*.02)[...,:2]


class Model(nn.Module):
    def __init__(self,bf16):
        super().__init__()
        self.model_type='t2v';self.dim=self.freq_dim=4;self.text_len=3
        self.freqs=torch.zeros(1)
        self.patch_embedding=Patch(bf16)
        self.time_embedding=nn.Identity();self.time_projection=Projection()
        self.text_embedding=nn.Linear(5,4)
        self.blocks=nn.ModuleList([Block(.2*(i+1)) for i in range(4)])
        self.head=Head()
    def unpatchify(self,x,grids):
        return [v.transpose(0,1).reshape(2,*g.tolist()) for v,g in zip(x,grids)]


def reference_state(model,threshold,retention):
    model.cnt=0;model.num_steps=100;model.probe_depth=1
    model.rel_l1_thresh=threshold;model.ret_ratio=retention
    model.accumulated_rel_l1_distance=[0.,0.]
    model.residual_cache=[None,None];model.probe_residual_cache=[None,None]
    model.residual_window=[[],[]];model.probe_residual_window=[[],[]]
    model.previous_internal_states=[None,None];model.previous_input=[None,None]
    model.previous_output=[None,None];model.resume_flag=[False,False]


class OfficialParityTests(unittest.TestCase):
    def test_full_forward_matches_official_fp32_bf16_and_lifecycle(self):
        oracle=official_forward()
        stub=types.ModuleType('wan.modules.model');stub.sinusoidal_embedding_1d=embedding
        with patch.dict(sys.modules,{'wan.modules.model':stub}),torch.no_grad(),warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for bf16 in [False,True]:
                for threshold,retention in [(0.,.2),(.08,.2),(.2,.2),(100.,.2),(100.,.02)]:
                    with self.subTest(bf16=bf16,threshold=threshold,retention=retention):
                        torch.manual_seed(42)
                        original=Model(bf16);actual=copy.deepcopy(original)
                        reference_state(original,threshold,retention)
                        actual.dicache_controller=DiCacheController(DiCacheConfig(threshold,retention_ratio=retention))
                        latent=torch.randn(2,1,1,3)+2
                        contexts=[torch.randn(2,5),torch.randn(2,5)]
                        for video in range(2):
                            actual.dicache_controller.reset()
                            for i in range(100):
                                x=[latent+(i//2)*(.004 if i%2==0 else .05)]
                                kw=dict(t=torch.tensor([1000.-(i//2)*.1]),context=[contexts[i%2]],seq_len=3)
                                before=sum(b.calls for b in original.blocks)
                                expected=oracle(original,x,**kw)[0]
                                observed=dicache_forward(actual,x,**kw,dicache_branch=('cond','uncond')[i%2],
                                                          dicache_step_index=i//2,dicache_num_steps=50)[0]
                                self.assertTrue(torch.equal(observed,expected))
                                count=sum(b.calls for b in original.blocks)-before
                                row=actual.dicache_controller.decisions[-1]
                                self.assertEqual(count,row['probe_blocks_executed']+row['deep_blocks_executed'])
                            self.assertEqual(actual.dicache_controller.state.residual_cache,[None,None])
                        if threshold==100 and retention==.2:
                            self.assertEqual(actual.dicache_controller.decisions[20]['residual_window_length'],3)
                            self.assertEqual(actual.dicache_controller.decisions[20]['block_output_dtype'],
                                             'torch.bfloat16' if bf16 else 'torch.float32')
                        if threshold==100 and retention==.02:
                            self.assertIsNone(actual.dicache_controller.decisions[2]['dcta_gamma'])

    def test_zero_denominators_are_not_stabilized(self):
        class Identity:
            def __call__(self,x,**kw):return x
        c=DiCacheController(DiCacheConfig(.2))
        for i in range(20):
            c.execute(blocks=[Identity(),Identity()],x=torch.zeros(1),kwargs={},branch=('cond','uncond')[i%2],step_index=i//2,num_steps=50)
        out=c.execute(blocks=[Identity(),Identity()],x=torch.zeros(1),kwargs={},branch='cond',step_index=10,num_steps=50)
        self.assertEqual(c.decisions[-1]['probe_relative_change'],'nan')
        self.assertEqual(c.decisions[-1]['action'],'full_compute')
        # Nonzero gate denominator but zero DCTA denominator preserves NaN output.
        c=DiCacheController(DiCacheConfig(.2))
        for i in range(20):
            c.execute(blocks=[Identity(),Identity()],x=torch.ones(1),kwargs={},branch=('cond','uncond')[i%2],step_index=i//2,num_steps=50)
        out=c.execute(blocks=[Identity(),Identity()],x=torch.ones(1),kwargs={},branch='cond',step_index=10,num_steps=50)
        self.assertTrue(torch.isnan(out).all())
        self.assertEqual(c.decisions[-1]['dcta_gamma'],'nan')

    def test_invalid_cfg_order_and_reset(self):
        c=DiCacheController(DiCacheConfig(.2))
        with self.assertRaises(ValueError):
            c.execute(blocks=[None,None],x=torch.ones(1),kwargs={},branch='uncond',step_index=0,num_steps=50)
        c.reset()
        self.assertEqual(c.decisions,[])


if __name__=='__main__':unittest.main()
