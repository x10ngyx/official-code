"""Causality, archive/live equivalence and ten-group training integration."""
import unittest,tempfile
from pathlib import Path
from copy import deepcopy
import torch
from test_rl import bundle,ScriptPolicy
from ours4wan21.latent_features import GROUPS,LatentFeatureHistory,sea_filter,contract
from ours4wan21.contracts import FEATURE_MODES,EXP_ROOT,TrainingConfig,state_contract
from ours4wan21.runtime import Controller,reference
from ours4wan21.feature_cache import extract
from ours4wan21.train import validate_bundle,make_networks,make_optimizers
from ours4wan21.data import Transitions
from ours4wan21.local_iql import _run_epoch,compute_normalizer,apply_normalizer
from ours4wan21.overhead import network_flops
from torch.utils.data import DataLoader

class LatentGroupsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        self.x=torch.randn(50,16,2,4,4)*.2
        self.sigmas=torch.linspace(.99,.01,50)
        self.actions=[0 if i in (0,49) or i%7==0 else 1 for i in range(50)]
    def test_single_joint_quantization_and_no_stage32_reset(self):
        self.assertEqual(len(FEATURE_MODES),10)
        together=LatentFeatureHistory(GROUPS);single={g:LatentFeatureHistory([g]) for g in GROUPS}
        for t in range(50):
            values=together.observe(self.x[t],t,self.sigmas[t])
            for g,h in single.items():
                out=h.observe(self.x[t].half(),t,self.sigmas[t])[g]
                torch.testing.assert_close(out,values[g],atol=0,rtol=0)
                self.assertEqual(out.shape,(1,GROUPS[g][0]))
                if t==0:self.assertEqual(out.count_nonzero(),0)
                if t==32:self.assertGreater(out.count_nonzero(),0)
                h.commit(self.actions[t])
            together.commit(self.actions[t])
        self.assertEqual(contract('cache_update192')['stage_boundaries'],[0])
    def test_future_cannot_change_prefix(self):
        a,b=LatentFeatureHistory(GROUPS),LatentFeatureHistory(GROUPS)
        for t in range(50):
            x=a.observe(self.x[t],t,self.sigmas[t]);y=b.observe(self.x[t] if t<21 else self.x[t]*3,t,self.sigmas[t])
            if t<21:
                for g in GROUPS:torch.testing.assert_close(x[g],y[g],atol=0,rtol=0)
            a.commit(self.actions[t]);b.commit(self.actions[t] if t<21 else 0)
    def test_sea_matches_official(self):
        c=reference.SeaCacheController(reference.SeaCacheConfig(threshold=1.))
        x=self.x[1:2].half().float();s=float(self.sigmas[1])
        torch.testing.assert_close(sea_filter(x,s),c._apply_sea_from_ab(x,1-s,s,dims=(2,3,4)),rtol=1e-5,atol=1e-6)
    def test_controller_all_groups_exactK_cfg_and_offline_parity(self):
        for mode,g in FEATURE_MODES.items():
            policy=ScriptPolicy();policy.mode=mode;c=Controller(policy,25);c.set_scheduler_sigmas(self.sigmas)
            history=LatentFeatureHistory([g])
            for t in range(50):
                expected=history.observe(self.x[t].half(),t,self.sigmas[t])[g][0];c.observe_latent(self.x[t],t)
                feature=torch.arange(32).float().reshape(1,8,4)+1+t*.1;decisions=[]
                for branch in ('cond','uncond'):
                    reuse=c.plan_step(branch=branch,step_index=t,num_steps=50,feature=feature,grid_size=torch.tensor([2,2,2]))
                    if reuse:c.reuse_residual(branch,t)
                    else:c.record_recompute(branch,t,torch.ones_like(feature))
                    torch.testing.assert_close(torch.tensor(c.decisions[-1]['state'][:-7]),expected,atol=0,rtol=0);decisions.append(reuse)
                self.assertEqual(*decisions);history.commit(int(decisions[0]))
            self.assertEqual(c.summary()['step_reuse'],25);self.assertEqual(c.feature_overhead()['calls'],50)
            self.assertEqual(c.summary()['actor_queries'],policy.calls)
            c.reset();self.assertEqual(c.feature_overhead()['calls'],0);self.assertEqual(c.feature_history.last_step,-1)
    def test_archive_extraction_and_dtype_guard(self):
        with tempfile.TemporaryDirectory(dir=EXP_ROOT) as folder:
            records=[]
            for t in range(50):
                p=Path(folder)/f'{t}.pt';torch.save(self.x[t].half(),p)
                records.append(dict(step_index=t,model_stage='single',sigma=float(self.sigmas[t]),latent_path=str(p),latent_shape=list(self.x[t].shape),latent_dtype='torch.float16'))
            features,hashes=extract(records,torch.tensor(self.actions),strict_shape=False);self.assertEqual(len(hashes),50)
            h=LatentFeatureHistory(GROUPS)
            for t in range(50):
                values=h.observe(self.x[t],t,self.sigmas[t])
                for g in GROUPS:torch.testing.assert_close(values[g][0],features[g][t],atol=0,rtol=0)
                h.commit(self.actions[t])
            torch.save(self.x[0],records[0]['latent_path'])
            with self.assertRaisesRegex(ValueError,'dtype'):extract(records,self.actions,strict_shape=False)
    def test_feature_cache_rejects_other_actions_and_changed_trace(self):
        import json
        from ours4wan21.contracts import dump,sha256
        from ours4wan21.feature_cache import load_features
        with tempfile.TemporaryDirectory(dir=EXP_ROOT) as folder:
            root=Path(folder);trace=root/'trace.json';trace.write_text('{}')
            actions=torch.tensor(self.actions);row=dict(trajectory_id='t',sample_id='p',split='train')
            values={g:torch.zeros(50,dim) for g,(dim,_) in GROUPS.items()}
            target=root/'features.pt'
            torch.save(dict(**row,trace_sha256=sha256(trace),actions=actions,features=values),target)
            dump(root/'index.json',dict(contracts={g:contract(g) for g in GROUPS},rows={'t':dict(file='features.pt',sha256=sha256(target))}))
            dump(root/'COMPLETE.json',dict(index_sha256=sha256(root/'index.json')))
            self.assertEqual(set(load_features(root,row,trace,actions)),set(GROUPS))
            with self.assertRaisesRegex(ValueError,'another trajectory'):load_features(root,row,trace,1-actions)
            trace.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'another trajectory'):load_features(root,row,trace,actions)

    def test_ten_groups_real_iql_backward(self):
        base=bundle('sea7');features={g:[] for g in GROUPS}
        for e in range(4):
            h=LatentFeatureHistory(GROUPS)
            for t in range(50):
                values=h.observe(self.x[t],t,self.sigmas[t])
                for g in GROUPS:features[g].append(values[g][0])
                h.commit(int(base['tensors']['action'][e*50+t]))
        for mode,g in FEATURE_MODES.items():
            b=deepcopy(base);b['manifest']['state']=state_contract(mode)
            x=torch.cat((torch.stack(features[g]),base['tensors']['state']),1);b['tensors']['state']=x
            b['tensors']['next_state']=torch.cat([torch.cat((x[i+1:i+50],x[i+49:i+50])) for i in range(0,200,50)])
            validate_bundle(b,mode);cfg=TrainingConfig();mc,nets=make_networks(mode,cfg,'cpu')
            self.assertEqual(mc.input_dim,7+GROUPS[g][0]);norm=compute_normalizer(x,b['train_indices'].tolist())
            for key in ('state','next_state'):b['tensors'][key]=apply_normalizer(b['tensors'][key],norm)
            before=next(nets['policy_net'].parameters()).detach().clone()
            metrics=_run_epoch(loader=DataLoader(Transitions(b['tensors'],b['train_indices']),batch_size=64),**nets,args=cfg,device=torch.device('cpu'),optimizers=make_optimizers(nets,cfg))
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()));self.assertFalse(torch.equal(before,next(nets['policy_net'].parameters())))
            self.assertGreater(network_flops(nets['policy_net'])['flops_per_call'],271360)
if __name__=='__main__':unittest.main()
