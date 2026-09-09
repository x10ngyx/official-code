"""Online RNG/lineage, actor masking, selection boundaries and restart contracts."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT),str(PROJECT/'tests')]
from test_rl import bundle, trajectory, ScriptPolicy
from ours4wan21.contracts import EXP_ROOT, MODEL_ROOT, PROTOCOL, TrainingConfig, state_contract, dump, sha256
from ours4wan21.local_iql import compute_normalizer
from ours4wan21.train import make_networks
from ours4wan21.policy import Policy
from ours4wan21.online_common import (OnlineConfig, isolate_prompts, make_plan, prepare_directory,
    seal, verified, atomic_torch_save, read)
from ours4wan21.online_training import (OnlineTrainer, UniformReplay, fixed_support, select_checkpoint,
    trace_transitions, train_round)
from ours4wan21.online_pipeline import dispatch, evaluate_round, run_pipeline


def parent(b,mode='sea7'):
    torch.manual_seed(42)
    mc,nets=make_networks(mode,TrainingConfig(),'cpu')
    p={k:n.state_dict() for k,n in nets.items()}
    p.update(schema='ours4wan21_iql_checkpoint_v1',protocol=PROTOCOL,state=state_contract(mode),
        model_config=asdict(mc),train_config=asdict(TrainingConfig()),dataset_manifest=b['manifest'],
        normalizer=compute_normalizer(b['tensors']['state'],b['train_indices'].tolist()),smoke_only=True)
    return p


class OnlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='ours21_online_test_',dir=EXP_ROOT)
        self.models=tempfile.TemporaryDirectory(prefix='ours21_online_test_',dir=MODEL_ROOT)
        self.root=Path(self.tmp.name);self.weights=Path(self.models.name)
        (self.root/'README.md').write_text('Temporary CPU-only unit-test artifacts.\n')
        (self.weights/'README.md').write_text('Temporary CPU-only test weights.\n')

    def tearDown(self):self.tmp.cleanup();self.models.cleanup()

    def test_confirmed_config(self):
        c=OnlineConfig()
        self.assertEqual((c.joint_epochs,c.selection_start,c.selection_end),(20,11,20))
        self.assertEqual((c.critic_warmup_epochs,c.evaluation_every,c.evaluation_prompts),(5,4,20))

    def test_prompt_leakage_by_text_even_if_renamed(self):
        registry=[dict(sample_id='a',prompt='train'),dict(sample_id='v',prompt='held out')]
        evaluation=[dict(sample_id=f'e{i}',prompt=f'evaluation {i}') for i in range(20)]
        split={'a':'train','v':'evaluation'}
        isolate_prompts([registry[0]],evaluation,registry,split)
        for row in [dict(sample_id='new',prompt=' HELD   OUT '),evaluation[0]]:
            with self.assertRaises(ValueError):isolate_prompts([row],evaluation,registry,split)
        with self.assertRaises(ValueError):isolate_prompts([registry[0]],evaluation[:-1],registry,split)

    def test_plan_reproducible_stratified_and_with_replacement(self):
        pool=[dict(sample_id=f'p{i}',prompt=f'prompt {i}') for i in range(3)]
        a=make_plan(pool,1,4,lambda t:round(50*(1-1/t)))
        self.assertEqual(a,make_plan(pool,1,4,lambda t:round(50*(1-1/t))))
        self.assertNotEqual(a,make_plan(pool,2,4,lambda t:round(50*(1-1/t))))
        bins=sorted(int((r['target_speedup']-1.3)/2.2*100) for r in a)
        self.assertEqual(bins,list(range(100)))
        self.assertEqual(len({r['sampling_seed'] for r in a}),100)
        self.assertLess(len({r['sample_id'] for r in a}),100)
        for sid in {r['sample_id'] for r in a}:
            self.assertEqual(len({r['slot'] for r in a if r['sample_id']==sid}),1)

    def test_uniform_replay_not_fixed_source_mixture(self):
        b=bundle('scalar5')['tensors'];old={k:v[:100] for k,v in b.items()};new={k:v[:50] for k,v in b.items()}
        replay=UniformReplay(old,[new],seed=42)
        replay.sample(30000)
        self.assertLess(abs(replay.online_draws/30000-1/3),.015)

    def test_sampling_rng_and_forced_steps(self):
        b=bundle('scalar5');p=parent(b,'scalar5')
        for value in p['policy_net'].values():value.zero_()
        file=self.weights/'parent.pt';torch.save(p,file)
        policy=Policy(file,device='cpu',allow_smoke=True);policy.set_sampling(123)
        x=b['tensors']['state'][1]
        def sample():
            output=[]
            for _ in range(200):
                policy.reset_measurements();output.append(policy.choose(x)[0])
            return output
        a=sample();policy.set_sampling(123);self.assertEqual(a,sample())
        self.assertTrue(65<sum(a)<135)
        policy.set_sampling(42);rng=policy.sampling_generator.get_state().clone()
        c=trajectory(policy,48)
        self.assertTrue(torch.equal(rng,policy.sampling_generator.get_state()))
        self.assertEqual(c.summary()['actor_queries'],0)
        policy.set_sampling(None);self.assertEqual(policy.choose(x)[0],0)

    def test_online_latent_state_is_preserved(self):
        policy=ScriptPolicy();policy.action_mode='policy_categorical'
        trace=trajectory(policy,25).summary()
        mode='sea7_dynamics_raw_sea128'
        trace['state_contract']=state_contract(mode)
        for row in trace['decisions']:row['state']=[float(row['step_index'])]*128+row['state']
        t=trace_transitions(trace,22.,mode)
        self.assertEqual(t['state'].shape,(50,135))
        self.assertEqual(float(t['state'][32,0]),32.)
        self.assertEqual(float(t['reward'][:-1].sum()),0.)
        self.assertEqual(float(t['reward'][-1]),22.)
        bad=deepcopy(trace);bad['decisions'][2]['policy_queried']=False
        with self.assertRaises(ValueError):trace_transitions(bad,22.,mode)

    def test_selection_excludes_e10_and_uses_two_pairs(self):
        p=torch.ones(20,2)*.8;a=torch.ones(20,2,dtype=torch.long)
        result=select_checkpoint(p,a,torch.tensor([.25,.75]))
        self.assertEqual(result['selected_epoch'],20)
        self.assertEqual([r['epoch'] for r in result['candidates']],list(range(11,21)))
        # e11 must include e9->e10; this preceding flip cannot be hidden.
        a[8,0]=0;p[8,0]=.2
        rows=select_checkpoint(p,a,torch.tensor([.25,.75]))['candidates']
        self.assertAlmostEqual(rows[0]['actor_agreement'],.75)
        # A forced tie at p=.5 must use saved argmax actions, not p>=.5.
        p.fill_(.5);a.zero_();self.assertEqual(select_checkpoint(p,a,torch.ones(2))['selected_epoch'],20)

    def test_warmup_freezes_actor_and_preserves_normalizer(self):
        b=bundle('sea7');p=parent(b);t=OnlineTrainer(p)
        old={k:v[:100] for k,v in b['tensors'].items()};r=UniformReplay(old,[old])
        actor=deepcopy(t.nets['policy_net'].state_dict());critic=deepcopy(t.nets['q1_net'].state_dict())
        t.update(r,actor=False)
        self.assertTrue(all(torch.equal(v,t.nets['policy_net'].state_dict()[k]) for k,v in actor.items()))
        self.assertTrue(any(not torch.equal(v,t.nets['q1_net'].state_dict()[k]) for k,v in critic.items()))
        self.assertTrue(torch.equal(t.normalizer['mean'],p['normalizer']['mean']))
        self.assertEqual(len(t.optimizers[2].state),0)
        t.update(r,actor=True);self.assertGreater(len(t.optimizers[2].state),0)

    def test_no_actor_update_on_forced_only_replay(self):
        b=bundle('scalar5');p=parent(b,'scalar5');t=OnlineTrainer(p)
        old={k:v[[0,49]] for k,v in b['tensors'].items()};r=UniformReplay(old,[old])
        before=deepcopy(t.nets['policy_net'].state_dict());metrics=t.update(r,actor=True)
        self.assertEqual(metrics['actor_examples'],0.)
        self.assertTrue(all(torch.equal(v,t.nets['policy_net'].state_dict()[k]) for k,v in before.items()))

    def test_seals_reject_corruption_and_preserve_incomplete(self):
        d=self.root/'artifact';identity={'key':'x'}
        self.assertTrue(prepare_directory(d,identity));(d/'file').write_text('ok')
        self.assertTrue(prepare_directory(d,identity));self.assertTrue((d.parent/'incomplete').is_dir())
        (d/'file').write_text('ok');seal(d,['file'],identity=identity)
        self.assertFalse(prepare_directory(d,identity))
        (d/'file').write_text('bad')
        with self.assertRaises(ValueError):verified(d,identity)

    def test_checkpoint_round_resume_is_bitwise_equivalent(self):
        b=bundle('scalar5');p=parent(b,'scalar5');cfg=replace(OnlineConfig(),trajectories_per_round=1)
        source=self.weights/'parent.pt';torch.save(p,source)
        offline=self.root/'offline.pt';torch.save(b,offline)
        support=self.root/'fixed.pt';torch.save(fixed_support(b),support)
        one={k:v[:50] for k,v in b['tensors'].items()};data=self.root/'online1.pt'
        torch.save(dict(round=1,state=state_contract('scalar5'),tensors=one),data)
        # Interrupt after a committed epoch. Resume must exactly replay the remaining draws.
        original=OnlineTrainer.update;calls=0
        def interrupted(obj,*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==9:raise RuntimeError('simulated interruption')
            return original(obj,*args,**kwargs)
        out=self.root/'resumed';w=self.weights/'resumed'
        with patch.object(OnlineTrainer,'update',interrupted):
            with self.assertRaisesRegex(RuntimeError,'simulated'):
                train_round(source,offline,[data],support,out,w,1,cfg,'cpu',allow_smoke=True)
        result=train_round(source,offline,[data],support,out,w,1,cfg,'cpu',allow_smoke=True)
        clean=train_round(source,offline,[data],support,self.root/'clean',self.weights/'clean',1,cfg,'cpu',allow_smoke=True)
        a=torch.load(result,weights_only=False);z=torch.load(clean,weights_only=False)
        for net in ('policy_net','q1_net','q2_net','value_net','target_q1','target_q2'):
            self.assertTrue(all(torch.equal(v,z[net][k]) for k,v in a[net].items()),net)
        self.assertEqual(a['online']['selection'],z['online']['selection'])
        self.assertTrue(torch.equal(a['online']['replay_rng'],z['online']['replay_rng']))
        # Completed round is reusable, then round2 restores its optimizer counters.
        self.assertEqual(result,train_round(source,offline,[data],support,out,w,1,cfg,'cpu',allow_smoke=True))
        trainer=OnlineTrainer(a,cfg,'cpu',restore=True)
        self.assertEqual(float(next(iter(trainer.optimizers[2].state.values()))['step']),
                         a['online']['joint_epoch'])
        data2=self.root/'online2.pt';torch.save(dict(round=2,state=state_contract('scalar5'),tensors=one),data2)
        two=train_round(result,offline,[data,data2],support,self.root/'round2',self.weights/'round2',2,cfg,'cpu',allow_smoke=True)
        q=torch.load(two,weights_only=False)
        self.assertEqual(q['epoch'],q['online']['joint_epoch'])
        self.assertGreater(float(q['optimizer_states'][2]['state'][0]['step']),
                           float(a['optimizer_states'][2]['state'][0]['step']))
        self.assertTrue(torch.equal(q['normalizer']['std'],a['normalizer']['std']))

    def test_dispatch_conflicting_output_rejected(self):
        with self.assertRaises(ValueError):
            dispatch(self.root,{},[dict(output='same',kind='a'),dict(output='same',kind='b')],'test')

    def test_vbench20_builds_100_cells_for_each_checkpoint(self):
        source=self.weights/'offline.pt';source.write_text('offline')
        selected=self.weights/'selected.pt';selected.write_text('online')
        m=dict(paths=dict(start_checkpoint=str(source),calibration='unused'),gpus=['0','1'],
               evaluation=[dict(sample_id=f'e{i}',prompt=f'remote prompt {i}') for i in range(20)])
        dispatched=[];scored=[]
        def fake_dispatch(run,manifest,jobs,label):dispatched.extend(jobs)
        def fake_score(run,manifest,rows,out):
            scored.append(len(rows))
            return dict(vbench_score=.8,raw_dimension_scores={str(i):.8 for i in range(10)})
        def fake_quality(run,manifest,pairs,out):
            return {sid:dict(psnr_rgb_db_mean=22.,ssim_rgb_mean=.8,lpips_alex_v0_1_spatial_mean=.1) for sid,_,_ in pairs}
        def fake_aggregate(pairs,q,score):
            return dict(mean_quality=next(iter(q.values())),generate_speedup=2.,mean_generate_seconds=100.,mean_dit_tflops=100.,vbench_score=.8)
        with patch('ours4wan21.policy.resolve_budget',return_value=25),\
             patch('ours4wan21.online_pipeline.dispatch',side_effect=fake_dispatch),\
             patch('ours4wan21.online_pipeline.paired_quality',side_effect=fake_quality),\
             patch('ours4wan21.online_pipeline.vbench',side_effect=fake_score),\
             patch('ours4wan21.online_pipeline.aggregate_target',side_effect=fake_aggregate):
            evaluate_round(self.root,m,selected,4,OnlineConfig())
        self.assertEqual(len(dispatched),220)
        self.assertEqual(scored,[20]*11)
        self.assertEqual(len({j['output'] for j in dispatched}),220)
        result=read(self.root/'rounds/round_004/evaluation/metrics.json')
        self.assertEqual((result['candidate_cells'],result['reference_cells']),(100,100))
        self.assertEqual(len(result['per_target']),5)

    def test_calibration_checks_gpu_and_exports_measured_ratios(self):
        from ours4wan21.online_setup import calibration_from_runs
        folders=[]
        for name,method,k,seconds in [('base','baseline',None,100.),('k0','ours',0,101.),('k40','ours',40,25.)]:
            d=self.root/name;d.mkdir();folders.append(d)
            dump(d/'run.json',dict(protocol=PROTOCOL,method=method,skip_budget=k,
                checkpoint_dir='model',gpu_uuid='gpu-1',prompts=[dict(sample_id='p',prompt='text')]))
            dump(d/'COMPLETE.json',dict(videos=1))
            dump(d/'components.json',dict(rows=[dict(sample_id='p',generate_seconds=seconds)]))
        result=calibration_from_runs(folders[0],folders[1:])
        self.assertEqual(result['entries'][-1]['calibrated_speedup'],4.)
        bad=read(folders[-1]/'run.json');bad['gpu_uuid']='gpu-2';dump(folders[-1]/'run.json',bad)
        with self.assertRaises(ValueError):calibration_from_runs(folders[0],folders[1:])

    def test_r1_explicitly_discards_offline_optimizer_moments(self):
        b=bundle('scalar5');p=parent(b,'scalar5');t=OnlineTrainer(p)
        old={k:v[:100] for k,v in b['tensors'].items()};r=UniformReplay(old,[old])
        t.update(r,actor=True)
        p['optimizer_states']=[o.state_dict() for o in t.optimizers]
        fresh=OnlineTrainer(p,restore=False)
        self.assertTrue(all(len(o.state)==0 for o in fresh.optimizers))

    def test_orchestrator_evaluates_only_every_four_rounds(self):
        # Exercise the actual eight-round state machine with generation/training replaced.
        m=dict(paths=dict(start_checkpoint='start'),gpus=['0'])
        evaluated=[]
        selected=self.weights/'selected.pt';selected.write_text('mock')
        def fake_collect(run,manifest,parent,r,c):return f'online{r}'
        def fake_invoke(*args,**kwargs):
            r=int(args[0][-1]);d=self.root/'rounds'/f'round_{r:03d}'/'training'
            dump(d/'selected.json',dict(path=str(selected),sha256=sha256(selected)))
        with patch('ours4wan21.online_pipeline.require_environment'),patch('ours4wan21.online_pipeline.check_run',return_value=m),\
             patch('ours4wan21.online_pipeline.collect_round',side_effect=fake_collect),\
             patch('ours4wan21.online_pipeline.invoke',side_effect=fake_invoke),\
             patch('ours4wan21.online_pipeline.evaluate_round',side_effect=lambda run,m,p,r,c:evaluated.append(r)):
            run_pipeline(type('Args',(),dict(run_dir=self.root))())
        self.assertEqual(evaluated,[4,8])
        self.assertEqual(read(self.root/'RESULT.json')['evaluated_rounds'],[4,8])


if __name__=='__main__':unittest.main()
