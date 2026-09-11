"""Online RNG/lineage, actor masking, selection boundaries and restart contracts."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
import csv
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
    seal, verified, atomic_torch_save, read, run_config, balanced_training_slots)
from ours4wan21.online_training import (OnlineTrainer, UniformReplay, fixed_support, select_checkpoint,
    trace_transitions, train_round)
from ours4wan21.online_pipeline import dispatch, evaluate_round, run_pipeline, paired_quality
from ours4wan21.online_reference import build_evaluation, load_evaluation_bundle, select_evaluation


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

    def tearDown(self):
        for link in (PROJECT/'experiment_results').glob(self.root.name+'*'):
            if link.is_symlink() and link.resolve().is_relative_to(self.root):link.unlink()
        self.tmp.cleanup();self.models.cleanup()

    def test_confirmed_config(self):
        c=OnlineConfig()
        self.assertEqual((c.joint_epochs,c.selection_start,c.selection_end),(20,11,20))
        self.assertEqual((c.critic_warmup_epochs,c.evaluation_every,c.evaluation_prompts),(5,4,20))
        self.assertEqual((c.target_min,c.target_max),(1.5,3.5))
        self.assertEqual(c.evaluation_targets,(1.8,2.4,3.0))
        self.assertFalse(c.vbench_enabled)
        self.assertEqual(c.payload(),read(PROJECT/'configs/online.json'))
        # Match the offline data generation contract without importing its runtime.
        source=(PROJECT/'data_collection/src/ours4wan21_data/manifest.py').read_text()
        self.assertIn('TARGET_SPEEDUP_MIN = 1.5',source)
        self.assertIn('TARGET_SPEEDUP_MAX = 3.5',source)

    def test_prompt_leakage_by_text_even_if_renamed(self):
        registry=[dict(sample_id='a',prompt='train'),dict(sample_id='v',prompt='held out')]
        evaluation=[dict(sample_id=f'e{i}',prompt=f'evaluation {i}') for i in range(20)]
        split={'a':'train','v':'evaluation'}
        isolate_prompts([registry[0]],evaluation,registry,split)
        for row in [dict(sample_id='new',prompt=' HELD   OUT '),evaluation[0]]:
            with self.assertRaises(ValueError):isolate_prompts([row],evaluation,registry,split)
        with self.assertRaises(ValueError):isolate_prompts([registry[0]],evaluation[:-1],registry,split)

    def test_explicit_unpacked_environment_requires_matching_interpreter(self):
        from ours4wan21.online_common import require_environment, THREAD_KEYS
        env={k:'1' for k in THREAD_KEYS}
        env['WAN22_PYTHON']=sys.executable
        with patch.dict(os.environ,env),patch('sys.prefix','/unpacked/Wan2.2-conda-env'):
            require_environment()
            with patch.dict(os.environ,{'WAN22_PYTHON':'/different/bin/python'}):
                with self.assertRaises(ValueError):require_environment()

    def test_plan_reproducible_stratified_and_with_replacement(self):
        pool=[dict(sample_id=f'p{i}',prompt=f'prompt {i}') for i in range(3)]
        a=make_plan(pool,1,4,lambda t:round(50*(1-1/t)))
        self.assertEqual(a,make_plan(pool,1,4,lambda t:round(50*(1-1/t))))
        self.assertNotEqual(a,make_plan(pool,2,4,lambda t:round(50*(1-1/t))))
        bins=sorted(int((r['target_speedup']-1.5)/2.0*100) for r in a)
        self.assertEqual(bins,list(range(100)))
        self.assertEqual(len({r['sampling_seed'] for r in a}),100)
        self.assertLess(len({r['sample_id'] for r in a}),100)
        for sid in {r['sample_id'] for r in a}:
            self.assertEqual(len({r['slot'] for r in a if r['sample_id']==sid}),1)

    def test_budget_round_count_is_frozen_and_final_round_is_evaluated(self):
        config=OnlineConfig(rounds=2)
        self.assertEqual(run_config({'config':config.payload()}).evaluation_rounds(),[2])
        self.assertEqual(OnlineConfig().evaluation_rounds(),[4,8])
        bad=config.payload();bad['joint_epochs']=1
        with self.assertRaises(ValueError):run_config({'config':bad})
        with self.assertRaises(ValueError):run_config({'config':OnlineConfig(rounds=0).payload()})

    def test_aggressive_online_profile_matches_a2_a3_midpoint_and_is_frozen(self):
        from ours4wan21.online_common import online_config
        from ours4wan21.contracts import IQL_PROFILE_PARAMETERS
        config=online_config(rounds=8,prompt_pool_size=800,profile='aggressive_a2_a3_v1')
        a2,a3=IQL_PROFILE_PARAMETERS['aggressive_a2_v1'],IQL_PROFILE_PARAMETERS['aggressive_v1']
        for key in ('tau','beta','weight_max'):
            self.assertAlmostEqual(getattr(config,key),(a2[key]+a3[key])/2)
        default=OnlineConfig(rounds=8,prompt_pool_size=800).payload()
        self.assertEqual({k for k,v in config.payload().items() if v!=default[k]}, {'tau','beta','weight_max'})
        m=dict(config=config.payload(),iql_profile='aggressive_a2_a3_v1',paths=dict(training_bundle='/reference'),pool=[{}]*800)
        self.assertEqual(run_config(m),config)
        m['config']['tau']=.9
        with self.assertRaises(ValueError):run_config(m)
        with self.assertRaises(ValueError):online_config(profile='unknown')

    def test_aggressive_online_trainer_receives_loss_parameters(self):
        from ours4wan21.online_common import online_config
        b=bundle('scalar5');p=parent(b,'scalar5')
        config=online_config(profile='aggressive_a2_a3_v1')
        trainer=OnlineTrainer(p,config,'cpu')
        self.assertEqual((trainer.config.tau,trainer.config.beta,trainer.config.weight_max),(.85,2.5,75.))
        self.assertEqual([o.param_groups[0]['lr'] for o in trainer.optimizers],[1e-4,1e-4,4e-5])
        import ours4wan21.online_training as training
        rows={k:v[:100] for k,v in b['tensors'].items()}
        replay=UniformReplay(rows,[rows])
        with patch.object(training,'expectile_loss',wraps=training.expectile_loss) as value_loss, \
             patch.object(training,'_actor_terms',wraps=training._actor_terms) as actor_loss:
            trainer.update(replay,actor=True)
        self.assertEqual(value_loss.call_args.args[1],.85)
        self.assertEqual(actor_loss.call_args.kwargs['beta'],2.5)
        self.assertEqual(actor_loss.call_args.kwargs['weight_max'],75.)
        saved=trainer.checkpoint(replay,dict(round=1,joint_epoch=1))
        resumed=OnlineTrainer(saved,config,'cpu',restore=True)
        self.assertTrue(resumed.restored)
        with self.assertRaises(ValueError):OnlineTrainer(saved,OnlineConfig(),'cpu',restore=True)

    def test_balanced_training_slots_keep_repeated_prompt_baselines_on_same_gpu(self):
        pool=[dict(sample_id=f'prompt_{i}',prompt=f'prompt {i}') for i in range(3000)]
        c=OnlineConfig(rounds=2);slots=balanced_training_slots(pool,4,c)
        self.assertEqual(slots,balanced_training_slots(pool,4,c))
        counts=[]
        for r in (1,2):
            plan=make_plan(pool,r,4,lambda t:25,c,gpu_slots=slots)
            counts.append([sum(x['slot']==g for x in plan) for g in range(4)])
            for row in plan:self.assertEqual(row['slot'],slots[row['sample_id']])
        self.assertTrue(all(max(n)-min(n)<=4 for n in counts))

    def test_offline_training_population_and_pool_config(self):
        from ours4wan21.online_train_reference import offline_train_pool
        registry=[dict(sample_id=x,prompt=x) for x in ('z','a','v','t','unused')]
        manifest=dict(prompt_splits={'z':'train','a':'train','v':'evaluation','t':'test'})
        pool=offline_train_pool(manifest,registry)
        self.assertEqual([r['sample_id'] for r in pool],['a','z'])
        with self.assertRaises(ValueError):offline_train_pool(manifest,registry[1:])
        m=dict(config=OnlineConfig(rounds=3,prompt_pool_size=2).payload(),pool=pool,paths=dict(training_bundle='/reference'))
        self.assertEqual(run_config(m).prompt_pool_size,2)
        m['config']['actor_lr']=.1
        with self.assertRaises(ValueError):run_config(m)

    def test_collection_reuses_training_baseline_and_original_gpu(self):
        from ours4wan21.online_pipeline import collect_round
        row=dict(sample_id='a',prompt='training a',baseline_gpu_uuid='GPU-original')
        m=dict(pool=[row],gpus=['GPU-other','GPU-original'],training_gpu_slots={'a':1},
               paths=dict(training_bundle=str(self.root/'reference'),calibration='unused'),mode='sea7')
        checkpoint=self.weights/'parent.pt';checkpoint.write_bytes(b'checkpoint')
        captured=[]
        def stop_after_dispatch(run,manifest,jobs,label):
            captured.extend(jobs)
            raise RuntimeError('test stops before quality and replay')
        with patch('ours4wan21.online_pipeline.verified',return_value=True),              patch('ours4wan21.policy.resolve_budget',return_value=29),              patch('ours4wan21.online_pipeline.dispatch',side_effect=stop_after_dispatch):
            with self.assertRaisesRegex(RuntimeError,'test stops'):
                collect_round(self.root,m,checkpoint,1,OnlineConfig(prompt_pool_size=1))
        self.assertEqual(len(captured),100)
        self.assertTrue(all(j['kind']=='collection' and j['slot']==1 and j['expected_gpu_uuid']=='GPU-original' for j in captured))
        self.assertEqual(len({j['output'] for j in captured}),100)

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

    def evaluation_fixture(self):
        source=self.root/'source50';source.mkdir()
        rows=[dict(sample_id=f'p{i:03d}',prompt=f'prompt {i}') for i in range(50)]
        (source/'prompts').mkdir()
        (source/'prompts/selected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        profile=source/'profile.json';dump(profile,{'profile':'fixture'})
        config=dict(protocol=PROTOCOL,prompt_count=50,selected_ids=[r['sample_id'] for r in rows],
            targets=[1.8,2.4,3.0],skip_budgets=[23,29,35],flops_profile=str(profile),
            source_sha256={str(profile):sha256(profile)},shard_ids={})
        for g in range(4):
            shard=rows[g::4];config['shard_ids'][str(g)]=[r['sample_id'] for r in shard]
            base=source/'shards'/f'gpu{g}'/'baseline';base.mkdir(parents=True)
            dump(base/'run.json',dict(protocol=PROTOCOL,method='baseline',skip_budget=None,
                policy_checkpoint=None,flops_profile_sha256=sha256(profile),prompts=shard,
                gpu_uuid=f'GPU-{g}',gpu='fixture',checkpoint_dir='model'))
            dump(base/'COMPLETE.json',dict(status='generation_complete',videos=len(shard)))
            measured=[]
            for sub in ('videos','timings','traces'):(base/sub).mkdir()
            for row in shard:
                sid=row['sample_id']
                measured.append(dict(sample_id=sid,generate_seconds=100.,dit_tflops=1000.,
                    t5_cuda_seconds=1.,dit_cuda_seconds=90.,vae_decode_cuda_seconds=9.,
                    estimated_t5_tflops_per_video=10.,estimated_vae_decode_tflops_per_video=100.))
                (base/'videos'/f'{sid}.mp4').write_bytes(sid.encode())
                dump(base/'timings'/f'{sid}.json',dict(status='success',full_compute_forward_calls=100,
                    reuse_forward_calls=0,pipeline_generate_wall_seconds=100.))
                dump(base/'traces'/f'{sid}.json',dict(step_reuse=0,step_recompute=50))
            dump(base/'components.json',dict(rows=measured))
        dump(source/'config.json',config)
        dump(source/'COMPLETE.json',dict(status='complete',baseline_videos=50))
        out=self.root/(self.root.name+'_evaluation_bundle')
        args=type('Args',(),dict(source_run=source,output_dir=out))()
        with patch('ours4wan21.online_reference.video_geometry',return_value={'fixture':True}):
            build_evaluation(args)
        return source,out,args

    def test_reference_selection_reuses_original_files_and_detects_corruption(self):
        source,out,args=self.evaluation_fixture()
        original=read(source/'config.json')
        data=load_evaluation_bundle(out)
        expected=sorted(random.Random(42).sample(original['selected_ids'],20))
        self.assertEqual([r['sample_id'] for r in data['rows']],expected)
        before=sha256(out/'manifest.json');build_evaluation(args)
        self.assertEqual(sha256(out/'manifest.json'),before)
        for row in data['rows']:
            base=out/'baselines'/row['sample_id']
            self.assertTrue((base/'video.mp4').is_symlink())
            self.assertTrue((base/'video.mp4').resolve().is_relative_to(source))
            self.assertEqual(read(base/'measurement.json')['generate_seconds'],100.)
        row=data['rows'][0]
        (out/'baselines'/row['sample_id']/'video.mp4').resolve().write_bytes(b'corrupt')
        with self.assertRaises(ValueError):load_evaluation_bundle(out)
        with self.assertRaises(ValueError):build_evaluation(args)

    def test_evaluation20_has_60_cells_no_baseline_generation_or_vbench(self):
        archive,reference,_=self.evaluation_fixture()
        source=self.weights/'offline.pt';source.write_text('offline')
        selected=self.weights/'selected.pt';selected.write_text('online')
        data=load_evaluation_bundle(reference)
        m=dict(paths=dict(start_checkpoint=str(source),calibration='unused',evaluation_bundle=str(reference)),
               gpus=['GPU-3','GPU-2','GPU-1','GPU-0'],evaluation=data['rows'],evaluation_budgets=[23,29,35])
        dispatched=[];dispatch_calls=[]
        def fake_dispatch(run,manifest,jobs,label):
            dispatched.extend(jobs);dispatch_calls.append(label)
        def fake_quality(run,manifest,pairs,out):
            self.assertEqual(len(pairs),20)
            for _,base,_ in pairs:self.assertTrue(Path(base).is_relative_to(reference))
            return {sid:dict(psnr_rgb_db_mean=22.,ssim_rgb_mean=.8,lpips_alex_v0_1_spatial_mean=.1) for sid,_,_ in pairs}
        def fake_aggregate(pairs,q):
            return dict(mean_quality=next(iter(q.values())),generate_speedup=2.,mean_generate_seconds=100.,mean_dit_tflops=100.)
        with patch('ours4wan21.online_pipeline.dispatch',side_effect=fake_dispatch),\
             patch('ours4wan21.online_pipeline.paired_quality',side_effect=fake_quality),\
             patch('ours4wan21.online_pipeline.aggregate_target',side_effect=fake_aggregate),\
             patch('ours4wan21.online_pipeline.invoke',side_effect=AssertionError('unexpected evaluation subprocess')):
            evaluate_round(self.root,m,selected,4,OnlineConfig())
        self.assertEqual(len(dispatched),120)
        self.assertEqual(len(dispatch_calls),1)
        self.assertEqual({j['kind'] for j in dispatched},{'evaluation'})
        self.assertEqual({j['target_speedup']:j['skip_budget'] for j in dispatched},{1.8:23,2.4:29,3.0:35})
        self.assertEqual(len({j['output'] for j in dispatched}),120)
        by_id={r['sample_id']:r for r in data['rows']}
        for job in dispatched:
            self.assertEqual(m['gpus'][job['slot']],by_id[job['sample_id']]['baseline_gpu_uuid'])
            self.assertEqual(job['expected_gpu_uuid'],m['gpus'][job['slot']])
        result=read(self.root/'rounds/round_004/evaluation/metrics.json')
        self.assertEqual((result['candidate_cells'],result['reference_cells']),(60,60))
        self.assertEqual(len(result['per_target']),3)
        self.assertEqual(result['vbench_status'],'skipped_by_user')
        self.assertEqual(result['reused_baseline_videos'],20)
        self.assertNotIn('vbench_score',json.dumps(result))
        self.assertFalse(list(self.root.rglob('*vbench20*')))

    def test_reference_rejects_wrong_native_protocol_and_prompt(self):
        source,out,args=self.evaluation_fixture()
        row=load_evaluation_bundle(out)['rows'][0]
        shard=next(k for k,ids in read(source/'config.json')['shard_ids'].items() if row['sample_id'] in ids)
        path=source/'shards'/f'gpu{shard}'/'baseline/run.json'
        manifest=read(path);manifest['protocol']['seed']=0;dump(path,manifest)
        args.output_dir=self.root/'bad_bundle'
        with self.assertRaisesRegex(ValueError,'native baseline identity'):build_evaluation(args)
        self.assertFalse(args.output_dir.exists())

    def test_parallel_quality_merges_complete_original_rows_and_reuses_results(self):
        _,reference,_=self.evaluation_fixture();data=load_evaluation_bundle(reference);pairs=[]
        for row in data['rows'][:8]:
            sid=row['sample_id'];candidate=self.root/'candidates'/sid;candidate.mkdir(parents=True)
            identity=dict(job=dict(sample_id=sid,prompt=row['prompt']))
            dump(candidate/'generation.json',dict(identity=identity,protocol=PROTOCOL,gpu_uuid=row['baseline_gpu_uuid']))
            (candidate/'video.mp4').write_bytes(sid.encode())
            seal(candidate,['generation.json','video.mp4'],identity=identity)
            pairs.append((sid,str(reference/'baselines'/sid),str(candidate)))
        devices=[]
        def fake_metrics(command,log,gpu):
            devices.append(gpu)
            reference_dir=Path(command[command.index('--reference-dir')+1])
            output=Path(command[command.index('--output-dir')+1]);output.mkdir()
            ids=sorted(p.stem for p in reference_dir.glob('*.mp4'))
            rows=[dict(video_id=sid,frames=81,height=480,width=832,psnr_rgb_db_mean=22.,
                ssim_rgb_mean=.8,lpips_alex_v0_1_spatial_mean=.1) for sid in ids]
            with (output/'per_video.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with (output/'per_frame.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['video_id','frame_index']);writer.writeheader()
                writer.writerows(dict(video_id=sid,frame_index=i) for sid in ids for i in range(81))
            dump(output/'summary.json',dict(videos=len(ids)))
        out=self.root/'quality';m=dict(gpus=['GPU-0','GPU-1','GPU-2','GPU-3'])
        with patch('ours4wan21.online_pipeline.invoke',side_effect=fake_metrics):
            result=paired_quality(self.root,m,pairs,out)
        self.assertEqual(set(devices),set(m['gpus']))
        self.assertEqual(set(result),{p[0] for p in pairs})
        self.assertEqual(read(out/'metrics/summary.json')['frames'],8*81)
        self.assertEqual(read(out/'metrics/summary.json')['mean_metrics']['psnr_rgb_db_mean'],22.)
        with (out/'metrics/per_frame.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))),8*81)
        with patch('ours4wan21.online_pipeline.invoke',side_effect=AssertionError('must reuse complete quality')):
            self.assertEqual(paired_quality(self.root,m,pairs,out),result)

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
        m=dict(paths=dict(start_checkpoint='start'),gpus=['0'],config=OnlineConfig().payload())
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
