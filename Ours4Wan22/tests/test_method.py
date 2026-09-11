"""CPU regression: formal network parity, causal history, all budgets and metrics."""
import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ours4wan22.contracts import DIM, FEATURE, PROTOCOL, scalar_state
from ours4wan22.features import History, pool
from ours4wan22.model import CNN, networks, reference, encode_normalized
from ours4wan22.runtime import Controller, apply_policy
from ours4wan22.metrics import performance
from ours4wan22.shared import OFFICIAL, verify_sources


class Actor:
    def __init__(self, action=1):
        self.action = action
        self.reset_measurements()
    def reset_measurements(self):
        self.calls = []
    def choose(self,x):
        self.calls.append(x.clone())
        return self.action, .8 if self.action else .2


def trajectory(k,action=1):
    policy = Actor(action)
    ctrl = Controller(policy,k)
    ctrl.set_scheduler_sigmas(torch.linspace(1,0,51))
    # Test budgeting/filter/CFG/residual execution without allocating full video features.
    for step in range(50):
        stage = 'high' if step < 32 else 'low'
        if step == 32:
            ctrl.clear_stage('high')
        ctrl.history.step,ctrl.history.pending = step,True
        ctrl.history.current = torch.tensor(float(step))
        ctrl.latent_feature = torch.zeros(18432)
        feature = torch.ones(1,8,4)*(step+1)
        kw = dict(stage=stage,step_index=step,num_steps=50,feature=feature,grid_size=torch.tensor([2,2,2]))
        for branch in ('cond','uncond'):
            reuse = ctrl.plan_step(**kw)
            if reuse:
                residual = ctrl.reuse_residual(stage,branch,step)
                # Separate residual content must survive shared decisions.
                assert residual.item() % 2 == (0 if branch=='cond' else 1)
            else:
                ctrl.record_recompute(stage,branch,step,torch.tensor([2*step+(branch=='uncond')]))
    return ctrl,policy


class MethodTests(unittest.TestCase):
    def test_source_lock(self):
        verify_sources()

    def test_formal_network_exact_parity(self):
        torch.manual_seed(7)
        a = reference.CNN('G1',2)
        b = CNN()
        b.load_state_dict(a.state_dict())
        x = torch.randn(2,DIM)
        self.assertTrue(torch.equal(a(x),b(x)))
        self.assertEqual(sum(p.numel() for p in b.parameters()),316034)
        self.assertEqual(CNN(1)(x).shape,(2,))

    def test_predictor_flops_from_executed_shapes(self):
        net = CNN()
        counts,handles = [],[]
        def count(module,inputs,output):
            if isinstance(module,torch.nn.Linear):
                counts.append(output.numel()*module.in_features*2)
            else:
                import math
                counts.append(output.numel()*(module.in_channels//module.groups)*math.prod(module.kernel_size)*2)
        for m in net.modules():
            if isinstance(m,(torch.nn.Conv2d,torch.nn.Conv3d,torch.nn.Linear)):
                handles.append(m.register_forward_hook(count))
        net(torch.zeros(1,DIM))
        for h in handles:h.remove()
        self.assertEqual(sum(counts),24580608)

    def test_checkpoint_loader_rejects_wan21_and_smoke(self):
        import tempfile
        from ours4wan22.policy import Policy,SCHEMA
        from ours4wan22.model import ARCH
        from ours4wan22.shared import MODELS
        with tempfile.TemporaryDirectory(prefix='ours22_unit_',dir=MODELS) as folder:
            path = Path(folder)/'actor.pt'
            net = CNN()
            payload = dict(schema=SCHEMA,architecture=ARCH,feature_contract=FEATURE,protocol=PROTOCOL,
                group='G1',smoke_only=False,policy_net=net.state_dict(),
                normalizer=dict(mean=torch.zeros(DIM),std=torch.ones(DIM)))
            torch.save(payload,path)
            actor = Policy(path,device='cpu')
            action,p = actor.choose(torch.zeros(DIM))
            self.assertIn(action,(0,1));self.assertTrue(0 <= p <= 1)
            self.assertEqual(actor.overhead_summary()['call_count'],1)
            payload['smoke_only']=True;torch.save(payload,path)
            with self.assertRaises(ValueError):Policy(path,device='cpu')
            payload['smoke_only']=False;payload['protocol']=dict(PROTOCOL,frames=81)
            torch.save(payload,path)
            with self.assertRaises(ValueError):Policy(path,device='cpu')

    def test_speed_budget_requires_empirical_wan22_contract(self):
        from ours4wan22.policy import resolve_budget
        self.assertEqual(resolve_budget(target_speedup=2.),27)
        with self.assertRaises(ValueError):resolve_budget(skip_budget=23,target_speedup=2.)
        empirical=dict(schema='ours4wan22_speed_to_k_v1',status='calibrated',protocol=PROTOCOL,
            forced_steps=[0,32,49],entries=[dict(skip_budget=20,calibrated_speedup=1.7),dict(skip_budget=25,calibrated_speedup=2.1)])
        with patch('ours4wan22.policy.read',return_value=empirical):
            self.assertEqual(resolve_budget(target_speedup=2.,calibration='fixture'),25)
        with patch('ours4wan22.policy.read',return_value=dict(empirical,protocol={})):
            with self.assertRaises(ValueError):resolve_budget(target_speedup=2.,calibration='fixture')

    def test_default_full_generate_calibration_targets_and_bounds(self):
        from ours4wan22.policy import resolve_budget, DEFAULT_CALIBRATION
        from ours4wan22.shared import read
        expected = {1.5:18, 1.8:24, 2.:27, 2.2:29, 2.4:31,
                    2.5:32, 2.6:33, 2.8:35, 3.:36, 3.5:38}
        for target, k in expected.items():
            with self.subTest(target=target):
                self.assertEqual(resolve_budget(target_speedup=target), k)
        for target in (1.49, 3.51, float('nan'), float('inf')):
            with self.assertRaises(ValueError):resolve_budget(target_speedup=target)
        self.assertEqual(resolve_budget(skip_budget=23),23)
        with self.assertRaises(ValueError):resolve_budget(skip_budget=23,calibration=DEFAULT_CALIBRATION)
        table=read(DEFAULT_CALIBRATION)
        entries={e['skip_budget']:e for e in table['entries']}
        self.assertEqual(entries[32]['estimate'],'in_support_latency_interpolation')
        self.assertAlmostEqual(entries[32]['calibrated_speedup'],
            1/(.5/entries[31]['calibrated_speedup']+.5/entries[33]['calibrated_speedup']))

    def test_real_prepared_forward_all_recompute_equals_native(self):
        import ast
        import math
        path = OFFICIAL/'SeaCache4Wan22/build/Wan2.2-42bf4cf-prepared-modnorm-fix/wan/modules/model.py'
        if not path.exists():
            self.skipTest('prepared Wan22 tree not available')
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WanModel')
        forward = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        scope = dict(torch=torch,sinusoidal_embedding_1d=lambda dim,t:torch.zeros(len(t),dim))
        exec(compile(ast.Module(body=[forward],type_ignores=[]),str(path),'exec'),scope)
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__();self.modulation=torch.zeros(1,6,4);self.calls=0
            def _modulated_norm1(self,x,e):return x
            def forward(self,x,**kwargs):self.calls+=1;return x+kwargs['context'].mean()
        model = types.SimpleNamespace(model_type='t2v',patch_embedding=torch.nn.Conv3d(16,4,1),
            freqs=torch.zeros(1),freq_dim=4,dim=4,text_len=2,
            time_embedding=torch.nn.Identity(),time_projection=torch.nn.Linear(4,24),text_embedding=torch.nn.Identity(),
            blocks=[Block(),Block()],head=lambda x,e:x,unpatchify=lambda x,g:list(x))
        ctrl=Controller(Actor(),0);ctrl.set_scheduler_sigmas(torch.linspace(1,0,51))
        x=[torch.randn(16,2,2,2)]
        for step in range(50):
            stage='high' if step<32 else 'low'
            if step==32:ctrl.clear_stage('high')
            ctrl.history.step,ctrl.history.pending,ctrl.history.current=step,True,x[0]
            ctrl.latent_feature=torch.zeros(18432)
            for branch in ('cond','uncond'):
                kwargs=dict(t=torch.tensor([float(50-step)]),context=[torch.ones(2,4)*(1 if branch=='cond' else -1)],seq_len=8)
                native=scope['forward'](model,x,**kwargs)[0]
                cached=scope['forward'](model,x,**kwargs,seacache=ctrl,seacache_stage=stage,
                    seacache_branch=branch,seacache_step_index=step,seacache_num_steps=50)[0]
                self.assertTrue(torch.equal(native,cached))
        self.assertEqual(ctrl.summary()['reuse'],0)

    def test_independent_encoders_and_targets(self):
        nets = networks()
        pointers = [next(n.parameters()).data_ptr() for n in nets.values()]
        self.assertEqual(len(set(pointers)),6)
        self.assertTrue(torch.equal(nets['q1_net'].c3[0].weight,nets['target_q1'].c3[0].weight))
        self.assertFalse(next(nets['target_q1'].parameters()).requires_grad)

    def test_normalized_fp16_boundary(self):
        x = torch.randn(2,DIM)
        norm = dict(mean=torch.randn(DIM),std=torch.rand(DIM)+.1)
        self.assertTrue(torch.equal(encode_normalized(x,norm),((x-norm['mean'])/norm['std']).half()))

    def test_raw_history_formula_and_stage_reset(self):
        h = History()
        z = torch.arange(12).reshape(1,12,1,1).expand(16,12,60,104).float()
        self.assertEqual(h.observe(z,0,1.).count_nonzero(),0)
        h.commit(0)
        value = h.observe(z+1,1,.98)
        roles = [(z+1).half().float()[None],z[None],z[None]]
        parts = [pool(role) for role in roles]
        expected = torch.cat([p[0] for p in parts]+[p[1] for p in parts])
        self.assertTrue(torch.equal(value,expected))
        h.commit(1)
        self.assertTrue(torch.equal(h.cache,z[None]))
        self.assertTrue(torch.equal(h.previous,(z+1)[None]))
        h.step,h.sigma = 31,.4
        self.assertEqual(h.observe(z+100,32,.3).count_nonzero(),0)
        h.commit(0)
        self.assertTrue(torch.equal(h.cache,(z+100)[None]))

    def test_history_rejects_malformed_sequence(self):
        h = History()
        with self.assertRaises(ValueError):h.observe(torch.zeros(16,21,60,104),0,1.)
        with self.assertRaises(ValueError):h.commit(0)

    def test_all_budgets_both_actor_extremes(self):
        for action in (0,1):
            for k in range(48):
                ctrl,policy = trajectory(k,action)
                self.assertEqual(ctrl.summary()['reuse'],k)
                self.assertTrue(all(ctrl.decisions[i]['action']=='recompute' for i in (0,32,49)))
                self.assertEqual(len(policy.calls),sum(d['policy_queried'] for d in ctrl.decisions))
                self.assertFalse(ctrl.decisions[32]['state'][2])
                self.assertEqual(ctrl.decisions[32]['used_skips_before'],sum(d['action']=='reuse' for d in ctrl.decisions[:32]))
                self.assertTrue(all(d['actor_mask']==float(d['policy_queried']) for d in ctrl.decisions))

    def test_invalid_budget_and_missing_sigma(self):
        for k in (-1,48,True,1.5):
            with self.assertRaises(ValueError):Controller(Actor(),k)
        c = Controller(Actor(),20)
        with self.assertRaises(ValueError):c.set_scheduler_sigmas(None)
        with self.assertRaises(ValueError):c.set_scheduler_sigmas(torch.ones(51))

    def test_cfg_order_and_summary_fail_closed(self):
        c = Controller(Actor(),20)
        with self.assertRaises(RuntimeError):c.summary()
        with self.assertRaises(RuntimeError):c.reuse_residual('high','uncond',0)
        with self.assertRaises(ValueError):c.plan_step(stage='low',step_index=0,num_steps=50,feature=torch.ones(1,8,4),grid_size=torch.tensor([2,2,2]))

    def test_scoped_integration_restores_on_error(self):
        sampler = types.ModuleType('wan.text2video')
        sampler.SeaCacheController = object()
        original = sampler.SeaCacheController
        wan = types.ModuleType('wan');wan.text2video=sampler
        high,low = types.SimpleNamespace(forward=lambda *a,**k:None),types.SimpleNamespace(forward=lambda *a,**k:None)
        old = high.forward
        pipeline = types.SimpleNamespace(t5_cpu=False,rank=0,sp_size=1,t5_fsdp=False,dit_fsdp=False,use_sp=False,param_dtype=torch.bfloat16,boundary=.875,high_noise_model=high,low_noise_model=low,generate=lambda *a,**k:None)
        with patch.dict(sys.modules,{'wan':wan,'wan.text2video':sampler}):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                with apply_policy(pipeline,Actor(),20):
                    raise RuntimeError('injected')
        self.assertIs(high.forward,old)
        self.assertIs(sampler.SeaCacheController,original)

    def test_metrics_follow_actual_branch_execution(self):
        ctrl,_ = trajectory(23)
        fixture = importlib.util.spec_from_file_location('_sea_metric_fixture',OFFICIAL/'SeaCache4Wan22/tests/test_performance.py')
        module = importlib.util.module_from_spec(fixture);fixture.loader.exec_module(module)
        timing = module.make_dit_forward(candidate=True,block_count=40)
        timing.update(schema_version=2,status='success',pipeline_generate_wall_seconds=100.)
        for c in timing['calls']:
            full = ctrl.decisions[c['step_index']]['action']=='recompute'
            c.update(blocks_executed=40 if full else 0,full_compute=full,reuse=not full)
        profile = dict(input=dict(transformer_blocks=40),stages={s:dict(branches={b:dict(estimated_full_flops=1e12,estimated_always_on_flops=1e10) for b in ('cond','uncond')}) for s in ('high','low')},
            component_profiles={s:dict(calls_per_video=n,estimated_flops_per_video=2e12,estimated_tflops_per_video=2.) for s,n in (('t5',2),('vae_decode',1))})
        trace = ctrl.trace()
        row = performance(timing,profile,trace)
        self.assertAlmostEqual(row['estimated_dit_tflops'],54.+.46)
        self.assertEqual(row['estimated_t5_tflops_per_video'],2.)
        corrupt = copy.deepcopy(timing);corrupt['calls'][1]['blocks_executed']=0
        with self.assertRaises(ValueError):performance(corrupt,profile,trace)
        corrupt = copy.deepcopy(timing);del corrupt['component_latency']['t5']
        with self.assertRaises(ValueError):performance(corrupt,profile,trace)

    def test_training_mask_split_and_real_iql_update(self):
        from ours4wan22.training import validate_data,Transitions
        from ours4wan21.local_iql import _run_epoch
        from ours4wan21.contracts import TrainingConfig
        ctrl,_ = trajectory(23)
        data = dict(schema='ours4wan22_cnn_G1_dataset_v1',protocol=PROTOCOL,feature_contract=FEATURE,
            states=torch.zeros(2,50,DIM),actions=torch.tensor([[int(d['action']=='reuse') for d in ctrl.decisions]]*2),
            terminal_psnr=torch.tensor([30.,31.]),prompt_ids=['a','b'],split=['train','val'])
        data['states'][:,:,-7:] = torch.tensor([d['state'] for d in ctrl.decisions])
        mask = validate_data(data)
        self.assertTrue(torch.equal(mask[0],torch.tensor([d['actor_mask'] for d in ctrl.decisions])))
        bad = dict(data,prompt_ids=['same','same'])
        with self.assertRaises(ValueError):validate_data(bad)
        nets = networks()
        optimizers = tuple(torch.optim.AdamW(p,lr=1e-4) for p in (nets['value_net'].parameters(),list(nets['q1_net'].parameters())+list(nets['q2_net'].parameters()),nets['policy_net'].parameters()))
        dataset = Transitions(data['states'].half(),data,mask,'train')
        # Include both a free action and terminal reward, keeping the test bounded.
        indices = [next(i for i in range(50) if mask[0,i]),49]
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset,indices),batch_size=2)
        before = nets['q1_net'].head[-1].weight.detach().clone()
        metrics = _run_epoch(loader=loader,**nets,args=TrainingConfig(),device=torch.device('cpu'),optimizers=optimizers)
        self.assertTrue(torch.isfinite(torch.tensor(list(metrics.values()))).all())
        self.assertFalse(torch.equal(before,nets['q1_net'].head[-1].weight))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
