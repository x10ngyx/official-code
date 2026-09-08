"""CPU contracts and forward/backward tests; no video/model inference launched."""
import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import types

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from ours4wan21.contracts import (FORCED, MODES, PROTOCOL, TrainingConfig,
                                  observation, state_contract)
from ours4wan21.data import episode, Transitions
from ours4wan21 import local_iql
from ours4wan21.policy import resolve_budget
from ours4wan21.runtime import Controller, reference, apply_policy, integration_functions
from ours4wan21.train import make_networks, make_optimizers, validate_bundle
import torch
from torch.utils.data import DataLoader


class ScriptPolicy:
    mode = 'sea7'
    def __init__(self, action=1):
        self.action, self.calls = action, 0
    def choose(self, state):
        self.calls += 1
        return self.action, .8 if self.action else .2


def trajectory(policy, budget, *, mismatch=False):
    controller = Controller(policy, budget)
    controller.set_scheduler_sigmas(torch.linspace(.99, 0., 51))
    grid = torch.tensor([2, 2, 2])
    for step in range(50):
        feature = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4) / 20 + 1 + step / 100
        for branch in ('cond', 'uncond'):
            x = feature + .1 if mismatch and step == 5 and branch == 'uncond' else feature
            reuse = controller.plan_step(branch=branch, step_index=step, num_steps=50,
                                         feature=x, grid_size=grid)
            if reuse:
                controller.reuse_residual(branch, step)
            else:
                controller.record_recompute(branch, step, torch.full_like(feature, step + (1 if branch == 'cond' else 2)))
    return controller


def bundle(mode):
    rows = []
    for budget, quality, action in ((20, 21., 1), (30, 23., 0), (25, 19., 1), (15, 24., 0)):
        policy = ScriptPolicy(action)
        policy.mode = mode
        rows.append(episode(trajectory(policy, budget).decisions, quality, mode))
    tensors = {k: torch.cat([row[k] for row in rows]) for k in rows[0]}
    return dict(tensors=tensors, train_indices=torch.arange(100), val_indices=torch.arange(100, 200),
        manifest=dict(schema='ours4wan21_training_data_v1', state=state_contract(mode),
            protocol=PROTOCOL, quality_mode='absolute', reward_timing='terminal_only',
            sources=[dict(trajectory_id=f't{i}', sample_id=f'p{i//2}', split='train' if i < 2 else 'evaluation') for i in range(4)],
            prompt_splits={'p0': 'train', 'p1': 'evaluation'}))


class RLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_kernels_match_local_sources(self):
        lock = json.loads((PROJECT / 'local_training_lock.json').read_text())
        source = (PROJECT / 'ours4wan21/local_iql.py').read_text()
        lines = source.splitlines(keepends=True)
        actual = {}
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
                actual[node.name] = hashlib.sha256(''.join(lines[start:node.end_lineno]).encode()).hexdigest()
        expected = {k: v for entry in lock['sources'].values() for k, v in entry['symbols'].items()}
        self.assertEqual(actual, expected)

    def test_scalar_modes_and_order(self):
        args = dict(step=5, budget=20, used=2, consecutive=1, cached_valid=True,
                    adjacent=.2, accumulated=.3)
        five, seven = [observation(mode, **args) for mode in ('scalar5', 'sea7')]
        self.assertTrue(torch.equal(five, seven[2:]))
        self.assertTrue(torch.allclose(five, torch.tensor([1., 5/49, .4, .04, .02])))

    def test_reserved_latent_fails(self):
        with self.assertRaises(NotImplementedError):
            state_contract('sea7_latent')
        with self.assertRaises(NotImplementedError):
            episode([], 22., 'sea7_latent')

    def test_native_boundaries_are_wan21_only(self):
        policy = ScriptPolicy(1)
        c = trajectory(policy, 48)
        self.assertEqual([d['step_index'] for d in c.decisions if d['branch']=='cond' and d['action']=='recompute'], [0,49])
        self.assertEqual(policy.calls, 0)
        self.assertEqual(c.decisions[64]['action'], 'reuse')

    def test_all_k_and_policy_extremes(self):
        for budget in range(49):
            for preferred in (0, 1):
                used = 0
                for step in range(50):
                    action, _ = local_iql.required_hard_budget_action(step_index=step,
                        used_skips=used, skip_budget=budget, num_steps=50, forced_steps=FORCED)
                    used += preferred if action is None else action
                self.assertEqual(used, budget)
        for budget in (0, 1, 25, 47, 48):
            for preferred in (0, 1):
                c = trajectory(ScriptPolicy(preferred), budget)
                self.assertEqual(c.summary()['step_reuse'], budget)
                self.assertEqual(c.used, {'cond': budget, 'uncond': budget})

    def test_offline_online_state_action_mask_parity(self):
        for mode in ('scalar5', 'sea7'):
            policy = ScriptPolicy()
            policy.mode = mode
            c = trajectory(policy, 25)
            e = episode(c.decisions, 22., mode)
            self.assertTrue(torch.equal(e['state'], torch.tensor([d['state'] for d in c.decisions[::2]])))
            self.assertEqual(e['actor_mask'].tolist(), [d['actor_mask'] for d in c.decisions[::2]])
            self.assertEqual(e['reward'].nonzero().flatten().tolist(), [49])
            self.assertEqual(float(e['reward'][-1]), 22.)
            self.assertTrue(torch.equal(e['next_state'][:-1], e['state'][1:]))
            self.assertTrue(torch.equal(e['next_state'][-1], e['state'][-1]))

    def test_mixed_cfg_fails_online(self):
        with self.assertRaises(RuntimeError):
            trajectory(ScriptPolicy(), 25, mismatch=True)

    def test_malformed_offline_trace_fails(self):
        rows = trajectory(ScriptPolicy(), 25).decisions
        for edit in ('branch', 'execution', 'accumulated_distance_before', 'stored_feature'):
            bad = deepcopy(rows)
            bad[8][edit] = {'branch':'uncond', 'execution':None,
                           'accumulated_distance_before':100., 'stored_feature':'raw'}[edit]
            with self.assertRaises(ValueError, msg=edit):
                episode(bad, 22., 'sea7')

    def test_sea_filter_and_residuals_are_inherited(self):
        self.assertIs(Controller._filter_feature, reference.SeaCacheController._filter_feature)
        self.assertIs(Controller.record_recompute, reference.SeaCacheController.record_recompute)
        self.assertIs(Controller.reuse_residual, reference.SeaCacheController.reuse_residual)
        c = trajectory(ScriptPolicy(), 48)
        self.assertFalse(torch.equal(c.residuals['cond'], c.residuals['uncond']))

    def test_reset_and_pending_validation(self):
        c = trajectory(ScriptPolicy(), 25)
        c.reset()
        self.assertFalse(c.previous_features)
        self.assertFalse(c.residuals)
        self.assertEqual(c.used['cond'], 0)
        with self.assertRaises(RuntimeError):
            c.summary()
        with self.assertRaises(ValueError):
            c.set_scheduler_sigmas(None)

    def test_bundle_rejects_corruption_and_leakage(self):
        b = bundle('sea7')
        validate_bundle(b, 'sea7')
        for key in ('state', 'next_state', 'actor_mask', 'done', 'reward'):
            bad = deepcopy(b)
            bad['tensors'][key][3] += 1
            with self.assertRaises(ValueError, msg=key):
                validate_bundle(bad, 'sea7')
        b['manifest']['sources'][2]['sample_id'] = 'p0'
        with self.assertRaises(ValueError):
            validate_bundle(b, 'sea7')

    def test_train_only_normalization(self):
        x = torch.tensor([[0., 1.], [2., 3.], [100., 1000.]])
        n = local_iql.compute_normalizer(x, [0,1])
        self.assertTrue(torch.equal(n['mean'], torch.tensor([1.,2.])))
        self.assertTrue(torch.equal(n['std'], torch.ones(2)))

    def test_real_iql_backward_both_modes(self):
        for mode in ('scalar5', 'sea7'):
            b = bundle(mode)
            cfg = TrainingConfig()
            torch.manual_seed(42)
            _, nets = make_networks(mode, cfg, 'cpu')
            opt = make_optimizers(nets, cfg)
            initial = {k: next(n.parameters()).detach().clone() for k,n in nets.items()}
            loader = DataLoader(Transitions(b['tensors'], b['train_indices']), batch_size=64)
            metrics = local_iql._run_epoch(loader=loader, **nets, args=cfg,
                device=torch.device('cpu'), optimizers=opt)
            self.assertGreater(metrics['actor_examples'], 0)
            for key in nets:
                self.assertFalse(torch.equal(initial[key], next(nets[key].parameters())), key)

    def test_no_actor_update_on_forced_only_batch(self):
        cfg = TrainingConfig()
        _, nets = make_networks('scalar5', cfg, 'cpu')
        opt = make_optimizers(nets, cfg)
        initial = deepcopy(nets['policy_net'].state_dict())
        b = bundle('scalar5')
        b['tensors']['actor_mask'].zero_()
        metrics = local_iql._run_epoch(loader=DataLoader(Transitions(b['tensors'], [0,49]), batch_size=2),
            **nets, args=cfg, device=torch.device('cpu'), optimizers=opt)
        self.assertEqual(metrics['actor_examples'], 0)
        self.assertTrue(all(torch.equal(v, nets['policy_net'].state_dict()[k]) for k,v in initial.items()))

    def test_budget_requires_local_calibration(self):
        self.assertEqual(resolve_budget(skip_budget=25), 25)
        for value in (-1,49,1.5,True):
            with self.assertRaises(ValueError):
                resolve_budget(skip_budget=value)
        with self.assertRaises(ValueError):
            resolve_budget(target_speedup=2.)

    def test_sampler_rejects_foreign_protocol(self):
        pipe = SimpleNamespace(model=SimpleNamespace(), t5_cpu=False, rank=0,
                               sp_size=1, param_dtype=torch.bfloat16)
        apply_policy(pipe, ScriptPolicy(), 25)
        for args in (dict(frame_num=45), dict(seed=1), dict(offload_model=True), dict(sample_solver='dpm++')):
            with self.assertRaises(ValueError):
                pipe.generate('test', **args)

    def test_actual_forward_matches_seacache_at_budget_extremes(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.modulation = torch.nn.Parameter(torch.zeros(1,6,4))
                self.norm1 = torch.nn.LayerNorm(4)
            def forward(self,x,**kw):
                return x + kw['context'].mean() * .02 + .1
        class Head(torch.nn.Module):
            def forward(self,x,e):
                return x + e[:,None,:] * .01
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model_type = 't2v'
                self.freq_dim = self.dim = 4
                self.text_len = 2
                self.freqs = torch.ones(1)
                self.patch_embedding = torch.nn.Conv3d(4,4,1)
                self.time_embedding = torch.nn.Linear(4,4)
                self.time_projection = torch.nn.Linear(4,24)
                self.text_embedding = torch.nn.Linear(4,4)
                self.blocks = torch.nn.ModuleList([Block(),Block()])
                self.head = Head()
            def unpatchify(self,x,grid):
                return [x[0]]
        module = types.ModuleType('wan.modules.model')
        module.sinusoidal_embedding_1d = lambda dim,t: t[:,None].expand(-1,dim).float()
        forward = integration_functions().seacache_forward
        for budget, threshold in ((0,1e-30),(48,1e30)):
            torch.manual_seed(42)
            left = Model()
            right = deepcopy(left)
            left.seacache_controller = Controller(ScriptPolicy(),budget)
            right.seacache_controller = reference.SeaCacheController(reference.SeaCacheConfig(threshold))
            with patch.dict(sys.modules,{'wan.modules.model':module}), torch.no_grad():
                for video in range(2):
                    for model in (left,right):
                        model.seacache_controller.reset()
                        model.seacache_controller.set_scheduler_sigmas(torch.linspace(.99,0.,51))
                    for step in range(50):
                        latent = torch.arange(32).reshape(4,2,2,2).float()/20 + 1 + step*.001
                        for branch in ('cond','uncond'):
                            kw=dict(x=[latent],t=torch.tensor([50-step]),
                                context=[torch.full((2,4),1. if branch=='cond' else -1.)],seq_len=8,
                                seacache_branch=branch,seacache_step_index=step,seacache_num_steps=50)
                            a,b=forward(left,**kw),forward(right,**kw)
                            self.assertTrue(torch.equal(a[0],b[0]))
                    self.assertEqual(left.seacache_controller.summary()['step_reuse'],budget)


if __name__ == '__main__':
    unittest.main()
