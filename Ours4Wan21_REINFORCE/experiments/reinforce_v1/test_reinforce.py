"""Algorithm, on-policy identity, causal features, and atomic recovery tests."""
import copy
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import torch

from core import (CNN, Config, DIM, RunningMoments, encode_normalized, new_checkpoint,
                  policy_version, reinforce_update, required_hard_budget_action,
                  validate_checkpoint, validate_episode)
from runtime_reinforce import Controller, G1History, Policy
from features import pool
from artifacts import complete, dump, identity, load_prompts, make_plan, save, seal, sha, training_position

torch.set_num_threads(1)


def synthetic_episode(checkpoint, k=30, seed=12, reward=25.):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn((50, DIM), generator=g) * .1
    raw[0, :-7] = 0
    raw[:, -7:-5] = raw[:, -7:-5].abs()
    raw[[0, 49], -7:-5] = 0
    actor = CNN('G1', 2).eval()
    actor.load_state_dict(checkpoint['policy_net'])
    actions, masks, logps = [], [], []
    used = consecutive = 0
    with torch.no_grad():
        for t in range(50):
            raw[t, -5:] = torch.tensor([float(t > 0), t/49, k/50, used/50, consecutive/50])
            required, _ = required_hard_budget_action(step_index=t, used_skips=used, skip_budget=k, num_steps=50)
            if required is None:
                x = encode_normalized(raw[t], checkpoint['normalizer']).float()
                logits = actor(x[None])[0]
                a = int(torch.rand((), generator=g) < logits.softmax(-1)[1])
                lp = float(logits.log_softmax(-1)[a])
            else:
                a, lp = required, 0.
            actions.append(a); masks.append(required is None); logps.append(lp)
            used += a; consecutive = consecutive + 1 if a else 0
    return dict(trajectory_id=f'synthetic_{seed}_{k}', sample_id=f'prompt_{seed}',
        policy_version=checkpoint['version'], run_id=checkpoint['run_id'],
        batch_index=checkpoint['batch_index'], split='train', k=k,
        action_mode='policy_categorical', sampling_seed=seed, raw=raw,
        inputs=encode_normalized(raw, checkpoint['normalizer']), actions=torch.tensor(actions),
        free=torch.tensor(masks), logp=torch.tensor(logps), reward=reward)


class Tests(unittest.TestCase):
    def test_epoch_coverage_and_resume(self):
        config = dict(training={'seed': 42}, epochs=8, batch_size=32,
                      prompts=[dict(sample_id=str(i), prompt=str(i), split='train' if i < 800 else 'test')
                               for i in range(900)], baselines={}, gpus=[0, 1, 2, 3])
        orders = []
        for epoch in range(8):
            rows = []
            for batch in range(epoch * 25, (epoch + 1) * 25):
                jobs = make_plan(config, batch, 'checkpoint', 'sha', 'version')
                self.assertEqual(jobs, make_plan(config, batch, 'checkpoint', 'sha', 'version'))
                self.assertEqual(len(jobs), 32)
                self.assertTrue(all(j['split'] == 'train' and 20 <= j['k'] <= 40 for j in jobs))
                rows.extend(j['sample_id'] for j in jobs)
            self.assertEqual(len(set(rows)), 800)
            self.assertEqual(set(rows), {str(i) for i in range(800)})
            orders.append(rows)
        self.assertNotEqual(orders[0], orders[1])
        self.assertEqual(training_position(config, 200)['completed_trajectories'], 6400)
        self.assertEqual(training_position(config, 200)['completed_epochs'], 8)

    def test_epoch_retains_last_partial_batch(self):
        config = dict(training={'seed': 42}, epochs=2, batch_size=32,
                      prompts=[dict(sample_id=str(i), prompt=str(i), split='train') for i in range(35)],
                      baselines={}, gpus=[0])
        self.assertEqual([len(make_plan(config, b, 'p', 's', 'v')) for b in range(4)], [32, 3, 32, 3])
        self.assertEqual(training_position(config, 2)['completed_trajectories'], 35)
        self.assertEqual(training_position(config, 4)['completed_trajectories'], 70)

    @classmethod
    def setUpClass(cls):
        cls.p = new_checkpoint(Config(), 'unit-test', smoke_only=True)
        cls.episodes = [synthetic_episode(cls.p, 20, 12, 23.), synthetic_episode(cls.p, 40, 13, 27.)]

    def test_identity_then_population_moments_and_zero_variance(self):
        m = RunningMoments(3)
        torch.testing.assert_close(m.normalizer()['mean'], torch.zeros(3), rtol=0, atol=0)
        torch.testing.assert_close(m.normalizer()['std'], torch.ones(3), rtol=0, atol=0)
        x = torch.tensor([[1., 2., 7.], [3., 4., 7.], [6., 8., 7.]], dtype=torch.float64)
        m.update(x[:1]); m.update(x[1:])
        torch.testing.assert_close(m.mean, x.mean(0))
        torch.testing.assert_close(m.m2/m.count, x.var(0, unbiased=False))
        self.assertEqual(m.count, 3)
        self.assertAlmostEqual(float(m.normalizer()['std'][-1]), 1e-6)

    def test_gradient_matches_direct_trajectory_objective(self):
        # Independent unchunked reference, including differing free-action counts.
        p = copy.deepcopy(self.p)
        p['config'] = vars(replace(Config(), grad_clip=1e9))
        p['reward_baselines'] = {'20': 20., '40': 29.}
        actor = CNN('G1', 2).eval(); actor.load_state_dict(p['policy_net'])
        opt = torch.optim.AdamW(actor.parameters(), lr=1e-4, weight_decay=.01)
        opt.load_state_dict(p['optimizer'])
        loss = 0
        for e in self.episodes:
            mask = e['free']
            lp = actor(e['inputs'][mask].float()).log_softmax(-1).gather(1, e['actions'][mask, None]).sum()
            loss = loss - (e['reward'] - p['reward_baselines'][str(e['k'])]) * lp / len(self.episodes)
        loss.backward(); opt.step()
        result = reinforce_update(p, self.episodes)
        self.assertAlmostEqual(result['metrics']['loss'], float(loss), places=4)
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(result['policy_net'][name], value, rtol=1e-4, atol=2e-6)
        self.assertEqual(result['metrics']['optimizer_steps'], 1)
        self.assertEqual(result['moments']['count'], 100)
        self.assertAlmostEqual(result['reward_baselines']['20'], 20.3)
        self.assertAlmostEqual(result['reward_baselines']['40'], 28.8)
        self.assertEqual(p['moments']['count'], 0)  # input checkpoint was not mutated

    def test_two_batches_and_saved_checkpoint_resume(self):
        one = reinforce_update(self.p, self.episodes)
        next_eps = [synthetic_episode(one, 20, 31), synthetic_episode(one, 40, 32)]
        reference = reinforce_update(one, next_eps)
        with tempfile.TemporaryDirectory() as tmp:
            save(Path(tmp)/'checkpoint.pt', one)
            loaded = torch.load(Path(tmp)/'checkpoint.pt', weights_only=False)
            restored = reinforce_update(loaded, next_eps)
        self.assertEqual(reference['version'], restored['version'])
        self.assertEqual(restored['moments']['count'], 200)
        self.assertEqual(restored['batch_index'], 2)
        self.assertEqual(restored['metrics']['parent_normalizer_count'], 100)
        with self.assertRaisesRegex(ValueError, 'stale'):
            reinforce_update(one, self.episodes)

    def test_reject_changed_normalizer_behavior_and_eval_rows(self):
        for mutate in (lambda e: e.update(split='test'),
                       lambda e: e.update(action_mode='policy_argmax'),
                       lambda e: e['logp'].__setitem__(e['free'], -100.),
                       lambda e: e['inputs'].__setitem__((1, 0), 100.)):
            ep = copy.deepcopy(self.episodes[0]); mutate(ep)
            with self.assertRaises(ValueError):
                reinforce_update(self.p, [ep])
        changed = copy.deepcopy(self.p); changed['normalizer']['mean'][0] = 1
        with self.assertRaises(ValueError):
            validate_checkpoint(changed)
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            reinforce_update(self.p, [self.episodes[0]] * 2)

    def test_forced_masks_all_budgets_and_cfg_shared_action(self):
        class FakePolicy:
            mode='sea7'; current_feature=None; version='test'
            def reset_measurements(self): self.calls=0
            def choose(self, state): self.calls+=1; return self.calls % 2, .5
        class FakeHistory:
            def __init__(self): self.actions=[]
            def commit(self, a): self.actions.append(a)
        for k in [0, *range(20,41), 48]:
            p = FakePolicy(); c = Controller(p, k)
            c.feature_history = FakeHistory()
            c.set_scheduler_sigmas(torch.linspace(1., .01, 50))
            c._filter_feature = lambda feature, *args: feature
            for t in range(50):
                p.current_feature = torch.zeros(DIM-7)
                for branch in ('cond', 'uncond'):
                    reuse = c.plan_step(branch=branch, step_index=t, num_steps=50,
                        feature=torch.ones(1,4,16), grid_size=torch.tensor([1,2,2]))
                    if reuse: c.reuse_residual(branch, t)
                    else: c.record_recompute(branch, t, torch.ones(1,4,16))
            summary = c.summary()
            self.assertEqual(summary['step_reuse'], k)
            self.assertEqual(summary['actor_queries'], p.calls)
            self.assertEqual(sum(c.feature_history.actions), k)
            self.assertEqual(len(c.raw_states), 50)
            c.reset(); self.assertFalse(c.residuals); self.assertEqual(c.raw_states, [])

    def test_g1_history_uses_actual_cached_state(self):
        h = G1History()
        x = torch.arange(21, dtype=torch.float32)[None,:,None,None].expand(16,21,60,104).contiguous()/21
        z0 = h.observe(x, 0, 1.); self.assertEqual(z0.count_nonzero(), 0); h.commit(0)
        h.observe(x+1, 1, .9); h.commit(1)
        observed = h.observe(x+2, 2, .8)
        parts = [pool(z.half().float().unsqueeze(0)) for z in (x+2, x+1, x)]
        expected = torch.cat([p[0] for p in parts]+[p[1] for p in parts])
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        h.commit(0)
        self.assertTrue(torch.equal(h.cache, h.current))

    def test_sampling_rng_independent_of_wan_and_reset(self):
        p = Policy(self.p, device='cpu', sampling_seed=71, allow_smoke=True)
        p.current_feature = torch.zeros(DIM-7)
        scalar = torch.tensor([.1,.2,1.,.5,.6,.2,.1])
        first = [p.choose(scalar)[0] for _ in range(40)]
        torch.manual_seed(999); torch.randn(100)
        p.reset_measurements(); p.set_sampling(71)
        second = [p.choose(scalar)[0] for _ in range(40)]
        self.assertEqual(first, second); self.assertEqual(set(first), {0,1})
        p.reset_measurements(); self.assertEqual(p.log_probs, [])

    def test_plan_iid_inclusive_k_train_only_and_reproducible(self):
        cfg = dict(training={'seed':42}, batch_size=21000, baselines={}, gpus=[{}],
            prompts=[dict(sample_id='a',prompt='A',split='train'), dict(sample_id='v',prompt='V',split='val')])
        jobs = make_plan(cfg, 0, '/tmp/policy.pt', 'sha', 'version')
        self.assertEqual(jobs, make_plan(cfg, 0, '/tmp/policy.pt', 'sha', 'version'))
        self.assertEqual({j['k'] for j in jobs}, set(range(20,41)))
        self.assertTrue(all(j['split']=='train' for j in jobs))
        counts = [sum(j['k']==k for j in jobs) for k in range(20,41)]
        self.assertLess(max(abs(n-1000) for n in counts), 150)

    def test_sealed_update_is_not_applied_twice(self):
        from run import commit_update
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); models = root/'models'; models.mkdir()
            config = dict(model_dir=str(models), run_id=self.p['run_id'])
            save(models/'batch_0000.pt', self.p)
            jobs = []
            for i, ep in enumerate(self.episodes):
                job=dict(id=ep['trajectory_id'], output=f'episode{i}')
                folder=root/job['output']; folder.mkdir()
                save(folder/'episode.pt', ep)
                seal(folder, identity(config, job), ['episode.pt']); jobs.append(job)
            first = commit_update(root, config, 0, jobs)
            checkpoint_sha = sha(models/'batch_0001.pt')
            (root/'batches/0000/update.json').unlink()  # Simulate crash after checkpoint commit.
            second = commit_update(root, config, 0, jobs)
            self.assertEqual(first['version'], second['version'])
            self.assertEqual(checkpoint_sha, sha(models/'batch_0001.pt'))
            with (root/jobs[0]['output']/'episode.pt').open('ab') as f: f.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                commit_update(root, config, 0, jobs)

    def test_split_text_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'prompts.json'
            dump(path, [dict(sample_id='a',prompt='same prompt',split='train'),
                        dict(sample_id='b',prompt='same  prompt',split='test')])
            with self.assertRaisesRegex(ValueError, 'leaks'):
                load_prompts(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
