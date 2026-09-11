"""Causality and cache semantics checks on the independent compact prototype."""
import unittest
import sys
from pathlib import Path
import torch
from features import CompactHistory, DIMS
from prepare import select_group

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tests'))
from test_rl import ScriptPolicy
from ours4wan21.runtime import Controller


class CompactTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.x = torch.randn(16, 3, 4, 6)

    def test_quantized_live_and_archive_agree(self):
        for group in DIMS:
            live, archived = CompactHistory(group), CompactHistory(group)
            for step in range(50):
                x = self.x + step * .01
                a = live.observe(x, step, 1 - step / 50)
                b = archived.observe(x.half(), step, 1 - step / 50)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertEqual(a.shape, (1, DIMS[group]))
                action = int(step not in (0, 49) and step % 4 != 0)
                live.commit(action)
                archived.commit(action)

    def test_cache_updates_only_after_recompute(self):
        history = CompactHistory('a')
        history.observe(self.x, 0, 1.)
        history.commit(0)
        a = history.observe(self.x, 1, .9)
        self.assertTrue(torch.equal(a.reshape(1, 16, 4)[..., 0], torch.zeros(1, 16)))
        history.commit(1)
        history.observe(self.x + 1, 2, .8)
        torch.testing.assert_close(history.cache[0], self.x.half().float())
        history.commit(0)
        a = history.observe(self.x + 1, 3, .7).reshape(1, 16, 4)
        self.assertTrue(torch.equal(a[..., 0], torch.zeros(1, 16)))

    def test_identical_prefix_ignores_future_and_current_action(self):
        histories = [CompactHistory(), CompactHistory()]
        values = []
        for history in histories:
            history.observe(self.x, 0, 1.)
            history.commit(0)
            values.append(history.observe(self.x + .2, 1, .9).clone())
        histories[0].commit(0)
        histories[1].commit(1)
        histories[0].observe(self.x * 20, 2, .8)
        histories[1].observe(self.x / 20, 2, .8)
        torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)

    def test_zero_norm_and_first_valid_history(self):
        history = CompactHistory()
        for step in range(3):
            value = history.observe(torch.zeros_like(self.x), step, 1-step/50)
            self.assertTrue(torch.isfinite(value).all())
            self.assertFalse(value.any())
            history.commit(0)

    def test_trend_uses_only_previous_valid_values(self):
        history = CompactHistory('ac')
        history.observe(self.x, 0, 1.)
        history.commit(0)
        first = history.observe(self.x + .1, 1, .9)
        self.assertFalse(first[:, -16:].any())
        q1 = history.current_q.clone()
        history.commit(1)
        second = history.observe(self.x + .3, 2, .8)
        delta = history.current_q - q1
        torch.testing.assert_close(second[:, -16:], torch.cat((delta, delta), 1))

    def test_reset_invalidates_cache_and_trend(self):
        history = CompactHistory()
        history.observe(self.x, 0, .9)
        history.commit(0)
        history.observe(self.x+.1, 1, .8)
        history.commit(1)
        value = history.observe(self.x, 2, .95, stage='new_expert')
        self.assertFalse(value.any())
        self.assertEqual(len(history.past), 0)
        with self.assertRaises(ValueError):
            history.commit(1)
        history.commit(0)
        self.assertFalse(history.observe(self.x+.1, 3, .8, stage='new_expert')[:, -16:].any())

    def test_no_implicit_step32_reset_and_final_forced(self):
        history = CompactHistory('a')
        for step in range(50):
            value = history.observe(self.x+step/10, step, 1-step/50)
            if step == 32:
                self.assertTrue(value.any())
            if step == 49:
                with self.assertRaises(ValueError):
                    history.commit(1)
            history.commit(int(step not in (0,49)))

    def test_cannot_skip_or_double_observe(self):
        history = CompactHistory()
        with self.assertRaises(ValueError):
            history.observe(self.x, 1, .9)
        history.observe(self.x, 0, 1.)
        with self.assertRaises(ValueError):
            history.observe(self.x, 1, .9)
        with self.assertRaises(ValueError):
            history.reset_cache()

    def test_shared_extraction_slices_equal_independent_groups(self):
        histories = {g: CompactHistory(g) for g in DIMS}
        for step in range(50):
            values = {g:h.observe(self.x+step*.05,step,1-step/50) for g,h in histories.items()}
            for group in DIMS:
                torch.testing.assert_close(select_group(values['abc'],group),values[group],rtol=0,atol=0)
                histories[group].commit(int(step not in (0,49) and step%3!=0))

    def test_controller_cfg_budget_extremes_and_archive_parity(self):
        sigmas = torch.linspace(.99,.01,50)
        for group in DIMS:
            for budget in (0,23,29,35,48):
                policy=ScriptPolicy()
                policy.mode='sea7_bloc_'+group
                controller=Controller(policy,budget)
                controller.set_scheduler_sigmas(sigmas)
                archive=CompactHistory(group)
                for step in range(50):
                    x=self.x+step*.05
                    expected=archive.observe(x.half(),step,sigmas[step])[0]
                    controller.observe_latent(x,step)
                    feature=torch.arange(32).float().reshape(1,8,4)+1+step*.1
                    actions=[]
                    for branch in ('cond','uncond'):
                        reuse=controller.plan_step(branch=branch,step_index=step,num_steps=50,
                                                   feature=feature,grid_size=torch.tensor([2,2,2]))
                        if reuse:
                            controller.reuse_residual(branch,step)
                        else:
                            controller.record_recompute(branch,step,torch.ones_like(feature))
                        torch.testing.assert_close(torch.tensor(controller.decisions[-1]['state'][:-7]),expected,rtol=0,atol=0)
                        actions.append(reuse)
                    self.assertEqual(actions[0],actions[1])
                    archive.commit(int(actions[0]))
                self.assertEqual(controller.summary()['step_reuse'],budget)
                self.assertEqual(controller.summary()['actor_queries'],policy.calls)
                controller.reset()
                self.assertIsNone(controller.feature_history.cache)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
