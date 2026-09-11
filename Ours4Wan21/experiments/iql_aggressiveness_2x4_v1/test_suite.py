"""Bounded CPU contracts for four levels, frozen subsets and persistent dispatch."""
from copy import deepcopy
import tempfile
import unittest
import threading
import fcntl
from unittest.mock import patch
from common import *
import suite
from generate_worker import job_identity
from ours4wan21.contracts import TrainingConfig

class SuiteTests(unittest.TestCase):
    def test_gpu_lock_waits_for_previous_descriptor_release(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'gpu.lock';owner=path.open('a')
            fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            release=threading.Timer(.15,owner.close);release.start()
            try:
                acquired=suite.acquire_gpu_lock(path,timeout=2.,poll=.01)
                acquired.close()
            finally:release.join();owner.close()

    def test_gpu_lock_does_not_steal_a_busy_device(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'gpu.lock'
            with path.open('a') as owner:
                fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(TimeoutError):suite.acquire_gpu_lock(path,timeout=.05,poll=.01)
            acquired=suite.acquire_gpu_lock(path,timeout=.1);acquired.close()

    def test_only_three_parameters_change_and_levels_increase(self):
        old=asdict(TrainingConfig());previous=[old[k] for k in ('tau','beta','weight_max')]
        for level,profile in LEVELS.items():
            new=asdict(training_config(profile));now=[new[k] for k in ('tau','beta','weight_max')]
            self.assertTrue(all(a<b for a,b in zip(previous,now)))
            self.assertEqual({k for k in new if new[k]!=old[k]},{'tau','beta','weight_max'})
            self.assertEqual(new['epochs'],400);previous=now
        self.assertEqual(training_config(),TrainingConfig())

    def test_eight_groups_and_adopt_third(self):
        groups=[group(f,a) for f in FEATURES for a in LEVELS]
        self.assertEqual(len({g['name'] for g in groups}),8)
        self.assertEqual([g['name'] for g in groups if g['adopted']],['dynamics128_a3'])
        g=group('dynamics128','a3');actual=read(Path(g['training'])/'config.json')
        self.assertTrue(all(actual[k]==v for k,v in g['training_config'].items()))
        self.assertEqual(sorted(sum(suite.SCHEDULE.values(),[])),sorted(g['name'] for g in groups))
        self.assertTrue(all(len(v)==2 for v in suite.SCHEDULE.values()))

    def test_frozen_ten_same_gpu_and_240_jobs(self):
        ref=load_evaluation_bundle(REFERENCE);uuids=gpu_uuids(['0','1','2','3'])
        prompts=freeze_prompts(ref,uuids)
        self.assertEqual(prompts,freeze_prompts(ref,uuids))
        self.assertEqual([sum(r['baseline_gpu_uuid']==u for r in prompts) for u in uuids],[3,3,2,2])
        groups=[group(f,a) for f in FEATURES for a in LEVELS]
        # Dispatch actual baseline checkpoint selections as fixtures, without running generation.
        for g in groups:g['analysis']=g['baseline_analysis']
        config=dict(groups=groups,prompts=prompts,gpu_uuids=uuids)
        with tempfile.TemporaryDirectory(dir=EXP_ROOT,prefix='iql_suite_cpu_') as temp,patch.object(suite,'ROOT',Path(temp)):
            (Path(temp)/'jobs').mkdir();(Path(temp)/'evaluation').mkdir()
            queues=suite.evaluation_jobs(config)
            self.assertEqual([len(queues[g]) for g in range(4)],[72,72,48,48])
            jobs=sum(queues.values(),[]);self.assertEqual(len({j['output'] for j in jobs}),240)
            for g in groups:
                for k in BUDGETS:
                    rows=[j for j in jobs if j['group']==g['name'] and j['skip_budget']==k]
                    self.assertEqual({j['sample_id'] for j in rows},{r['sample_id'] for r in prompts})
                    self.assertTrue(all(j['state_mode']==g['mode'] for j in rows))

    def test_actual_trace_and_reject_changed_budget_or_blocks(self):
        ref=load_evaluation_bundle(REFERENCE);uuids=gpu_uuids(['0','1','2','3'])
        for r in freeze_prompts(ref,uuids):
            gpu=uuids.index(r['baseline_gpu_uuid'])
            for k in BUDGETS:
                d=OLD/'shards'/f'gpu{gpu}'/f'K{k}';sid=r['sample_id']
                trace=read(d/'traces'/f'{sid}.json');timing=read(d/'timings'/f'{sid}.json')
                audited=audit_trace(trace,timing,k);self.assertEqual(len(audited['skip_path']),50)
                with self.assertRaises(AssertionError):audit_trace(trace,timing,k+1)
                bad=deepcopy(timing);bad['calls'][0]['blocks_executed']=1
                with self.assertRaises(AssertionError):audit_trace(trace,bad,k)

    def test_identity_locks_checkpoint_feature_and_budget(self):
        manifest=dict(paths=dict(wan_checkpoint='model'),inputs=dict(flops_profile=dict(sha256='profile')),source_hashes={'x':'hash'})
        job=dict(checkpoint={'sha256':'policy'},state_mode='sea7',skip_budget=23)
        old=job_identity(deepcopy(job),manifest)
        for key,value in [('state_mode','sea7_dynamics_raw_sea128'),('skip_budget',29),('checkpoint',{'sha256':'other'})]:
            changed=deepcopy(job);changed[key]=value
            self.assertNotEqual(old,job_identity(changed,manifest))

if __name__=='__main__':unittest.main()
