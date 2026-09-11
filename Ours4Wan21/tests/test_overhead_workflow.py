"""CPU regression for overhead accounting and the remote training workflow."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import tempfile
from unittest.mock import patch, Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ours4wan21 import contracts
from ours4wan21.local_iql import PolicyNet, IQLModelConfig
from ours4wan21.overhead import network_flops, predictor_fields, aggregate
from ours4wan21.policy import Policy
from ours4wan21.selection import select_rows, completed_rows
from ours4wan21.data import build
from ours4wan21.train import validate_bundle
from ours4wan21.analysis import rank_checkpoints
from test_rl import trajectory, ScriptPolicy
import numpy as np
import torch


def policy(mode):
    # Construct the same runtime object without persisting synthetic weights.
    p = Policy.__new__(Policy)
    p.mode, p.device = mode, torch.device('cpu')
    dim=5 if mode=='scalar5' else 7
    p.net=PolicyNet(IQLModelConfig(dim)).eval()
    p.normalizer=dict(mean=torch.zeros(dim),std=torch.ones(dim))
    p.flops_profile=network_flops(p.net)
    p._event_pairs=[]
    p.reset_measurements()
    return p


class OverheadWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_flops_match_real_calflops(self):
        from calflops import calculate_flops
        for dim,expected in ((5,270336),(7,271360)):
            net=PolicyNet(IQLModelConfig(dim)).eval()
            actual,_,_=calculate_flops(model=net,input_shape=(1,dim),
                output_as_string=False,print_results=False,print_detailed=False)
            self.assertEqual(network_flops(net)['flops_per_call'],actual)
            self.assertEqual(actual,expected)

    def test_actual_actor_calls_reset_and_forced_zero(self):
        for mode in ('scalar5','sea7'):
            p=policy(mode)
            for k in (25,0,48,15):
                c=trajectory(p,k)
                timing=dict(predictor=p.overhead_summary(),pipeline_generate_wall_seconds=10.)
                fields=predictor_fields(timing,c.summary())
                self.assertEqual(fields['predictor_call_count'],c.summary()['actor_queries'])
                self.assertEqual(fields['predictor_tflops'],c.summary()['actor_queries']*p.flops_profile['flops_per_call']/1e12)
                self.assertIsNone(fields['predictor_network_cuda_seconds'])
                self.assertGreaterEqual(fields['predictor_decision_wall_seconds'],fields['predictor_network_host_span_seconds'])
                if k in (0,48):
                    self.assertEqual(fields['predictor_call_count'],0)
                    self.assertEqual(fields['predictor_decision_wall_seconds'],0.)
                self.assertEqual(p.overhead_summary(),timing['predictor'])

    def test_predictor_values_unchanged_by_measurement(self):
        p=policy('sea7')
        x=torch.arange(7,dtype=torch.float32)
        with torch.no_grad():
            logits=p.net(x[None])[0]
        a,prob=p.choose(x)
        self.assertEqual(a,int(logits.argmax()))
        self.assertEqual(prob,float(logits.softmax(-1)[1]))
        self.assertEqual(p.overhead_summary()['call_count'],1)

    def test_missing_or_corrupt_overhead_fails(self):
        p=policy('sea7'); c=trajectory(p,25)
        t=dict(predictor=p.overhead_summary(),pipeline_generate_wall_seconds=10.)
        for key in ('call_count','tflops','decision_wall_seconds'):
            bad=deepcopy(t);bad['predictor'][key]+=1
            with self.assertRaises(ValueError):
                predictor_fields(bad,c.summary())
        with self.assertRaises(ValueError):
            predictor_fields({},c.summary())

    def test_cuda_event_units_and_finalize_only_sync(self):
        p=policy('scalar5');p.device=torch.device('cuda')
        start,end=Mock(),Mock()
        start.elapsed_time.return_value=2.5
        p._event_pairs=[(start,end)]
        p._measurements=[dict(network_host_span_seconds=.001,decision_wall_seconds=.004)]
        result=p.overhead_summary()
        self.assertEqual(result['network_cuda_seconds'],.0025)
        end.synchronize.assert_called_once()
        start.synchronize.assert_not_called()
        p.reset_measurements()
        self.assertEqual(p.overhead_summary()['network_cuda_seconds'],0.)

    def test_ratio_of_sums_aggregation(self):
        rows=[]
        for duration,wall in ((1.,.1),(9.,.2)):
            rows.append(dict(generate_seconds=duration,predictor_call_count=2,predictor_tflops=1e-6,
                predictor_network_cuda_seconds=None,predictor_network_host_span_seconds=wall/2,
                predictor_decision_wall_seconds=wall))
        result=aggregate(rows)
        self.assertAlmostEqual(result['predictor_decision_wall_pct_of_generate'],3.)
        self.assertEqual(result['predictor_call_count_total'],4)

    def test_subset_is_deterministic_and_quality_blind(self):
        rows=[dict(trajectory_id=f't{i:04d}_{j}',sample_id=f'p{i:04d}',
            split='train' if i<2400 else 'val' if i<2700 else 'test',
            policy_family='random_continuous_seacache_threshold',mean_psnr=float(j))
            for i in range(3000) for j in range(3)]
        a=select_rows(rows,strategy='one-per-prompt')
        self.assertEqual(len({r['sample_id'] for r in a}),3000)
        self.assertEqual(sum(r['split']=='train' for r in a),2400)
        for strategy in ('one-per-prompt','uniform-trajectories'):
            a=select_rows(rows,strategy=strategy)
            changed=[dict(r,mean_psnr=-100.) for r in reversed(rows)]
            b=select_rows(changed,strategy=strategy)
            self.assertEqual([r['trajectory_id'] for r in a],[r['trajectory_id'] for r in b])
            self.assertEqual(len(a),3000)
        bad=deepcopy(rows);bad[0]['policy_family']='fixed_seacache_threshold'
        with self.assertRaises(ValueError):
            select_rows(bad,strategy='uniform-trajectories')

    def test_existing_3000_are_all_kept_without_resampling(self):
        rows=[dict(trajectory_id=f't{i:04d}',sample_id=f'p{i//3:04d}',
            split='train' if i<2400 else 'val' if i<2700 else 'test',
            policy_family='random_continuous_seacache_threshold') for i in range(3000)]
        a=select_rows(list(reversed(rows)),strategy='all-completed',seed=42)
        b=select_rows(rows,strategy='all-completed',seed=100)
        self.assertEqual(a,rows)
        self.assertEqual(a,b)
        for wrong in (rows[:-1],rows+[dict(rows[-1],trajectory_id='extra')]):
            with self.assertRaises(ValueError):
                select_rows(wrong,strategy='all-completed')

    def test_completed_batch_does_not_require_unfinished_plan_members(self):
        with tempfile.TemporaryDirectory(dir=contracts.EXP_ROOT) as directory:
            root=Path(directory)
            rows=[dict(trajectory_id=f't{i}',sample_id=f'p{i}',split='train',shard_index=0,
                policy_family='random_continuous_seacache_threshold',protocol={}) for i in range(9)]
            for row in rows[:3]:
                path=root/'completed'/f"{row['trajectory_id']}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(schema='ours4wan21_candidate_complete_v3',trajectory_row=row)))
            available,paths=completed_rows(root,rows)
            self.assertEqual(available,rows[:3])
            self.assertEqual(len(paths),3)
            with self.assertRaises(ValueError):
                completed_rows(root,rows,require_all=True)
            paths['t0'].write_text('{}')
            with self.assertRaises(ValueError):
                completed_rows(root,rows)

    def test_legacy_completion_layout_remains_readable(self):
        with tempfile.TemporaryDirectory(dir=contracts.EXP_ROOT) as directory:
            root=Path(directory)
            row=dict(trajectory_id='t0',sample_id='p0',split='train',shard_index=0,
                policy_family='random_continuous_seacache_threshold',protocol={})
            path=root/'shards/shard_00/candidates/t0/CANDIDATE_COMPLETE.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(schema='ours4wan21_candidate_complete_v3',trajectory_row=row)))
            available,paths=completed_rows(root,[row])
            self.assertEqual(available,[row])
            self.assertEqual(paths['t0'],path)

    def test_native_val_test_split_is_preserved(self):
        decisions=trajectory(ScriptPolicy(),25).decisions
        source=[dict(trajectory_id=f't{i}',sample_id=f'p{i}',split=s)
                for i,s in enumerate(('train','val','test'))]
        with patch('ours4wan21.data.load_completion',side_effect=[(r,decisions,22.,[]) for r in source]):
            b=build([1,2,3],'sea7')
        validate_bundle(b,'sea7')
        self.assertEqual(b['train_indices'].tolist(),list(range(50)))
        self.assertEqual(b['val_indices'].tolist(),list(range(50,100)))
        self.assertEqual(b['test_indices'].tolist(),list(range(100,150)))
        b['val_indices'],b['test_indices']=b['test_indices'],b['val_indices']
        with self.assertRaises(ValueError):
            validate_bundle(b,'sea7')

    def test_local_two_sided_selection_rule(self):
        epochs=[300,301,302,303,304]
        base=np.array([[0.,1.],[2.,3.]],dtype=np.float32)
        q=np.stack([base+delta for delta in [0.,.2,.21,.22,1.]])
        actions=np.ones((5,2),dtype=np.uint8)
        logs={e:dict(val=dict(pi_loss=.5,q_loss=1.)) for e in epochs}
        _,ranked,selection=rank_checkpoints(epochs,q,actions,logs)
        self.assertEqual(selection['selected']['epoch'],302)
        self.assertEqual(selection['gate'],.96)
        self.assertEqual([r['epoch'] for r in ranked],[301,302,303])

    def test_remote_roots_propagate_to_shared_helpers(self):
        env=dict(os.environ,OURS4WAN21_WORKSPACE='/remote/ours21',OURS4WAN21_EXP_BASE='/remote/large/exp',
                 PYTHONPATH=str(contracts.PROJECT))
        script="from ours4wan21.shared import benchmark; from ours4wan21.contracts import MODEL_ROOT; m=benchmark(); print(MODEL_ROOT); print(m.external('/remote/large/exp/run')); print(m.ROOT)"
        r=subprocess.run([sys.executable,'-c',script],env=env,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(r.stdout.splitlines(),['/remote/ours21/models','/remote/large/exp/run','/remote/ours21'])


if __name__=='__main__':
    unittest.main()
