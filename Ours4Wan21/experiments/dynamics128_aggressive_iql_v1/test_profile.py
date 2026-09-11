"""Check explicit profiles and real CPU training checkpoint provenance."""
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'tests'))
from test_rl import bundle
from ours4wan21.contracts import TrainingConfig, training_config, EXP_ROOT, MODEL_ROOT
from ours4wan21.train import train
import torch


class ProfileTests(unittest.TestCase):
    def test_explicit_profile_and_only_three_changes(self):
        self.assertEqual(training_config(),TrainingConfig())
        old,new=map(asdict,(training_config(),training_config('aggressive_v1')))
        self.assertEqual({k:(old[k],new[k]) for k in old if old[k]!=new[k]},
                         dict(tau=(.6,.9),beta=(1.,3.),weight_max=(20.,100.)))
        self.assertEqual(training_config('aggressive_v1',seed=9).seed,9)
        with self.assertRaises(ValueError):training_config('unregistered')
        with self.assertRaises(ValueError):training_config(seed=-1)

    def test_production_requires_explicit_profile(self):
        with self.assertRaisesRegex(ValueError,'explicit IQL profile'):
            train(bundle('sea7'),'sea7',Path('unused'),Path('unused'),
                  config=training_config('aggressive_v1'),device='cpu')
        with self.assertRaisesRegex(ValueError,'explicit IQL profile'):
            train(bundle('sea7'),'sea7',Path('unused'),Path('unused'),
                  config=replace(training_config('aggressive_v1'),lr=.001),
                  device='cpu',iql_profile='aggressive_v1')

    def test_actual_training_records_effective_profile(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=EXP_ROOT,prefix='iql_profile_test_') as results, \
             tempfile.TemporaryDirectory(dir=MODEL_ROOT,prefix='iql_profile_test_') as weights:
            cfg=replace(training_config('aggressive_v1'),epochs=2)
            out=Path(results)/(Path(results).name+'_train');models=Path(weights)/'models'
            self.addCleanup(lambda:(PROJECT/'experiment_results'/out.name).unlink(missing_ok=True))
            final=train(bundle('sea7'),'sea7',out,models,config=cfg,device='cpu',smoke=True,iql_profile='aggressive_v1')
            checkpoint=torch.load(final,map_location='cpu',weights_only=False)
            self.assertEqual(checkpoint['train_config'],asdict(cfg))
            self.assertEqual(checkpoint['iql_profile'],'aggressive_v1')
            self.assertEqual(json.loads((out/'config.json').read_text())['tau'],.9)
            self.assertEqual(len((out/'epoch_metrics.jsonl').read_text().splitlines()),2)
            self.assertEqual(json.loads((out/'TRAINING_COMPLETE.json').read_text())['epochs'],2)


if __name__=='__main__':unittest.main()
