import unittest,json,sys
from pathlib import Path
import torch
from common import ROOT,TRAIN,FORMAL
from adapter import SelectedHistory,Controller,Policy
from features import History
from model import FrozenPolicy

class FakePolicy:
    mode='sea7';group='G2';action_mode='policy_argmax';sampling_seed=None
    def reset_measurements(self):self.calls=0
    def choose(self,state):self.calls+=1;return 1,.9
class FakeHistory:
    def __init__(self):self.actions=[]
    def commit(self,a):self.actions.append(a)

class Tests(unittest.TestCase):
    def test_selected_group_matches_training_extractor(self):
        tr=json.loads(Path(json.loads((TRAIN/'records.json').read_text())[0]['trace']).read_text());full=History();hs={g:SelectedHistory(g) for g in ['G1','G2','G3','G4']}
        for t,pos in enumerate([0,20,40]):
            rec=tr['step_records'][pos];x=torch.load(rec['latent_path'],weights_only=True);sigma=float(rec['sigma']);a=0 if t==0 else 1
            all_groups=full.observe(x,t,sigma)
            for g,h in hs.items():torch.testing.assert_close(h.observe(x,t,sigma),all_groups[g],rtol=0,atol=0);h.commit(a)
            full.commit(a)
    def test_cfg_exact_k_and_reset(self):
        for k in [0,23,29,35,48]:
            p=FakePolicy();c=Controller(p,k);h=FakeHistory();c.feature_history=h
            c.set_scheduler_sigmas(torch.linspace(1.,.01,50));c._filter_feature=lambda feature,*args:feature
            for step in range(50):
                c.saved_features.append(torch.zeros(12288));c.feature_wall_seconds.append(0.)
                for branch in ['cond','uncond']:
                    reuse=c.plan_step(branch=branch,step_index=step,num_steps=50,feature=torch.ones(1,4,16),grid_size=torch.tensor([1,2,2]))
                    if reuse:c.reuse_residual(branch,step)
                    else:c.record_recompute(branch,step,torch.ones(1,4,16))
            s=c.summary();self.assertEqual(s['step_reuse'],k);self.assertEqual(len(h.actions),50);self.assertEqual(sum(h.actions),k);self.assertEqual(s['actor_queries'],p.calls)
            c.reset();self.assertEqual(c.saved_features,[]);self.assertEqual(c.used,{'cond':0,'uncond':0});self.assertFalse(c.residuals)
    def test_real_checkpoint_policy_normalization_parity(self):
        states=torch.load(TRAIN/'raw_rows/0000.pt',weights_only=True)
        for g in ['G1','G2','G3','G4']:
            ckpts=sorted(Path('/mnt/hdd/xiongyuxiang/tmp/models/ours21_cnn_mixed3500_v1',g,'checkpoints').glob('epoch_*.pt'));self.assertTrue(ckpts)
            p=Policy(ckpts[0],device='cpu',state_mode='cnn_'+g);f=FrozenPolicy(ckpts[0]);raw=states[g][1];p.current_feature=raw[:-7]
            action,prob=p.choose(raw[-7:]);logits=f(raw.unsqueeze(0))[0]
            self.assertEqual(action,int(logits.argmax()));self.assertEqual(prob,float(logits.softmax(-1)[1]))
            self.assertEqual(p.overhead_summary()['call_count'],1)

if __name__=='__main__':torch.set_num_threads(1);unittest.main()
