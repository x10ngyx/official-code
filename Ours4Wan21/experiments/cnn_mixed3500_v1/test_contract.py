import io, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from unittest.mock import patch
from dataclasses import asdict
import json
from model import CNN,ARCH,dimension,encode_normalized,FrozenPolicy
from features import History,pool
from pipeline import CachedDataset,CFG
from ours4wan21.local_iql import _run_epoch
from torch.utils.data import DataLoader

class Contracts(unittest.TestCase):
    def test_200epoch_selection_boundaries(self):
        from pipeline import SELECTION_START,rank_checkpoints
        self.assertEqual(CFG.epochs,200);self.assertEqual(SELECTION_START,170)
        epochs=list(range(SELECTION_START,CFG.epochs+1))
        q=np.ones((len(epochs),4,2));actions=np.zeros((len(epochs),4),dtype=np.int64)
        logs={e:{'val':{'pi_loss':1.,'q_loss':1.}} for e in epochs}
        _,ranked,selection=rank_checkpoints(epochs,q,actions,logs)
        self.assertEqual([r['epoch'] for r in ranked],list(range(171,200)))
        self.assertEqual(selection['selected']['epoch'],199)
    def test_explicit_parameter_confirmation_gate(self):
        import pipeline
        with tempfile.TemporaryDirectory() as tmp,patch.object(pipeline,'ROOT',Path(tmp)):
            (Path(tmp)/'config.json').write_text(json.dumps(dict(training=asdict(CFG),architecture=ARCH,selection_start=170,epochs=200)))
            with patch.object(pipeline.time,'sleep',side_effect=RuntimeError('awaiting user')):
                with self.assertRaisesRegex(RuntimeError,'awaiting user'):pipeline.wait_for_training_confirmation('G1')
            marker=Path(tmp)/'TRAINING_PARAMETERS_CONFIRMED.json'
            marker.write_text(json.dumps({'training_signature':'wrong'}))
            with self.assertRaises(ValueError):pipeline.wait_for_training_confirmation('G1')
            marker.write_text(json.dumps({'training_signature':pipeline.training_signature()}))
            pipeline.wait_for_training_confirmation('G1')
    def test_temporal_variance_order(self):
        z=torch.tensor([[[[[1.,-1.]], [[-1.,1.]]]]]);_,b=pool(z)
        self.assertTrue(torch.equal(b[64:],torch.ones(64)))
        self.assertTrue(torch.equal(b[:64],torch.zeros(64)))
    def test_history_and_quantization(self):
        torch.manual_seed(12);h=History();xs=[torch.randn(16,21,60,104) for _ in range(3)]
        v=h.observe(xs[0],0,1.)
        self.assertTrue(all(not z.any() for z in v.values()))
        with self.assertRaises(ValueError):h.observe(xs[1],1,.99)
        h.commit(0);c=h.cache.clone();h.observe(xs[1],1,.99);h.commit(1)
        self.assertTrue(torch.equal(c,h.cache));self.assertTrue(torch.equal(h.previous,xs[1].half().float().unsqueeze(0)))
        out=h.observe(xs[2],2,.98)
        for g,z in out.items():self.assertEqual(z.numel()+7,dimension(g));self.assertTrue(torch.isfinite(z).all())
        # Current deltas must reference previous and last recompute separately.
        x=xs[2].half().float().unsqueeze(0);p1,p2=pool(x-h.previous),pool(x-c)
        expected=torch.cat((p1[0],p2[0],p1[1],p2[1]))
        self.assertTrue(torch.equal(expected,out['G2']))
        h.commit(0);self.assertTrue(torch.equal(h.cache,x))
    def test_successor_indices(self):
        x=torch.arange(100).float().reshape(100,1);lab={k:torch.zeros(100) for k in ('action','reward','done','actor_mask')}
        d=CachedDataset(x,lab,torch.tensor([48,49,50]))
        self.assertEqual(d[0]['next_state'].item(),49);self.assertEqual(d[1]['next_state'].item(),49);self.assertEqual(d[2]['next_state'].item(),51)
    def test_normalizer_and_portable_checkpoint(self):
        torch.manual_seed(42)
        for g,param in [('G1',316034),('G2',314498)]:
            net=CNN(g,2).eval();self.assertEqual(sum(p.numel() for p in net.parameters()),param)
            raw=torch.randn(2,dimension(g));norm={'mean':torch.zeros(dimension(g)),'std':torch.ones(dimension(g))}
            z=encode_normalized(raw,norm).float()
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'test.pt';torch.save(dict(schema='ours21_three_branch_cnn_checkpoint_v1',group=g,architecture=ARCH,normalizer=norm,policy_net=net.state_dict()),path)
                p=FrozenPolicy(path);torch.testing.assert_close(p(raw),net(z),rtol=0,atol=0)
    def test_iql_update_and_restore(self):
        torch.manual_seed(42);g='G2';nets={k:CNN(g,1 if k=='value_net' else 2) for k in ('value_net','q1_net','q2_net','target_q1','target_q2','policy_net')}
        opts=tuple(torch.optim.AdamW(p,lr=CFG.lr,weight_decay=CFG.weight_decay) for p in (nets['value_net'].parameters(),list(nets['q1_net'].parameters())+list(nets['q2_net'].parameters()),nets['policy_net'].parameters()))
        x=torch.randn(50,dimension(g)).half();labels={k:torch.zeros(50) for k in ('action','reward','done','actor_mask')};labels['action'][1::2]=1;labels['done'][-1]=1;labels['reward'][-1]=25;labels['actor_mask'][:]=1
        loader=DataLoader(CachedDataset(x,labels,torch.arange(4)),batch_size=4);before=nets['policy_net'].head[-1].weight.detach().clone()
        m=_run_epoch(loader=loader,**nets,args=CFG,device=torch.device('cpu'),optimizers=opts)
        self.assertTrue(all(np.isfinite(v) for v in m.values()));self.assertFalse(torch.equal(before,nets['policy_net'].head[-1].weight))
        buffer=io.BytesIO();torch.save(dict(net=nets['policy_net'].state_dict(),opt=opts[-1].state_dict()),buffer);buffer.seek(0);p=torch.load(buffer,weights_only=False)
        restored=CNN(g,2);restored.load_state_dict(p['net']);torch.testing.assert_close(restored(x[:2].float()),nets['policy_net'](x[:2].float()),rtol=0,atol=0)

if __name__=='__main__':torch.set_num_threads(1);unittest.main()
