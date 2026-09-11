import argparse,importlib.util,json,math,tempfile,time,unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import CNN,ARCH,GROUPS,dimension,encode_normalized,FrozenPolicy
from features import History,pool
from pipeline import CachedDataset,CFG,PREVIOUS,ROOT,rank_checkpoints
from ours4wan21.local_iql import _run_epoch,PolicyNet,IQLModelConfig
class Contracts(unittest.TestCase):
    def test_mlp_exact_original_architecture(self):
        for g in ['SEA7','G1_MLP']:
            model=CNN(g,2);old=PolicyNet(IQLModelConfig(dimension(g),256,3,0.))
            self.assertEqual(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in old.parameters()))
            old.net.load_state_dict(model.net.state_dict())
            x=torch.randn(3,dimension(g));torch.testing.assert_close(model(x),old(x),rtol=0,atol=0)
    def test_larger_cnn_same_weights_and_scalar_checkpoint(self):
        p=Path(__file__).resolve().parents[1]/'cnn_mixed3500_v1/model.py';spec=importlib.util.spec_from_file_location('original_g1_model',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        net=CNN('G1_CNN_16x20',2);original=m.CNN('G1',2)
        self.assertEqual(sum(p.numel() for p in net.parameters()),316034)
        self.assertEqual({k:tuple(v.shape) for k,v in net.state_dict().items()},{k:tuple(v.shape) for k,v in original.state_dict().items()})
        for g in GROUPS:
            a=CNN(g,2);norm={'mean':torch.zeros(dimension(g)),'std':torch.ones(dimension(g))};x=torch.randn(2,dimension(g))
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'p.pt';torch.save(dict(schema='ours21_g1_ablation_checkpoint_v1',group=g,architecture=ARCH[g],policy_net=a.state_dict(),normalizer=norm),p)
                torch.testing.assert_close(FrozenPolicy(p)(x),a(encode_normalized(x,norm).float()),rtol=0,atol=0)
            self.assertEqual(CNN(g,1)(x).shape,(2,))
    def test_variance_before_spatial_pool(self):
        z=torch.tensor([[[[[1.,-1.]], [[-1.,1.]]]]]);a,b=pool(z)
        self.assertTrue(torch.equal(b[:320],torch.zeros(320)));self.assertTrue(torch.equal(b[320:],torch.ones(320)))
        self.assertEqual(a.numel(),1280)
    def test_real_latents_layout_history_and_reuse(self):
        row=json.loads((PREVIOUS/'records.json').read_text())[0];tr=json.loads(Path(row['trace']).read_text());h=History()
        xs=[torch.load(r['latent_path'],weights_only=True) for r in tr['step_records'][:3]]
        self.assertFalse(h.observe(xs[0],0,tr['step_records'][0]['sigma']).any());h.commit(0)
        z=h.observe(xs[1],1,tr['step_records'][1]['sigma']);parts=[pool(x.unsqueeze(0).float()) for x in [xs[1],xs[0],xs[0]]]
        expected=torch.cat([p[0] for p in parts]+[p[1] for p in parts]);torch.testing.assert_close(z,expected,rtol=0,atol=0)
        h.commit(1);cache=h.cache.clone();h.observe(xs[2],2,tr['step_records'][2]['sigma']);torch.testing.assert_close(h.cache,cache)
        self.assertEqual(z.numel()+7,dimension('G1_CNN_16x20'))
        # Same pool at 8x8 reproduces archived GPU features within FP32 CPU/GPU reduction rounding.
        small=[pool(x.unsqueeze(0).float(),(8,8)) for x in [xs[1],xs[0],xs[0]]]
        target=torch.load(PREVIOUS/'raw_rows/0000.pt',weights_only=True)['G1'][1,:-7]
        torch.testing.assert_close(torch.cat([p[0] for p in small]+[p[1] for p in small]),target,rtol=1e-5,atol=5e-7)
    def test_sea7_source_and_selection(self):
        labels=torch.load(PREVIOUS/'labels.pt',weights_only=False);origin=torch.load(PREVIOUS/'cache/G1/normalizer.pt',weights_only=True);norm={k:v[-7:] for k,v in origin.items()}
        cache=np.load(PREVIOUS/'cache/G1/features.npy',mmap_mode='r');idx=torch.tensor([0,1,49,50,50000,174999])
        torch.testing.assert_close(encode_normalized(labels['tensors']['state'][idx],norm),torch.from_numpy(np.array(cache[idx.numpy(),-7:],copy=True)),rtol=0,atol=0)
        epochs=list(range(170,201));q=np.ones((31,4,2));act=np.zeros((31,4),dtype=np.int64);logs={e:{'val':{'pi_loss':1.,'q_loss':1.}} for e in epochs}
        _,ranked,s=rank_checkpoints(epochs,q,act,logs);self.assertEqual([r['epoch'] for r in ranked],list(range(171,200)))
        self.assertEqual((CFG.epochs,CFG.tau,CFG.beta,CFG.weight_max),(200,.7,1.5,30.))

def gpu_smoke(g):
    torch.manual_seed(42);torch.cuda.reset_peak_memory_stats()
    nets={k:CNN(g,1 if k=='value_net' else 2).cuda() for k in ('value_net','q1_net','q2_net','target_q1','target_q2','policy_net')}
    for k in ('q1','q2'):nets['target_'+k].load_state_dict(nets[k+'_net'].state_dict())
    opts=tuple(torch.optim.AdamW(p,lr=CFG.lr,weight_decay=CFG.weight_decay) for p in (nets['value_net'].parameters(),list(nets['q1_net'].parameters())+list(nets['q2_net'].parameters()),nets['policy_net'].parameters()))
    x=torch.randn(300,dimension(g)).half();labels={k:torch.zeros(300) for k in ('action','reward','done','actor_mask')};labels['action'][1::2]=1;labels['actor_mask'][:]=1;labels['done'][49::50]=1;labels['reward'][49::50]=25
    loader=DataLoader(CachedDataset(x,labels,torch.arange(256)),batch_size=256,pin_memory=True)
    before=next(nets['policy_net'].parameters()).detach().clone();times=[]
    for i in range(3):
        torch.cuda.synchronize();start=time.perf_counter();metrics=_run_epoch(loader=loader,**nets,args=CFG,device=torch.device('cuda'),optimizers=opts);torch.cuda.synchronize();times.append(time.perf_counter()-start);assert all(math.isfinite(v) for v in metrics.values())
    assert not torch.equal(before,next(nets['policy_net'].parameters()))
    out=ROOT/'prelaunch';out.mkdir(parents=True,exist_ok=True);(out/'README.md').write_text('Batch256 GPU backward checks with synthetic standardized features; not a full-epoch training estimate or production checkpoint.\n')
    result=dict(group=g,status='pass',batch_size=256,optimizer_updates=3,batch_wall_seconds=times,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,actor_parameters=sum(p.numel() for p in nets['policy_net'].parameters()),metrics=metrics,smoke_only=True)
    (out/f'{g}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('--gpu-smoke',choices=list(GROUPS));a=p.parse_args()
    if a.gpu_smoke:gpu_smoke(a.gpu_smoke)
    else:unittest.main(argv=['test_contract'])
