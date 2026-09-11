import argparse, json, time, sys, os, statistics, hashlib
from pathlib import Path
from dataclasses import asdict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from ours4wan21.local_iql import _run_epoch, build_mlp, PolicyNet, QNet, ValueNet, IQLModelConfig
from ours4wan21.contracts import TrainingConfig
from ours4wan21.latent_features import sea_filter
ROOT=Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_cnn_training_benchmark_v1')
WEIGHTS=Path('/mnt/hdd/xiongyuxiang/tmp/models/ours21_cnn_training_benchmark_v1')
SOURCE=Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_increase500_iql2_v1/ours21_increase500_iql2_v1_cache/transitions.pt')

def dump(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False))
def pool(z):
    return F.adaptive_avg_pool3d(z,(4,8,8)).flatten(), torch.cat((F.adaptive_avg_pool2d(z.mean(2),(8,8)),F.adaptive_avg_pool2d(z.var(2,unbiased=False),(8,8))),1).flatten()

def prepare():
    ROOT.mkdir(exist_ok=True); WEIGHTS.mkdir(exist_ok=True)
    (ROOT/'README.md').write_text('# CNN throughput benchmark\n\nfixture.pt is pooled real-trajectory data for timing only; each G*/result.json stores measured epochs. No quality conclusions.\n')
    (WEIGHTS/'README.md').write_text('# Disposable timing checkpoints\n\nPer-worker latest.pt files are throughput fixtures, not production policies.\n')
    b=torch.load(SOURCE,map_location='cpu',weights_only=False)
    i=next(i for i,s in enumerate(b['manifest']['sources']) if s['source_split']=='train')
    source=b['manifest']['sources'][i]; trace_path=next(Path(p) for p in source['files'] if p.endswith('/trace.json'))
    tr=json.loads(trace_path.read_text()); samples={g:[] for g in ['G1','G2','G3','G4']}
    start=time.perf_counter(); prev=cache=None; read_s=compute_s=0.; hashes=[]
    for t,rec in enumerate(tr['step_records']):
        tick=time.perf_counter(); p=Path(rec['latent_path']); raw=p.read_bytes(); hashes.append(hashlib.sha256(raw).hexdigest())
        import io
        x=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True).float().unsqueeze(0).cuda()
        torch.cuda.synchronize(); read_s+=time.perf_counter()-tick;tick=time.perf_counter()
        if t==0: prev=cache=x
        fields=[x,prev,cache]; filtered=[sea_filter(z,rec['sigma']) for z in fields]
        variants=[fields,[x-prev,x-cache],filtered,[filtered[0]-filtered[1],filtered[0]-filtered[2]]]
        for g, zs in zip(samples,variants):
            parts=[pool(z) for z in zs]
            # Separate role concatenation inside each branch, followed by SEA7.
            vec=torch.cat([torch.cat([v[0] for v in parts]),torch.cat([v[1] for v in parts])]).cpu()
            if t==0:vec.zero_()
            samples[g].append(vec)
        torch.cuda.synchronize();compute_s+=time.perf_counter()-tick
        if rec['action']=='recompute':cache=x
        prev=x
    small={k:v[i*50:(i+1)*50].clone() for k,v in b['tensors'].items()}
    for g,values in samples.items():
        v=torch.cat((torch.stack(values),small['state'][:,-7:]),1)
        mean=v.mean(0);std=v.std(0,unbiased=False).clamp_min(1e-6)
        samples[g]=((v-mean)/std).half()
    old=small['state'];samples['MLP']=((old-old.mean(0))/old.std(0,unbiased=False).clamp_min(1e-6)).half()
    torch.save(dict(features=samples,labels=small),ROOT/'fixture.pt')
    dump(ROOT/'prepare.json',dict(source=str(trace_path),source_hashes=source['files'],latent_sha256=hashes,steps=50,read_h2d_seconds=read_s,feature_seconds=compute_s,total_seconds=time.perf_counter()-start,scope='one trajectory; warm/cold disk cache uncontrolled; cannot project full 683GiB disk scan reliably',shapes={k:list(v.shape) for k,v in samples.items()}))
    print((ROOT/'prepare.json').read_text(),flush=True)

class CNN(nn.Module):
    def __init__(self,roles,out):
        super().__init__();self.r=roles;self.n3=16*roles*256;self.n2=32*roles*64
        self.c3=nn.Sequential(nn.Conv3d(16*roles,32,1),nn.SiLU(),nn.Conv3d(32,32,3,padding=1),nn.SiLU(),nn.Conv3d(32,64,3,stride=(1,2,2),padding=1),nn.SiLU(),nn.AdaptiveAvgPool3d((2,2,2)),nn.Flatten())
        self.c2=nn.Sequential(nn.Conv2d(32*roles,32,1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,64,3,stride=2,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d((2,2)),nn.Flatten())
        self.fuse=nn.Sequential(nn.Linear(768,128),nn.SiLU())
        self.head=build_mlp(135,out,256,2,0.)
        self.out=out
    def forward(self,x):
        a=self.c3(x[:,:self.n3].reshape(-1,16*self.r,4,8,8));b=self.c2(x[:,self.n3:self.n3+self.n2].reshape(-1,32*self.r,8,8))
        z=self.head(torch.cat((self.fuse(torch.cat((a,b),1)),x[:,-7:]),1))
        return z.squeeze(-1) if self.out==1 else z

class Fixture(Dataset):
    def __init__(self,features,labels,n):
        self.x=features.repeat((n+50)//50,1)[:n+1].contiguous();self.labels=labels;self.n=n
    def __len__(self):return self.n
    def __getitem__(self,i):
        t=i%50; nxt=i if t==49 else i+1
        return dict(state=self.x[i],next_state=self.x[nxt],**{k:self.labels[k][t] for k in ('action','reward','done','actor_mask')})

def worker(group):
    torch.set_num_threads(1);torch.manual_seed(42)
    device=torch.device('cuda'); cfg=TrainingConfig()
    data=torch.load(ROOT/'fixture.pt',weights_only=False); feat=data['features'][group];labels=data['labels']
    out=ROOT/group;out.mkdir(exist_ok=True);w=WEIGHTS/group;w.mkdir(exist_ok=True)
    (out/'README.md').write_text('# Timing only\n\nresult.json and progress.jsonl report cached-fixture epochs; weights are not quality-trained.\n')
    (w/'README.md').write_text('# Timing fixture weights\n\nlatest.pt is disposable and not valid for policy evaluation.\n')
    roles=3 if group in ('G1','G3') else 2
    def net(output):
        if group=='MLP':return (ValueNet if output==1 else PolicyNet)(IQLModelConfig(135)).cuda()
        return CNN(roles,output).cuda()
    nets={k:net(1 if k=='value_net' else 2) for k in ('value_net','q1_net','q2_net','target_q1','target_q2','policy_net')}
    for k in ('q1','q2'):nets['target_'+k].load_state_dict(nets[k+'_net'].state_dict())
    opts=tuple(torch.optim.AdamW(p,lr=cfg.lr,weight_decay=cfg.weight_decay) for p in (nets['value_net'].parameters(),list(nets['q1_net'].parameters())+list(nets['q2_net'].parameters()),nets['policy_net'].parameters()))
    tick=time.perf_counter()
    loaders={s:DataLoader(Fixture(feat,labels,n),batch_size=256,shuffle=s=='train',num_workers=0,pin_memory=True) for s,n in [('train',140000),('val',17500),('warm',256*30)]}
    setup=time.perf_counter()-tick
    def run(s,training):
        torch.cuda.synchronize();t=time.perf_counter()
        m=_run_epoch(loader=loaders[s],**nets,args=cfg,device=device,optimizers=opts if training else None)
        torch.cuda.synchronize()
        assert all(torch.isfinite(torch.tensor(v)) for v in m.values()),m
        return time.perf_counter()-t,m
    warm=run('warm',True)[0];torch.cuda.reset_peak_memory_stats(); rows=[]
    for epoch in range(2):
        train,tm=run('train',True);val,vm=run('val',False)
        torch.cuda.synchronize();tick=time.perf_counter()
        payload={k:n.state_dict() for k,n in nets.items()};payload['optimizers']=[o.state_dict() for o in opts]
        torch.save(payload,w/'latest.pt');torch.cuda.synchronize();save=time.perf_counter()-tick
        row=dict(epoch=epoch,train_seconds=train,val_seconds=val,save_seconds=save,total_seconds=train+val+save,train_metrics=tm,val_metrics=vm)
        rows.append(row)
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(group,json.dumps(row),flush=True)
    result=dict(group=group,roles=None if group=='MLP' else roles,rows=rows,epoch_seconds_median=statistics.median(r['total_seconds'] for r in rows),hours400=statistics.median(r['total_seconds'] for r in rows)*400/3600,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,setup_seconds=setup,warmup_seconds=warm,policy_parameters=sum(p.numel() for p in nets['policy_net'].parameters()),torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,training_config=asdict(cfg),scope='FP32 network, FP16 host cache, pinned DataLoader, full epoch counts, one real trajectory repeated: throughput only, no convergence estimate, no feature extraction inside training')
    dump(out/'result.json',result)

if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker']);p.add_argument('--group',default='G1');a=p.parse_args()
    prepare() if a.mode=='prepare' else worker(a.group)
