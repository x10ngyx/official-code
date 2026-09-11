import argparse, fcntl, hashlib, io, json, math, os, random, subprocess, sys, time, traceback
from pathlib import Path
from dataclasses import asdict,replace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT))
from model import CNN,ARCH,GROUPS,dimension,encode_normalized
from features import History,CONTRACT
from ours4wan21.contracts import TrainingConfig,PROTOCOL,sha256,training_config
from ours4wan21.local_iql import _run_epoch
from ours4wan21.train import validate_bundle
from ours4wan21.analysis import rank_checkpoints,metrics_report,write_csv
ROOT=Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_cnn_mixed3500_v1')
WEIGHTS=Path('/mnt/hdd/xiongyuxiang/tmp/models/ours21_cnn_mixed3500_v1')
SOURCE=Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_increase500_iql2_v1/ours21_increase500_iql2_v1_cache/transitions.pt')
CFG=replace(training_config('aggressive_a1_v1'),epochs=200,num_layers=2)
SELECTION_START=170

def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(tmp,p)
def save(p,x):
    tmp=p.with_name(p.name+'.tmp');torch.save(x,tmp);os.replace(tmp,p)
def directory(p,desc):
    p.mkdir(parents=True,exist_ok=True)
    if not (p/'README.md').exists():(p/'README.md').write_text(desc+'\n')
    return p
def sources():return [*sorted(HERE.glob('*.py')),PROJECT/'ours4wan21/local_iql.py',PROJECT/'ours4wan21/contracts.py',PROJECT/'ours4wan21/latent_features.py',PROJECT/'ours4wan21/train.py',PROJECT/'ours4wan21/analysis.py']
def verify_code():
    cfg=read(ROOT/'config.json')
    for p,h in cfg['code_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('frozen code changed: '+p)
    return cfg
def initialize():
    if (ROOT/'INITIALIZED.json').exists():verify_code();return
    directory(ROOT,(HERE/'README.md').read_text())
    for name in ('raw_rows','row_records','cache','training','analysis','logs','jobs'):directory(ROOT/name,'# '+name+'\n\nSee parent experiment README.')
    directory(WEIGHTS,'# Four production CNN IQL groups\n\nG1–G4 each store 200 epoch checkpoints and validation-selected links.')
    h=sha256(SOURCE)
    if h!=read(SOURCE.parent/'COMPLETE.json')['sha256']:raise ValueError('source cache hash')
    b=torch.load(SOURCE,map_location='cpu',weights_only=False)
    validate_bundle(b,b['manifest']['state']['mode'])
    sel=b['manifest']['selection']
    assert sel['selected_count']==3500 and sel['split_counts']=={'train':2800,'val':350,'test':350}
    assert sel['family_counts']=={'random_continuous_seacache_threshold':3000,'linear_increase_seacache_threshold':500}
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines();assert len(uuids)==4
    cfg=dict(schema='ours21_cnn_mixed3500_suite_v1',source=str(SOURCE),source_sha256=h,groups=GROUPS,feature_contract=CONTRACT,architecture=ARCH,training=asdict(CFG),iql_profile='aggressive_a1_v1',protocol=PROTOCOL,gpu_uuids=uuids,code_sha256={str(p):sha256(p) for p in sources()},epochs=CFG.epochs,video_evaluation=False,selection='two-sided stability: census e170-e200, candidates e171-e199',selection_start=SELECTION_START,training_parameters_confirmation='pending',seed=42)
    if (ROOT/'config.json').exists():assert read(ROOT/'config.json')==cfg
    else:dump(ROOT/'config.json',cfg)
    records=[];texts={}
    for i,s in enumerate(b['manifest']['sources']):
        for p,expected in s['files'].items():
            if sha256(Path(p))!=expected:raise ValueError('source metadata hash '+p)
        tp=next(Path(p) for p in s['files'] if p.endswith('/trace.json'));tr=read(tp)
        assert len(tr['step_records'])==50 and tr['trajectory_id']==s['trajectory_id']
        recs=tr['step_records'];assert [r['step_index'] for r in recs]==list(range(50))
        for t,r in enumerate(recs):
            assert r['cond_action']==r['uncond_action']==r['action']
            assert int(r['action']=='reuse')==int(b['tensors']['action'][i*50+t])
            assert Path(r['latent_path']).is_file() and r['latent_shape']==[16,21,60,104]
        prompt=tr.get('manifest_record',{}).get('prompt')
        if isinstance(prompt,str):
            norm=' '.join(prompt.split());assert texts.get(norm,s['split'])==s['split'];texts[norm]=s['split']
        records.append(dict(index=i,source=s,trace=str(tp),trace_sha256=sha256(tp)))
    save(ROOT/'labels.pt',dict(tensors={k:(v[:,-7:].clone() if k in ('state','next_state') else v) for k,v in b['tensors'].items()},train_indices=b['train_indices'],val_indices=b['val_indices'],test_indices=b['test_indices']))
    dump(ROOT/'dataset_manifest.json',b['manifest']);dump(ROOT/'records.json',records)
    dump(ROOT/'INITIALIZED.json',dict(status='complete',trajectories=3500,transitions=175000,source_sha256=h,labels_sha256=sha256(ROOT/'labels.pt'),records_sha256=sha256(ROOT/'records.json'),normalized_prompt_texts_checked=len(texts)))
    link=PROJECT/'experiment_results'/ROOT.name
    if not link.exists():link.symlink_to(ROOT)
    print('Initialized and validated mixed3500',flush=True)

def extract(shard):
    verify_code();rows=read(ROOT/'records.json');labels=torch.load(ROOT/'labels.pt',weights_only=False)['tensors'];began=time.time();done=0
    for row in rows[shard::4]:
        i=row['index'];p=ROOT/'raw_rows'/f'{i:04d}.pt';meta=ROOT/'row_records'/f'{i:04d}.json'
        if meta.exists():
            m=read(meta)
            if m['trace_sha256']!=row['trace_sha256'] or sha256(p)!=m['sha256']:raise ValueError('resume row hash')
            done+=1;continue
        assert sha256(Path(row['trace']))==row['trace_sha256'];tr=read(row['trace']);history=History();parts={g:[] for g in GROUPS};input_hashes=[]
        for t,rec in enumerate(tr['step_records']):
            path=Path(rec['latent_path']);raw=path.read_bytes();input_hashes.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
            x=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            if x.dtype!=torch.float16:raise ValueError('archive dtype')
            out=history.observe(x.cuda(),t,float(rec['sigma']))
            for g in GROUPS:parts[g].append(torch.cat((out[g].cpu(),labels['state'][i*50+t])).float())
            history.commit(int(labels['action'][i*50+t]))
        arrays={g:torch.stack(v) for g,v in parts.items()}
        for g,z in arrays.items():assert z.shape==(50,dimension(g)) and torch.isfinite(z).all() and not z[0,:-7].any()
        save(p,arrays);dump(meta,dict(index=i,trajectory_id=row['source']['trajectory_id'],trace_sha256=row['trace_sha256'],sha256=sha256(p),latent_inputs=input_hashes))
        done+=1
        if done%10==0:dump(ROOT/'jobs'/f'extract_{shard}.json',dict(completed=done,total=len(rows[shard::4]),elapsed_seconds=time.time()-began))
    dump(ROOT/'jobs'/f'extract_{shard}.json',dict(status='complete',completed=done,total=len(rows[shard::4]),elapsed_seconds=time.time()-began))

def build_cache(g):
    verify_code();out=directory(ROOT/'cache'/g,'# Normalized FP16 feature cache\n\nfeatures.npy is row-major full dataset; normalizer.pt uses train only; COMPLETE.json hashes all files.')
    complete=out/'COMPLETE.json'
    if complete.exists():
        for p,h in read(complete)['hashes'].items():assert sha256(Path(p))==h
        return
    rows=read(ROOT/'records.json');rawpath=out/'raw.npy';raw=np.lib.format.open_memmap(rawpath,mode='w+',dtype=np.float32,shape=(175000,dimension(g)))
    for row in rows:
        i=row['index'];p=ROOT/'raw_rows'/f'{i:04d}.pt';m=read(ROOT/'row_records'/f'{i:04d}.json');assert sha256(p)==m['sha256']
        z=torch.load(p,map_location='cpu',weights_only=True)[g];assert torch.isfinite(z).all();raw[i*50:(i+1)*50]=z.numpy()
    raw.flush();labels=torch.load(ROOT/'labels.pt',weights_only=False);idx=labels['train_indices'].numpy()
    total=np.zeros(dimension(g),dtype=np.float64);square=total.copy()
    for offset in range(0,len(idx),512):
        z=np.asarray(raw[idx[offset:offset+512]],dtype=np.float64);total+=z.sum(0);square+=(z*z).sum(0)
    mean=total/len(idx);std=np.maximum(np.sqrt(np.maximum(square/len(idx)-mean*mean,0)),1e-6)
    norm={'mean':torch.from_numpy(mean.astype('float32')),'std':torch.from_numpy(std.astype('float32'))};save(out/'normalizer.pt',norm)
    target=out/'features.npy';a=np.lib.format.open_memmap(target.with_suffix('.tmp.npy'),mode='w+',dtype=np.float16,shape=raw.shape)
    max_abs=0.
    for i in range(0,175000,512):
        z=encode_normalized(torch.from_numpy(np.array(raw[i:i+512],copy=True)),norm);a[i:i+512]=z.numpy();max_abs=max(max_abs,float(z.abs().max()))
    a.flush();del a;os.replace(target.with_suffix('.tmp.npy'),target)
    dump(complete,dict(status='complete',group=g,rows=175000,dim=dimension(g),dtype='float16',train_normalizer_rows=len(idx),max_abs_normalized=max_abs,contract=CONTRACT,hashes={str(p):sha256(p) for p in (target,out/'normalizer.pt')}))
    del raw;rawpath.unlink() # redundant intermediate, immutable per-trajectory raw files retained

class CachedDataset(Dataset):
    def __init__(self,x,labels,indices):self.x=x;self.labels=labels;self.indices=indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        j=int(self.indices[i]);nxt=j if j%50==49 else j+1
        return dict(state=self.x[j],next_state=self.x[nxt],**{k:self.labels[k][j] for k in ('action','reward','done','actor_mask')})

def train(g):
    wait_for_training_confirmation(g)
    cfg=verify_code();build_cache(g);out=directory(ROOT/'training'/g,'# Production training\n\nepoch_metrics.jsonl and STATUS.json record training; weights link to unified models.');w=directory(WEIGHTS/g,'# Production group checkpoints\n\ncheckpoints/ contains all epochs; selected.pt uses validation stability, final_model.pt is epoch200.')
    directory(w/'checkpoints','# Epoch checkpoints with all networks, optimizer, RNG, normalizer and feature contract.')
    if not (out/'model_weights').exists():(out/'model_weights').symlink_to(w)
    if (out/'TRAINING_COMPLETE.json').exists():return
    cache=ROOT/'cache'/g;payload=torch.load(ROOT/'labels.pt',weights_only=False);x=torch.from_numpy(np.load(cache/'features.npy',allow_pickle=False));norm=torch.load(cache/'normalizer.pt',weights_only=True)
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    nets={k:CNN(g,1 if k=='value_net' else 2).cuda() for k in ('value_net','q1_net','q2_net','target_q1','target_q2','policy_net')}
    for k in ('q1','q2'):nets['target_'+k].load_state_dict(nets[k+'_net'].state_dict())
    opts=tuple(torch.optim.AdamW(p,lr=CFG.lr,weight_decay=CFG.weight_decay) for p in (nets['value_net'].parameters(),list(nets['q1_net'].parameters())+list(nets['q2_net'].parameters()),nets['policy_net'].parameters()))
    loaders={s:DataLoader(CachedDataset(x,payload['tensors'],payload[s+'_indices']),batch_size=256,shuffle=s=='train',num_workers=0,pin_memory=True) for s in ('train','val')}
    base=dict(schema='ours21_three_branch_cnn_checkpoint_v1',group=g,architecture=ARCH,feature_contract=CONTRACT,train_config=asdict(CFG),protocol=PROTOCOL,normalizer=norm,cache_manifest=read(cache/'COMPLETE.json'),source_sha256=cfg['source_sha256'],config_sha256=sha256(ROOT/'config.json'),iql_profile=cfg['iql_profile'],smoke_only=False)
    dump(out/'config.json',dict(group=g,architecture=ARCH,feature_contract=CONTRACT,training=asdict(CFG),iql_profile=cfg['iql_profile'],cache_manifest=base['cache_manifest'],source_sha256=base['source_sha256']))
    first=1;elapsed=0.;rows=[]
    if (out/'epoch_metrics.jsonl').exists():rows=[json.loads(s) for s in (out/'epoch_metrics.jsonl').read_text().splitlines()]
    paths=sorted((w/'checkpoints').glob('epoch_*.pt'))
    if paths:
        old=torch.load(paths[-1],map_location='cpu',weights_only=False)
        assert old['config_sha256']==base['config_sha256'] and old['cache_manifest']==base['cache_manifest'] and old['architecture']==ARCH
        for k,n in nets.items():n.load_state_dict(old[k])
        for o,state in zip(opts,old['optimizer_states']):o.load_state_dict(state)
        torch.set_rng_state(old['torch_rng_state']);torch.cuda.set_rng_state_all(old['cuda_rng_states']);random.setstate(old['python_rng']);np.random.set_state(old['numpy_rng'])
        first=old['epoch']+1;elapsed=old['metrics']['elapsed_seconds'];rows=[r for r in rows if r['epoch']<old['epoch']]+[old['metrics']]
        (out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    start=time.perf_counter();best=min((r['val']['pi_loss'] for r in rows),default=float('inf'))
    for epoch in range(first,CFG.epochs+1):
        row={'epoch':epoch}
        for s in ('train','val'):
            row[s]=_run_epoch(loader=loaders[s],**nets,args=CFG,device=torch.device('cuda'),optimizers=opts if s=='train' else None)
        if not all(math.isfinite(v) for s in ('train','val') for v in row[s].values()):raise ValueError('nonfinite loss')
        torch.cuda.synchronize();row['elapsed_seconds']=elapsed+time.perf_counter()-start
        p=dict(base,**{k:n.state_dict() for k,n in nets.items()},epoch=epoch,metrics=row,optimizer_states=[o.state_dict() for o in opts],torch_rng_state=torch.get_rng_state(),cuda_rng_states=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state())
        dest=w/'checkpoints'/f'epoch_{epoch:03d}.pt';save(dest,p)
        with (out/'epoch_metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if row['val']['pi_loss']<best:
            best=row['val']['pi_loss'];link=w/'best_model.pt'
            if link.is_symlink():link.unlink()
            link.symlink_to(dest.relative_to(w))
        dump(out/'STATUS.json',dict(stage='training',epoch=epoch,epochs=CFG.epochs,elapsed_seconds=row['elapsed_seconds'],latest_checkpoint=str(dest)))
        print(g,json.dumps(row),flush=True)
    final=w/'final_model.pt'
    if final.is_symlink():final.unlink()
    final.symlink_to(f'checkpoints/epoch_{CFG.epochs:03d}.pt')
    dump(out/'TRAINING_COMPLETE.json',dict(status='complete',epochs=CFG.epochs,final_sha256=sha256(final),config_sha256=base['config_sha256']))

def select(g):
    verify_code();out=directory(ROOT/'analysis'/g,'# Validation-only post170 checkpoint selection and training readout.')
    if (out/'COMPLETE.json').exists():return
    rows=[json.loads(s) for s in (ROOT/'training'/g/'epoch_metrics.jsonl').read_text().splitlines()];assert [r['epoch'] for r in rows]==list(range(1,CFG.epochs+1))
    labels=torch.load(ROOT/'labels.pt',weights_only=False);idx=labels['val_indices'];idx=idx[labels['tensors']['actor_mask'][idx]>0]
    a=np.load(ROOT/'cache'/g/'features.npy',mmap_mode='r');x=torch.from_numpy(np.array(a[idx.numpy()],copy=True));nets=[CNN(g,2).cuda() for _ in range(3)]
    epochs=list(range(SELECTION_START,CFG.epochs+1));q=[];actions=[];hashes={}
    with torch.no_grad():
        for e in epochs:
            path=WEIGHTS/g/'checkpoints'/f'epoch_{e:03d}.pt';p=torch.load(path,map_location='cpu',weights_only=False);assert p['group']==g and p['architecture']==ARCH
            for n,k in zip(nets,('q1_net','q2_net','policy_net')):n.load_state_dict(p[k]);n.eval()
            qs=[];acts=[]
            for start in range(0,len(x),256):
                z=x[start:start+256].cuda().float();qs.append(torch.minimum(nets[0](z),nets[1](z)).cpu().numpy());acts.append(nets[2](z).argmax(-1).cpu().numpy())
            q.append(np.concatenate(qs));actions.append(np.concatenate(acts));hashes[str(path)]=sha256(path)
    values=np.stack(q);acts=np.stack(actions);adj,ranked,selection=rank_checkpoints(epochs,values,acts,{r['epoch']:r for r in rows})
    np.savez_compressed(out/'post170_checkpoint_census.npz',epochs=epochs,row_index=idx.numpy(),qmin=values,actor=acts)
    e=selection['selected']['epoch'];dest=WEIGHTS/g/'checkpoints'/f'epoch_{e:03d}.pt';link=WEIGHTS/g/'selected.pt'
    if link.is_symlink():link.unlink()
    link.symlink_to(dest.relative_to(WEIGHTS/g));dump(out/'checkpoint_selection.json',dict(selection,checkpoint=str(dest),sha256=sha256(dest),source_hashes=hashes))
    write_csv(out/'post170_adjacent_census.csv',adj);write_csv(out/'post170_ranked.csv',ranked);metrics_report(out,rows)
    dump(out/'COMPLETE.json',dict(status='complete',selected_epoch=e,checkpoint=str(dest),sha256=sha256(dest)))

def training_signature():
    cfg=read(ROOT/'config.json')
    contract={k:cfg[k] for k in ('training','architecture','selection_start','epochs')}
    return hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()

def wait_for_training_confirmation(g):
    # User explicitly requested parameter confirmation. Cache work remains authorized.
    marker=ROOT/'TRAINING_PARAMETERS_CONFIRMED.json'
    status=directory(ROOT/'training'/g,'# Training stage status and epoch metrics.')/'STATUS.json'
    while not marker.exists():
        dump(status,dict(stage='waiting_for_iql_parameter_confirmation',group=g))
        time.sleep(10)
    if read(marker).get('training_signature')!=training_signature():
        raise ValueError('confirmation does not match current training/selection contract')

def stageworker(g):
    global CFG
    build_cache(g)
    wait_for_training_confirmation(g)
    CFG=TrainingConfig(**verify_code()['training'])
    train(g)
    select(g)
def environment(gpu):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=read(ROOT/'config.json')['gpu_uuids'][gpu],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');return e
def command(mode,index):
    label=f'{mode}_{index}';args=[sys.executable,'-u',str(HERE/'pipeline.py'),mode,'--index',str(index)]
    dump(ROOT/'jobs'/f'{label}_command.json',dict(command=args,gpu=index,pid=os.getpid()))
    with (ROOT/'logs'/f'{label}.log').open('a') as f:subprocess.run(args,env=environment(index),stdout=f,stderr=subprocess.STDOUT,check=True)
def run():
    verify_code()
    with (ROOT/'pipeline.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'COMPLETE.json').exists():return
        with (ROOT.parent/'wan21_benchmark_4gpu.lock').open('a') as lock:
            dump(ROOT/'STATUS.json',dict(stage='waiting_for_gpu_lock',pid=os.getpid()));fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                for stage in ('extract','train_group'):
                    dump(ROOT/'STATUS.json',dict(stage=stage,pid=os.getpid(),started_at=time.time()))
                    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda i:command(stage,i),range(4)))
                summary={g:read(ROOT/'analysis'/g/'COMPLETE.json') for g in GROUPS}
                dump(ROOT/'COMPLETE.json',dict(status='complete',groups=summary,epochs_per_group=CFG.epochs,video_evaluation=False));dump(ROOT/'STATUS.json',dict(stage='complete'))
            except BaseException as e:
                dump(ROOT/'FAILED.json',dict(error=repr(e),traceback=traceback.format_exc()));dump(ROOT/'STATUS.json',dict(stage='failed',error=repr(e)));raise

if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['initialize','run','extract','train_group']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.mode=='initialize':initialize()
    elif a.mode=='run':run()
    elif a.mode=='extract':extract(a.index)
    else:stageworker(list(GROUPS)[a.index])
