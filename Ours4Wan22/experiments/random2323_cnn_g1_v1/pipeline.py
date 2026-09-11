"""Build formal G1 cache from existing random2323, then fresh Wan21-aligned IQL."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT))
from ours4wan22.shared import OFFICIAL,MODELS,read,write,sha256,result_dir,implementation_hashes,verify_sources
from ours4wan22.contracts import DIM,FEATURE,PROTOCOL,scalar_state,FORCED
from ours4wan22.features import History
from ours4wan22.model import CNN,ARCH,networks,encode_normalized
from ours4wan22.policy import SCHEMA
from ours4wan21.contracts import training_config
from ours4wan21.local_iql import required_hard_budget_action,_run_epoch

STORE=Path('/all/yiran07-disk3/huteng_data/exp')
DATA=STORE/'wan22_pooled1287_sea_distance_20260905/random2323/dataset'
PACKED=STORE/'wan22_active3815_full_latent_packed8_from_packed4_exactfp16_20260813_215521'
PROOF=STORE/'wan22_random2323_pooled_cnn_t12_h12w20_cache_gpu0_20260905/trajectories.json'
CFG=replace(training_config('aggressive_a1_v1'),epochs=200,num_layers=2)

def csvrows(path):
    import csv
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def save(path,payload):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,tmp);tmp.replace(path)

def metadata(rows):
    ids={r['trajectory_id']:i for i,r in enumerate(rows)}
    assert len(ids)==2323
    scalar=torch.zeros(2323,50,7);actions=torch.full((2323,50),-1,dtype=torch.long)
    distances=torch.zeros(2323,50,2);seen=set()
    import csv
    with (DATA/'data/tables/step_examples.csv').open() as f:
        for r in csv.DictReader(f):
            i,t=ids[r['trajectory_id']],int(r['step_index'])
            assert (i,t) not in seen;seen.add((i,t))
            assert r['model_stage']==('high' if t<32 else 'low')
            for key in ('rel_l1','accumulated_rel_l1','decision'):
                assert r[key]==r['cond_'+key]==r['uncond_'+key]
            assert r['decision'] in ('reuse','recompute')
            actions[i,t]=int(r['decision']=='reuse')
            if t not in FORCED:distances[i,t]=torch.tensor([float(r['rel_l1']),float(r['accumulated_rel_l1'])])
    assert len(seen)==116150
    masks=torch.zeros(2323,50);cache_indices=torch.zeros(2323,50,dtype=torch.long)
    for i,r in enumerate(rows):
        source=json.loads(r['cache_summary_json'])
        skip=set()
        for stage in ('high','low'):
            c=source[str((stage,'cond'))]['skipping_path'];u=source[str((stage,'uncond'))]['skipping_path']
            assert c==u;skip.update(c)
        assert skip==set(torch.nonzero(actions[i]).flatten().tolist())
        k=int(actions[i].sum());used=consecutive=cached=0;accumulator=0.
        for t in range(50):
            if t in (0,32):cached=t;accumulator=0.
            d,total=distances[i,t].tolist()
            if t not in FORCED:assert abs(total-accumulator-d)<2e-6
            scalar[i,t]=scalar_state(t,k,used,consecutive,t not in (0,32),d,total)
            cache_indices[i,t]=cached
            required,_=required_hard_budget_action(step_index=t,used_skips=used,skip_budget=k,num_steps=50,forced_steps=FORCED)
            action=int(actions[i,t]);assert required is None or action==required
            masks[i,t]=float(required is None)
            used+=action;consecutive=consecutive+1 if action else 0
            if not action:cached=t
            accumulator=total if action else 0.
    return dict(scalars=scalar,actions=actions,actor_mask=masks,cache_indices=cache_indices,
        train_indices=torch.tensor([i*50+t for i,r in enumerate(rows) if r['split']=='train' for t in range(50)]),
        val_indices=torch.tensor([i*50+t for i,r in enumerate(rows) if r['split']=='evaluation' for t in range(50)]))

def extract(raw,indices):
    """Pool each input exactly once; gather historical roles after pooling."""
    x=raw.cuda().float()
    direct=F.adaptive_avg_pool3d(x,(4,8,8)).flatten(1)
    stats=torch.cat((F.adaptive_avg_pool2d(x.mean(2),(8,8)),F.adaptive_avg_pool2d(x.var(2,unbiased=False),(8,8))),1).flatten(1)
    previous=torch.arange(50,device=x.device).sub(1).clamp_min(0)
    cached=indices.to(x.device)
    z=torch.cat((direct,direct[previous],direct[cached],stats,stats[previous],stats[cached]),1)
    z[[0,32]]=0
    assert z.shape==(50,18432) and torch.isfinite(z).all()
    return z.cpu()

def build(root):
    rows=read(root/'rows.json');labels=metadata(rows)
    save(root/'labels.pt',labels)
    manifest=read(PACKED/'manifest.json');assert manifest['status']=='complete' and manifest['latent_shape']==[16,12,60,104]
    proofs={r['trajectory_id']:r for r in read(PROOF)}
    sources=csvrows(Path(manifest['source_data_root'])/'data/tables/summary.csv')
    target={}
    for i,r in enumerate(rows):
        s=int(r['subset_source_release_index']);old=sources[s]
        assert all(r[k]==old[k] for k in ('trajectory_id','sample_id','split','latent_dir'))
        target[s]=i
    array=np.lib.format.open_memmap(root/'raw.npy',mode='w+',dtype=np.float32,shape=(116150,DIM))
    array[:,-7:]=labels['scalars'].reshape(-1,7).numpy()
    raw_bytes=50*16*12*60*104*2;completed=0;began=time.perf_counter()
    with (root/'raw_provenance.jsonl').open('w',buffering=1) as output:
        for shard in manifest['shards']:
            path=PACKED/shard['relative_path'];assert path.stat().st_size==shard['bytes']
            provenance=read(PACKED/shard['provenance_path'])
            fd=os.open(path,os.O_RDONLY)
            try:
                for local,global_index in enumerate(shard['trajectory_indices']):
                    if global_index not in target:continue
                    i=target[global_index];row=rows[i];proof=proofs[row['trajectory_id']]
                    assert provenance[local]['latent_dir']==row['latent_dir']
                    assert provenance[local]['global_trajectory_index']==global_index
                    assert shard['split']==row['split']
                    payload=bytearray(os.pread(fd,raw_bytes,local*raw_bytes));assert len(payload)==raw_bytes
                    digest=hashlib.sha256(payload).hexdigest()
                    assert digest==proof['source_payload_sha256']
                    raw=torch.frombuffer(payload,dtype=torch.float16).reshape(50,16,12,60,104)
                    with torch.no_grad():z=extract(raw,labels['cache_indices'][i])
                    if completed==0:
                        trace=read(Path(row['trace_json']));history=History()
                        for t in range(50):
                            serial=history.observe(raw[t].cuda(),t,float(trace['sigmas'][t])).cpu()
                            assert torch.equal(serial,z[t]),f'batched/online feature mismatch at {t}'
                            history.commit(int(labels['actions'][i,t]))
                        for t in (0,32,49):
                            original=torch.load(Path(row['latent_dir'])/f'step_{t:03d}_input.pt',map_location='cpu',weights_only=True)
                            assert torch.equal(original,raw[t])
                        write(root/'FEATURE_PARITY.json',dict(status='passed',trajectory=row['trajectory_id'],steps=50,raw_original_steps=[0,32,49]))
                    array[i*50:(i+1)*50,:-7]=z.numpy()
                    output.write(json.dumps(dict(index=i,trajectory_id=row['trajectory_id'],packed=str(path),local=local,payload_sha256=digest))+'\n')
                    del raw,z,payload
                    completed+=1
                    if completed%20==0 or completed==2323:
                        array.flush();status=dict(phase='feature_cache',completed=completed,total=2323,seconds=time.perf_counter()-began)
                        write(root/'STATUS.json',status);print(json.dumps(status),flush=True)
            finally:os.close(fd)
    assert completed==2323
    array.flush()
    idx=labels['train_indices'].numpy();total=np.zeros(DIM,dtype=np.float64);square=total.copy()
    for start in range(0,len(idx),512):
        z=np.asarray(array[idx[start:start+512]],dtype=np.float64);total+=z.sum(0);square+=(z*z).sum(0)
    mean=total/len(idx);std=np.maximum(np.sqrt(np.maximum(square/len(idx)-mean*mean,0)),1e-6)
    norm=dict(mean=torch.from_numpy(mean.astype('float32')),std=torch.from_numpy(std.astype('float32')))
    save(root/'normalizer.pt',norm)
    output=np.lib.format.open_memmap(root/'features.npy',mode='w+',dtype=np.float16,shape=array.shape)
    for start in range(0,116150,512):
        output[start:start+512]=encode_normalized(torch.from_numpy(np.array(array[start:start+512],copy=True)),norm).numpy()
    output.flush();del output,array
    write(root/'CACHE_COMPLETE.json',dict(status='complete',trajectories=2323,transitions=116150,
        input_dim=DIM,feature_contract=FEATURE,train_rows=len(idx),validation_rows=23600,
        normalizer='Wan21 train-only FP64 sum/sumsquare population statistics; float32 parameters; normalized FP16',
        hashes={f:sha256(root/f) for f in ('features.npy','normalizer.pt','labels.pt','raw.npy')},seconds=time.perf_counter()-began))

def rewards(root):
    sys.path.insert(0,str(OFFICIAL/'VideoMetrics'))
    from video_metrics.video import decode_video_rgb
    from video_metrics.core import psnr_per_frame
    rows=read(root/'rows.json');began=time.perf_counter()
    @lru_cache(maxsize=4)
    def baseline(path):return decode_video_rgb(path)
    def evaluate(item):
        i,r=item;b=baseline(r['baseline_video']);c=decode_video_rgb(r['candidate_video'])
        assert b.shape==c.shape==(45,3,480,832)
        per_frame=psnr_per_frame(b,c)
        return dict(index=i,trajectory_id=r['trajectory_id'],rgb_psnr=float(per_frame.mean()),
            per_frame=per_frame.tolist(),legacy_psnr=float(r['mean_psnr']),
            candidate_sha256=sha256(r['candidate_video']),baseline_sha256=sha256(r['baseline_video']))
    # Group by prompt to reuse baseline decodes; two bounded CPU workers, no BLAS fan-out.
    ordering=sorted(enumerate(rows),key=lambda p:p[1]['sample_id'])
    values={}
    with ThreadPoolExecutor(max_workers=2) as pool,(root/'rgb_rewards.jsonl').open('w',buffering=1) as out:
        for result in pool.map(evaluate,ordering):
            values[result['index']]=result['rgb_psnr'];out.write(json.dumps(result)+'\n')
            if len(values)%20==0 or len(values)==2323:
                write(root/'REWARD_STATUS.json',dict(completed=len(values),total=2323,seconds=time.perf_counter()-began))
    save(root/'terminal_psnr.pt',torch.tensor([values[i] for i in range(2323)],dtype=torch.float32))
    write(root/'REWARD_COMPLETE.json',dict(status='complete',trajectories=2323,protocol='VideoMetrics rgb_full_reference_v1',
        reward='arithmetic mean over 45 aligned RGB frame PSNR values, terminal only, absolute',
        sha256=sha256(root/'terminal_psnr.pt'),records_sha256=sha256(root/'rgb_rewards.jsonl'),seconds=time.perf_counter()-began))

class Cached(Dataset):
    def __init__(self,x,labels,reward,split):self.x,self.labels,self.reward,self.idx=x,labels,reward,labels[split+'_indices']
    def __len__(self):return len(self.idx)
    def __getitem__(self,item):
        j=int(self.idx[item]);i,t=divmod(j,50)
        return dict(state=self.x[j],next_state=self.x[j if t==49 else j+1],action=self.labels['actions'][i,t],
            reward=self.reward[i] if t==49 else torch.tensor(0.),done=torch.tensor(float(t==49)),actor_mask=self.labels['actor_mask'][i,t])

def train(root,weights):
    cache=read(root/'CACHE_COMPLETE.json');reward_manifest=read(root/'REWARD_COMPLETE.json')
    for f,h in cache['hashes'].items():assert sha256(root/f)==h
    assert sha256(root/'terminal_psnr.pt')==reward_manifest['sha256']
    labels=torch.load(root/'labels.pt',weights_only=True);reward=torch.load(root/'terminal_psnr.pt',weights_only=True)
    norm=torch.load(root/'normalizer.pt',weights_only=True)
    x=torch.from_numpy(np.load(root/'features.npy'))
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    nets=networks('cuda');opts=tuple(torch.optim.AdamW(p,lr=CFG.lr,weight_decay=CFG.weight_decay) for p in
        (nets['value_net'].parameters(),list(nets['q1_net'].parameters())+list(nets['q2_net'].parameters()),nets['policy_net'].parameters()))
    loaders={s:DataLoader(Cached(x,labels,reward,s),batch_size=256,shuffle=s=='train',num_workers=0,pin_memory=True) for s in ('train','val')}
    weights.mkdir(parents=True,exist_ok=False);(weights/'README.md').write_text('# Wan22 CNN+G1 weights\n\n200 epoch checkpoints, independent IQL networks, optimizer and RNG, train normalizer.\n')
    (root/'model_weights').symlink_to(weights,target_is_directory=True)
    base=dict(schema=SCHEMA,architecture=ARCH,feature_contract=FEATURE,protocol=PROTOCOL,group='G1',smoke_only=False,
        normalizer=norm,train_config=asdict(CFG),cache_manifest=cache,reward_manifest=reward_manifest,
        implementation_sha256=read(root/'config.json')['implementation_sha256'])
    write(root/'TRAINING_STARTED.json',dict(status='running',pid=os.getpid(),settings=asdict(CFG),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),weights=str(weights),train_transitions=92550,val_transitions=23600))
    began=time.perf_counter()
    for epoch in range(1,201):
        row=dict(epoch=epoch)
        for split in ('train','val'):
            row[split]=_run_epoch(loader=loaders[split],**nets,args=CFG,device=torch.device('cuda'),optimizers=opts if split=='train' else None)
        assert all(math.isfinite(v) for s in ('train','val') for v in row[s].values())
        torch.cuda.synchronize();row['elapsed_seconds']=time.perf_counter()-began
        save(weights/f'epoch_{epoch:03d}.pt',dict(**base,epoch=epoch,metrics=row,**{k:n.state_dict() for k,n in nets.items()},
            optimizer_states=[o.state_dict() for o in opts],torch_rng_state=torch.get_rng_state(),cuda_rng_states=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state()))
        with (root/'epoch_metrics.jsonl').open('a') as out:out.write(json.dumps(row,allow_nan=False)+'\n')
        write(root/'STATUS.json',dict(phase='training',epoch=epoch,total_epochs=200,elapsed_seconds=row['elapsed_seconds'],metrics=row))
        print(json.dumps(dict(epoch=epoch,seconds=row['elapsed_seconds'],val_pi=row['val']['pi_loss'])),flush=True)
    write(root/'TRAINING_COMPLETE.json',dict(status='complete',epochs=200,seconds=time.perf_counter()-began))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','rewards']);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    assert Path(sys.prefix).name=='wan2.2';torch.set_num_threads(1)
    if a.action=='rewards':return rewards(a.root)
    verify_sources()
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    root=result_dir(a.root,'# Random2323 formal CNN+G1\n\nExisting pure random trajectories only; raw/features cache, train normalizer, RGB reward recomputation and 200epoch Wan21-aligned IQL. No data collection. config.json locks inputs; STATUS.json tracks phase; model_weights links to models/.')
    rows=csvrows(DATA/'data/tables/summary.csv')
    assert len(rows)==2323 and Counter(r['split'] for r in rows)=={'train':1851,'evaluation':472}
    assert Counter(r['policy_family'] for r in rows)=={'random_continuous_threshold':1824,'random_continuous_threshold_high_speed_high_q':499}
    prompts={s:{r['sample_id'] for r in rows if r['split']==s} for s in ('train','evaluation')}
    assert len(prompts['train'])==80 and len(prompts['evaluation'])==20 and not prompts['train']&prompts['evaluation']
    write(root/'rows.json',rows)
    weights=MODELS/root.name
    write(root/'config.json',dict(settings=asdict(CFG),feature_contract=FEATURE,architecture=ARCH,protocol=PROTOCOL,
        dataset=str(DATA),packed=str(PACKED),weights=str(weights),trajectories=2323,split_trajectories=dict(train=1851,val=472),
        split_prompts=dict(train=80,val=20),split_policy='preserve frozen Wan22 random2323 manifest; evaluation used as validation, no new test split',
        source_sha256={str(p):sha256(p) for p in (DATA/'data/tables/summary.csv',DATA/'data/tables/step_examples.csv',PACKED/'manifest.json',PROOF)},
        implementation_sha256=implementation_hashes(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        thread_environment={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}))
    with (root/'reward_worker.log').open('w') as log:
        worker=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'rewards','--root',str(root)],stdout=log,stderr=subprocess.STDOUT)
        try:
            build(root)
            if worker.wait()!=0:raise RuntimeError('RGB reward worker failed; see reward_worker.log')
            train(root,weights)
        except BaseException as exc:
            if worker.poll() is None:worker.terminate();worker.wait()
            write(root/'FAILED.json',dict(error=repr(exc)));raise

if __name__=='__main__':main()
