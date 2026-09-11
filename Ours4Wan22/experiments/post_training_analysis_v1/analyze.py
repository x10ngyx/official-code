"""Automatic full training readout and inclusive e180--200 selection."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from ours4wan22.shared import read,write,sha256,implementation_hashes
from ours4wan22.model import CNN,ARCH
from ours4wan22.contracts import FEATURE,PROTOCOL
from ours4wan22.policy import SCHEMA,Policy
from ours4wan21.analysis import midpoint_iqr

def selection(epochs,q,actions):
    """Wan21 stability criterion with explicit one-sided interval endpoints."""
    if epochs!=list(range(180,201)) or q.shape[:2]!=actions.shape or q.shape[0]!=21 or q.shape[2]!=2:
        raise ValueError('complete e180--200 paired census required')
    if not np.isfinite(q).all() or q.shape[1]==0:raise ValueError('invalid census')
    scales=[midpoint_iqr(z) for z in q]
    adjacent=[]
    for i in range(1,len(epochs)):
        dq=float(np.abs(q[i].astype(float)-q[i-1]).mean())
        adjacent.append(dict(from_epoch=epochs[i-1],to_epoch=epochs[i],
            actor_agreement=float((actions[i]==actions[i-1]).mean()),dq=dq,
            dq_over_q_iqr=dq/max((scales[i]+scales[i-1])/2,1e-12)))
    ranked=[]
    for e in epochs:
        neighbors=[r for r in adjacent if e in (r['from_epoch'],r['to_epoch'])]
        ranked.append(dict(epoch=e,neighbor_count=len(neighbors),
            min_actor=min(r['actor_agreement'] for r in neighbors),
            mean_actor=float(np.mean([r['actor_agreement'] for r in neighbors])),
            mean_dq=float(np.mean([r['dq'] for r in neighbors])),
            mean_normalized_dq=float(np.mean([r['dq_over_q_iqr'] for r in neighbors]))))
    gate=.96 if any(r['min_actor']>=.96 for r in ranked) else (.95 if any(r['min_actor']>=.95 for r in ranked) else max(r['min_actor'] for r in ranked))
    eligible=sorted((r for r in ranked if r['min_actor']>=gate),key=lambda r:(r['mean_normalized_dq'],r['mean_dq'],-r['min_actor'],-r['epoch']))
    return dict(candidate_epochs=epochs,gate=gate,selected=eligible[0],eligible_sorted=eligible,
        ranked=ranked,adjacent=adjacent,
        criterion='Wan21 actor agreement gate .96, fallback .95, fallback maximum; minimum mean D_Q/IQR, then raw D_Q, higher minimum actor agreement, later epoch',
        endpoint_rule='180 uses 180->181; 200 uses 199->200; interior candidates use both adjacent pairs; all 21 candidates included')

def csvwrite(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def plot(out,logs,stats,selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs=[r['epoch'] for r in logs]
    fig,axes=plt.subplots(3,3,figsize=(16,12),layout='constrained')
    for ax,key,title in zip(axes[0],('q_loss','v_loss','pi_loss'),('Q loss','V loss','Actor loss')):
        for split in ('train','val'):ax.plot(epochs,[r[split][key] for r in logs],label=split,lw=1.3)
        ax.set_title(title);ax.legend()
    for ax,key,title in zip(axes[1:].flat,
            ('mean_Q','Q_IQR','mean_V','Q_skip_minus_recompute','D_Q_over_IQR','actor_agreement_previous'),
            ('Mean Q = min(Q1,Q2)','Q-IQR (midpoint P75 - P25)','Mean V','Mean Q(skip) - Q(recompute)','Adjacent D_Q / Q-IQR','Actor agreement with previous epoch')):
        ax.plot(epochs,[r[key] for r in stats],lw=1.4,color='#246f9e');ax.set_title(title)
        if key=='mean_Q':ax.fill_between(epochs,[r['Q_p25'] for r in stats],[r['Q_p75'] for r in stats],alpha=.18,color='#246f9e',label='Q P25--P75');ax.legend()
    for ax in axes.flat:
        ax.axvline(selected,color='#bc7134',ls='--',lw=1,label=f'e{selected}')
        ax.axvspan(180,200,color='#bc7134',alpha=.06);ax.set_xlabel('Epoch');ax.grid(alpha=.15)
    fig.suptitle(f'Wan22 CNN+G1 | random2323 | selected e{selected}\nQ/V census: same validation free-decision states, both actions; no video quality used for selection')
    for suffix in ('png','svg','pdf'):fig.savefig(out/f'training_analysis.{suffix}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,key,title in zip(axes.flat,('mean_Q','Q_IQR','D_Q_over_IQR','actor_agreement_previous'),('Mean Q','Q-IQR','Adjacent D_Q / IQR','Actor agreement')):
        ax.plot(epochs[-21:],[r[key] for r in stats[-21:]],marker='.',lw=1.3)
        ax.axvline(selected,color='#bc7134',ls='--');ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.15)
    fig.suptitle(f'Checkpoint window e180--e200 | selected e{selected} | endpoints use one adjacent pair')
    for suffix in ('png','svg','pdf'):fig.savefig(out/f'selection_window.{suffix}',dpi=160)
    plt.close(fig)

def run(root,out,device):
    completion=read(root/'TRAINING_COMPLETE.json')
    if completion.get('status')!='complete' or completion['epochs']!=200:raise ValueError('training not complete')
    logs=[json.loads(l) for l in (root/'epoch_metrics.jsonl').read_text().splitlines()]
    if [r['epoch'] for r in logs]!=list(range(1,201)):raise ValueError('incomplete epoch logs')
    cfg=read(root/'config.json');weights=Path(cfg['weights'])
    labels=torch.load(root/'labels.pt',map_location='cpu',weights_only=True)
    idx=labels['val_indices'];idx=idx[labels['actor_mask'].flatten()[idx]>0]
    if not len(idx):raise ValueError('no free validation states')
    x=np.load(root/'features.npy',mmap_mode='r')
    data=torch.from_numpy(np.array(x[idx.numpy()],copy=True))
    out.mkdir(exist_ok=False)
    (out/'README.md').write_text('# Training analysis\n\ntraining_analysis and selection_window are PNG/SVG/PDF plots. q_statistics.csv and training_metrics.csv are chart data. census.npz preserves all 200 checkpoint Q/V/actor outputs on identical validation free-decision rows. checkpoint_selection.json defines inclusive e180--200 ranking. COMPLETE.json is written after all outputs and selected actor verification.\n')
    write(out/'config.json',dict(source=str(root),candidate_epochs=list(range(180,201)),census_epochs=list(range(1,201)),
        state_count=len(idx),Q_definition='min(online Q1, online Q2), both actions flattened for mean and midpoint IQR',
        state_population='fixed full validation subset with actor_mask > 0',
        source_hashes={f:sha256(root/f) for f in ('config.json','labels.pt','features.npy','epoch_metrics.jsonl')},
        analysis_code_sha256=sha256(Path(__file__))))
    nets={k:CNN(1 if k=='value_net' else 2).to(device).eval() for k in ('q1_net','q2_net','policy_net','value_net')}
    qs=[];acts=[];vs=[];stats=[];hashes={};began=time.perf_counter()
    with torch.no_grad():
        for epoch in range(1,201):
            path=weights/f'epoch_{epoch:03d}.pt';p=torch.load(path,map_location='cpu',weights_only=False)
            if (p['schema']!=SCHEMA or p['architecture']!=ARCH or p['feature_contract']!=FEATURE or p['protocol']!=PROTOCOL or p['epoch']!=epoch):raise ValueError('checkpoint contract mismatch')
            for k,n in nets.items():n.load_state_dict(p[k]);n.eval()
            q=[];a=[];v=[]
            for start in range(0,len(data),256):
                z=data[start:start+256].to(device).float()
                q.append(torch.minimum(nets['q1_net'](z),nets['q2_net'](z)).cpu().numpy())
                a.append(nets['policy_net'](z).argmax(-1).cpu().numpy());v.append(nets['value_net'](z).cpu().numpy())
            q=np.concatenate(q);a=np.concatenate(a);v=np.concatenate(v)
            if not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('nonfinite census')
            iqr=midpoint_iqr(q);ordered=np.sort(q.astype(np.float64).flatten());centers=(np.arange(len(ordered))+.5)/len(ordered)
            p25,p75=np.interp([.25,.75],centers,ordered)
            dq=float(np.abs(q.astype(float)-qs[-1]).mean()) if qs else None
            stats.append(dict(epoch=epoch,states=len(data),mean_Q=float(q.mean(dtype=np.float64)),Q_p25=float(p25),Q_p75=float(p75),Q_IQR=iqr,
                mean_V=float(v.mean(dtype=np.float64)),Q_skip_minus_recompute=float((q[:,1]-q[:,0]).mean(dtype=np.float64)),
                D_Q_previous=dq,D_Q_over_IQR=dq/max((iqr+stats[-1]['Q_IQR'])/2,1e-12) if qs else None,
                actor_agreement_previous=float((a==acts[-1]).mean()) if qs else None))
            qs.append(q);acts.append(a);vs.append(v);hashes[str(path)]=sha256(path)
            write(out/'STATUS.json',dict(phase='checkpoint_census',epoch=epoch,total=200,seconds=time.perf_counter()-began))
    q=np.stack(qs);a=np.stack(acts);v=np.stack(vs)
    chosen=selection(list(range(180,201)),q[179:],a[179:]);epoch=chosen['selected']['epoch'];checkpoint=weights/f'epoch_{epoch:03d}.pt'
    chosen.update(checkpoint=str(checkpoint),sha256=sha256(checkpoint),source_hashes=hashes,state_count=len(idx))
    Policy(checkpoint,device='cpu').choose(torch.zeros(18439))
    np.savez_compressed(out/'census.npz',epoch=np.arange(1,201),row_indices=idx.numpy(),qmin=q,actor=a,value=v)
    csvwrite(out/'q_statistics.csv',stats)
    flat=[dict(epoch=r['epoch'],elapsed_seconds=r['elapsed_seconds'],**{f'{s}_{k}':v for s in ('train','val') for k,v in r[s].items()}) for r in logs]
    csvwrite(out/'training_metrics.csv',flat);csvwrite(out/'selection_candidates.csv',chosen['ranked']);csvwrite(out/'selection_adjacent.csv',chosen['adjacent'])
    write(out/'checkpoint_selection.json',chosen)
    plot(out,logs,stats,epoch)
    link=weights/'selected.pt'
    if link.exists() or link.is_symlink():raise FileExistsError('refuse to replace existing selected checkpoint')
    link.symlink_to(checkpoint.name)
    report=f'# Training readout\n\nSelected checkpoint: **e{epoch}**, from inclusive e180--e200.\n\nFixed census: {len(idx)} validation free-decision states × 2 actions. Q=min(Q1,Q2), midpoint Q-IQR=P75−P25. Full 200-epoch loss/Q/IQR/V/decision stability plots accompany CSV/NPZ data.\n\nSelection uses actor stability gate and minimum normalized adjacent Q drift; e180/e200 each have one available in-window neighbor, other candidates have two. No generated-video metrics enter selection.\n'
    (out/'REPORT.md').write_text(report)
    write(out/'COMPLETE.json',dict(status='complete',selected_epoch=epoch,checkpoint=str(checkpoint),sha256=sha256(checkpoint),
        elapsed_seconds=time.perf_counter()-began,artifacts={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()}))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--wait',action='store_true');ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    torch.set_num_threads(1)
    if Path(sys.prefix).name!='wan2.2':raise ValueError('wan2.2 environment required')
    root=a.root.resolve();out=root/'post_training_analysis'
    with (root/'post_training_analysis.lock').open('a') as lock:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(root/'POSTPROCESS_JOB.json',dict(status='waiting_for_training',pid=os.getpid(),candidate_epochs=list(range(180,201)),
            census_epochs=list(range(1,201)),device=a.device,output=str(out),code_sha256=sha256(Path(__file__))))
        try:
            while not (root/'TRAINING_COMPLETE.json').exists():
                if (root/'FAILED.json').exists():raise RuntimeError('training failed; postprocess not executed')
                started=root/'TRAINING_STARTED.json'
                if started.exists():
                    try:os.kill(read(started)['pid'],0)
                    except ProcessLookupError:
                        if (root/'TRAINING_COMPLETE.json').exists():break
                        raise
                if not a.wait:raise RuntimeError('training incomplete; use --wait')
                time.sleep(15)
            write(root/'POSTPROCESS_JOB.json',dict(status='running',pid=os.getpid(),output=str(out)))
            run(root,out,a.device)
            write(root/'POSTPROCESS_JOB.json',dict(status='complete',output=str(out),**{'selected_epoch':read(out/'COMPLETE.json')['selected_epoch']}))
        except BaseException as exc:
            write(root/'POSTPROCESS_JOB.json',dict(status='failed',error=repr(exc),output=str(out)));raise

if __name__=='__main__':main()
