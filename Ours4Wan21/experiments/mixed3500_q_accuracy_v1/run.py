import sys,json,csv,hashlib,math,statistics as st
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];sys.path.insert(0,str(PROJECT))
from ours4wan21.local_iql import IQLModelConfig,QNet,PolicyNet,apply_normalizer
from ours4wan21.online_common import verified
R=Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_increase500_iql2_v1');OUT=R/'analysis/q_accuracy';S={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def seal(p,h=None):
 value=sha(p)
 if h is not None:assert value==h,str(p)
 S[str(p)]=value;return p
def read(p):return json.loads(Path(seal(p)).read_text())
def write(name,rows):
 with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=True);(OUT/'README.md').write_text((HERE/'README.md').read_text());cfg=read(R/'config.json');done=read(R/'COMPLETE.json');seal(R/'per_video.csv',done['detail_sha256']);detail=list(csv.DictReader((R/'per_video.csv').open()));assert len(detail)==120
 steps=[];videos=[];max_p_err=0.
 for g in cfg['groups']:
  selected=read(R/g['name']/'SELECTED.json');p=Path(selected['checkpoint']);seal(p,selected['checkpoint_sha256']);ckpt=torch.load(p,map_location='cpu',weights_only=False)
  assert ckpt['train_config']['gamma']==1 and ckpt['train_config']['reward_scale']==1
  mc=IQLModelConfig(**ckpt['model_config']);nets={k:QNet(mc).eval() for k in ['q1_net','q2_net']};actor=PolicyNet(mc).eval();actor.load_state_dict(ckpt['policy_net'])
  for key,net in nets.items():net.load_state_dict(ckpt[key])
  for d in [d for d in detail if d['group']==g['name']]:
   sid=d['sample_id'];k=int(d['k']);folder=R/f"evaluation/candidates/{g['name']}/K{k}/{sid}";assert verified(folder)
   gen=read(folder/'generation.json');assert gen['identity']['job']['checkpoint']['sha256']==selected['checkpoint_sha256']
   tr=read(folder/'trace.json');tim=read(folder/'timing.json');cond=[x for x in tr['decisions'] if x['branch']=='cond'];uncond=[x for x in tr['decisions'] if x['branch']=='uncond'];assert len(cond)==len(uncond)==50
   assert [x['step_index'] for x in cond]==list(range(50))
   for x,y in zip(cond,uncond):assert x['state']==y['state'] and x['action']==y['action'] and x['actor_mask']==y['actor_mask']
   for x,c in zip(tr['decisions'],tim['calls']):assert c['blocks_executed']==(0 if x['action']=='reuse' else 30)
   states=torch.tensor([x['state'] for x in cond],dtype=torch.float32);assert states.shape==(50,135)
   x=apply_normalizer(states,ckpt['normalizer']);actions=torch.tensor([int(x['action']=='reuse') for x in cond]);assert actions.sum()==k
   with torch.inference_mode():
    q1=nets['q1_net'](x);q2=nets['q2_net'](x);q=torch.minimum(q1,q2).gather(1,actions[:,None]).squeeze(1).numpy();probs=actor(x).softmax(-1)[:,1].numpy()
   mask=np.array([x['actor_mask']>.5 for x in cond]);assert mask.any();actual=float(d['psnr_rgb_db']);error=q.astype(float)-actual
   for t,c in enumerate(cond):
    if c['p_skip'] is not None:max_p_err=max(max_p_err,abs(float(probs[t])-c['p_skip']))
    steps.append(dict(group=g['name'],epoch=ckpt['epoch'],k=k,sample_id=sid,step=t,free=bool(mask[t]),action=int(actions[t]),q_pred=float(q[t]),actual_rgb_psnr=actual,error=float(error[t])))
   free=q[mask].astype(float);err=error[mask];first=int(np.flatnonzero(mask)[0])
   videos.append(dict(group=g['name'],epoch=ckpt['epoch'],k=k,sample_id=sid,free_steps=int(mask.sum()),actual_rgb_psnr=actual,q_mean_free=float(free.mean()),bias_free=float(err.mean()),step_mae=float(np.abs(err).mean()),step_mse=float((err**2).mean()),meanq_error=float(free.mean()-actual),first_free_step=first,first_free_q=float(q[first]),first_free_error=float(error[first]),terminal_q=float(q[-1]),terminal_error=float(error[-1])))
 assert max_p_err<2e-5,max_p_err
 assert len(videos)==120 and len(steps)==6000
 summaries=[]
 for group in ['conservative','aggressive']:
  for k in [23,29,35,'all']:
   rr=[r for r in videos if r['group']==group and (k=='all' or r['k']==k)]
   actual=np.array([r['actual_rgb_psnr'] for r in rr]);pred=np.array([r['q_mean_free'] for r in rr]);e=pred-actual
   summaries.append(dict(group=group,k=k,n=len(rr),actual_psnr=float(actual.mean()),q_mean=float(pred.mean()),bias=float(e.mean()),step_mae=st.fmean(r['step_mae'] for r in rr),step_rmse=math.sqrt(st.fmean(r['step_mse'] for r in rr)),trajectory_meanq_mae=float(np.abs(e).mean()),trajectory_meanq_rmse=float(np.sqrt((e*e).mean())),pearson=float(np.corrcoef(actual,pred)[0,1]),first_free_bias=st.fmean(r['first_free_error'] for r in rr),first_free_mae=st.fmean(abs(r['first_free_error']) for r in rr),terminal_mae=st.fmean(abs(r['terminal_error']) for r in rr)))
 write('per_step.csv',steps);write('per_trajectory.csv',videos);write('summary.csv',summaries)
 plt.rcParams.update({'font.size':11,'svg.fonttype':'none'})
 fig,axes=plt.subplots(2,3,figsize=(15,9));fig.subplots_adjust(left=.06,right=.98,top=.89,bottom=.13,hspace=.35,wspace=.25)
 for i,g in enumerate(['conservative','aggressive']):
  for j,k in enumerate([23,29,35]):
   ax=axes[i,j];rr=[r for r in videos if r['group']==g and r['k']==k];a=next(a for a in summaries if a['group']==g and a['k']==k)
   ax.scatter([r['actual_rgb_psnr'] for r in rr],[r['q_mean_free'] for r in rr],c='#2879B0' if i==0 else '#BD4B76',s=30,alpha=.8)
   ax.plot([10,45],[10,45],ls='--',c='gray',lw=1);ax.set(xlim=(10,45),ylim=(10,45),xlabel='Actual RGB PSNR (dB)',ylabel='Mean selected-action Q (free steps)',title=f"{g.capitalize()} | K{k}\nbias {a['bias']:+.2f}, trajectory MAE {a['trajectory_meanq_mae']:.2f}");ax.grid(alpha=.15)
 fig.suptitle('Selected critic Q vs realized rollout RGB PSNR | same 20 prompts per K',fontsize=16)
 fig.text(.06,.04,'Q=min(Q1,Q2), gamma=1, terminal RGB reward. One dot per trajectory; Q averaged over discretionary steps.\nDashed line: perfect numerical agreement. Own rollouts differ across groups; no YUV comparison.',fontsize=11)
 for ext in ['png','svg']:fig.savefig(OUT/f'q_vs_psnr.{ext}',dpi=160,facecolor='white')
 plt.close(fig)
 text=['# Q预估与轨迹实际RGB PSNR差值','', 'Q=min(Q1,Q2)，取每步实际动作；gamma=1，reward_scale=1，只有末步奖励为RGB PSNR。每个CFG双分支共享动作/状态，只计一次。120条各自策略轨迹，同20prompt×三档；强制动作不纳入主指标，完整50步另存。','', '主指标：每条轨迹先计算自由步误差均值/绝对误差/MSE，再按轨迹等权汇总；RMSE为平均轨迹MSE的平方根。正bias表示高估。轨迹meanQ MAE是先平均Q再取绝对误差，会抵消步间正负误差，单独列示。','', '|组|K|N|实际PSNR|平均Q|bias|逐步MAE|逐步RMSE|轨迹meanQ MAE|首自由步MAE|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for a in summaries:text.append(f"|{a['group']}|{a['k']}|{a['n']}|{a['actual_psnr']:.3f}|{a['q_mean']:.3f}|{a['bias']:+.3f}|{a['step_mae']:.3f}|{a['step_rmse']:.3f}|{a['trajectory_meanq_mae']:.3f}|{a['first_free_mae']:.3f}|")
 text+=['','Q学习的是条件期望回报；此处以一条实际后续轨迹检验误差，而且IQL隐式后续策略与部署actor未必一致。因此这是实测回报预测诊断，不是对真实条件期望的直接证明。两组各自在自己轨迹上评估，状态不同；均值Q可能掩盖逐步误差。训练奖励不是YUV PSNR，不能把Q与YUV直接相减。','', f'CPU actor概率回放最大绝对误差={max_p_err:.3g}，核验checkpoint、normalizer与trace状态一致。']
 (OUT/'REPORT.md').write_text('\n'.join(text)+'\n');seal(__file__)
 (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',trajectories=120,unique_step_states=6000,actor_probability_replay_max_error=max_p_err,source_sha256=S,visual_inspection='pending',output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ['.csv','.png','.svg','.md']}),indent=2)+'\n');print('\n'.join(text))
if __name__=='__main__':main()
