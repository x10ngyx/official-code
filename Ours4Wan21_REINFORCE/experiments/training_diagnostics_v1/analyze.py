import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert a.output.resolve().is_relative_to(a.run.resolve())
a.output.mkdir(parents=True,exist_ok=True)
updates=sorted((a.run/'batches').glob('*/update.json')); assert len(updates)>=20
rows=[];batches=[];sources={}
for ix,u in enumerate(updates):
 assert int(u.parent.name)==ix
 d=read(u);sources[str(u)]=sha(u)
 jobs=read(u.parent/'plan.json');assert len(jobs)==32
 for j in jobs:
  root=a.run/j['output'];marker=read(root/'COMPLETE.json')
  assert marker['identity']['job']==j and j['split']=='train'
  assert marker['files']['episode.pt']==d['trajectories'][j['id']]
  q=root/'quality.json';assert sha(q)==marker['files']['quality.json'];sources[str(q)]=sha(q)
  val=read(q)['video']['psnr_rgb_db_mean'];assert np.isfinite(val)
  rows.append(dict(batch=ix+1,epoch=ix//25+1,prompt=j['sample_id'],k=j['k'],psnr=val))
 batchrows=rows[-32:];assert abs(np.mean([r['psnr'] for r in batchrows])-d['metrics']['reward_mean'])<1e-8
 batches.append(dict(batch=ix+1,**d['metrics'],mean_k=np.mean([r['k'] for r in batchrows])))
df=pd.DataFrame(rows);bf=pd.DataFrame(batches);n=len(bf)
early=df[df.batch<=10];late=df[df.batch>n-10]
def describe(x):return dict(n=len(x),psnr_mean=float(x.psnr.mean()),psnr_median=float(x.psnr.median()),psnr_p10=float(x.psnr.quantile(.1)),fraction_below_15=float((x.psnr<15).mean()),mean_k=float(x.k.mean()))
byk=pd.DataFrame({'early':early.groupby('k').psnr.mean(),'late':late.groupby('k').psnr.mean(),'early_n':early.groupby('k').size(),'late_n':late.groupby('k').size()});assert len(byk)==21 and not byk.isna().any().any();byk['delta']=byk.late-byk.early
# Descriptive equal-K standardization; holds budget mixture fixed, not prompt identity.
k_mean=df.groupby('k').psnr.mean();df['k_adjusted_psnr']=df.psnr-df.k.map(k_mean)+k_mean.mean()
bf['k_adjusted_psnr']=df.groupby('batch').k_adjusted_psnr.mean().to_numpy()
paired=df[df.epoch==1].merge(df[df.epoch==2],on='prompt',suffixes=('_e1','_e2'))
same=paired[paired.k_e1==paired.k_e2].copy();same['delta']=same.psnr_e2-same.psnr_e1
# Matching removes observed prompt/K differences; small observational subset, no convergence claim.
same_summary=dict(n=len(same))
if len(same):same_summary.update(mean_delta=float(same.delta.mean()),median_delta=float(same.delta.median()),fraction_improved=float((same.delta>0).mean()))
windows=[]
for lo in range(1,n+1,10):
 hi=min(lo+9,n);x=df[df.batch.between(lo,hi)];m=bf[bf.batch.between(lo,hi)]
 windows.append(dict(batches=f'{lo}-{hi}',**describe(x),entropy=float(m.entropy.mean()),advantage=float(m.advantage_mean.mean())))
summary=dict(time=datetime.now().astimezone().isoformat(),run=str(a.run),committed_batches=n,trajectories=len(df),first10=describe(early),last10=describe(late),raw_delta=float(late.psnr.mean()-early.psnr.mean()),equal_k_early=float(byk.early.mean()),equal_k_late=float(byk.late.mean()),equal_k_delta=float(byk.delta.mean()),k_groups_improved=int((byk.delta>0).sum()),same_prompt_same_k=same_summary,entropy_first10=float(bf.head(10).entropy.mean()),entropy_last10=float(bf.tail(10).entropy.mean()),gradient_clipped_fraction=float((bf.gradient_norm_before_clip>1).mean()),max_logp_error=float(bf.behavior_logp_max_error.max()),windows=windows,limitations=['Online training rollouts, not frozen held-out evaluations.','Early/late differ in prompt mix; equal-K standardization does not control prompt mix.','Input normalization changes after batch 1 and cumulatively thereafter.','Entropy decline alone does not imply improvement or collapse.','No causal confidence intervals: batches sequentially dependent and policy adaptive.'])
for name,x in [('trajectories',df),('batches',bf.drop(columns=['k_histogram'])),('by_k',byk.reset_index()),('same_prompt_same_k',same)]:x.to_csv(a.output/(name+'.csv'),index=False)
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
fig,axes=plt.subplots(3,1,figsize=(11,10),layout='constrained')
ax=axes[0];ax.plot(bf.batch,bf.reward_mean,color='#aaaaaa',lw=1,label='Batch mean (32 trajectories)');ax.plot(bf.batch,bf.reward_mean.rolling(5,min_periods=5).mean(),color='#225ea8',lw=2,label='Trailing 5-batch mean');ax.plot(bf.batch,bf.k_adjusted_psnr.rolling(5,min_periods=5).mean(),color='#be7917',lw=2,ls='--',label='Trailing mean, K-adjusted (descriptive)');ax.set(ylabel='PSNR (dB)',title=f'Training PSNR | {n} committed batches, {len(df):,} trajectories');ax.legend(loc='lower right',fontsize=9)
ax=axes[1];ax.plot(byk.index,byk.early,'o-',color='#225ea8',label='First 10 batches (320 trajectories)');ax.plot(byk.index,byk.late,'s--',color='#be7917',label='Last 10 batches (320 trajectories)');ax.set(xlabel='Exact reuse budget K',ylabel='Mean PSNR (dB)',title='Early / late comparison at the same K (different prompt mixes)');ax.set_xticks(range(20,41,2));ax.legend(fontsize=9)
ax=axes[2];ax.plot(bf.batch,bf.entropy,color='#225ea8',lw=1.5);ax.axhline(np.log(2),color='#555555',ls=':',label='Binary maximum: ln(2)');ax.set(xlabel='Batch / policy update (1-based)',ylabel='Policy entropy (nats)',ylim=(0,.73),title='Entropy over free decisions | lower means more decisive, not necessarily better');ax.legend(fontsize=9)
for ax in [axes[0],axes[2]]:
 ax.set_xlim(1,n);ax.set_xticks(sorted(set([1,n]+list(range(10,n-5,10)))))
 for boundary in range(25,n,25):ax.axvline(boundary+.5,color='#999999',ls=':',lw=1)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.savefig(a.output/'training_trends.png',dpi=160);fig.savefig(a.output/'training_trends.pdf');plt.close(fig)
(a.output/'README.md').write_text('# REINFORCE training diagnostics\n\nSnapshot includes only committed complete batches. See summary.json for early/late and exact-K standardized results, CSVs for all plotted rows, sources.json for SHA256 provenance, and training_trends.png/pdf for the scientific plot. No held-out evaluation or causal convergence claim; 5-batch rolling averages are trailing, minimum 5 batches. K-adjusted series subtracts each K overall mean and adds the equally weighted mean across 21 K values. Equal-K early/late summary separately averages the 21 within-K means. Prompt/K matches compare epoch 1 and epoch 2 with identical prompt and K, different current policies.\n')
print(json.dumps(summary,indent=2))
