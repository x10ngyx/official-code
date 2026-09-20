"""Recent-window descriptive trends from a frozen analyze.py CSV snapshot."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('snapshot',type=Path);a=p.parse_args()
d=pd.read_csv(a.snapshot/'trajectories.csv');n=int(d.batch.max())
windows=[]
for lo,hi in [(n-39,n-30),(n-29,n-20),(n-19,n-10),(n-9,n),(n-39,n-20),(n-19,n)]:
 x=d[d.batch.between(lo,hi)];windows.append(dict(start=lo,end=hi,n=len(x),mean_psnr=float(x.psnr.mean()),mean_k=float(x.k.mean()),equal_k_psnr=float(x.groupby('k').psnr.mean().mean())))
slopes={}
for width in [80,60,40,20]:
 x=d[d.batch>n-width].copy();t=x.batch-x.groupby('k').batch.transform('mean');y=x.psnr-x.groupby('k').psnr.transform('mean');slopes[str(width)]=float((t*y).sum()/(t*t).sum()*10)
pairs=[]
for e in range(2,int(d.epoch.max())+1):
 x=d[d.epoch==e-1].merge(d[d.epoch==e],on=['prompt','k'],suffixes=('_before','_after'));v=x.psnr_after-x.psnr_before
 pairs.append(dict(epochs=[e-1,e],n=len(v),mean_delta=float(v.mean()),median_delta=float(v.median()),improved=int((v>0).sum())))
left=d[d.batch.between(n-39,n-20)].groupby('k').psnr.mean();right=d[d.batch>n-20].groupby('k').psnr.mean()
result=dict(committed_batches=n,windows=windows,k_fixed_effect_slope_db_per_10_batches=slopes,adjacent_epoch_same_prompt_same_k=pairs,recent20_k_groups_improved=int((right-left>0).sum()),recent20_k_groups_total=len(right-left),scope='Descriptive adaptive training sequence, not held-out evaluation; no independence-based confidence claims.')
(a.snapshot/'recent_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
