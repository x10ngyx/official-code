import json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_increase500_iql2_v1');OUT=R/'analysis/q_statistics';HERE=Path(__file__).resolve().parent
S={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):S[str(p)]=sha(p);return json.loads(Path(p).read_text())
def main():
 OUT.mkdir(exist_ok=True);(OUT/'README.md').write_text((HERE/'README.md').read_text());cfg=read(R/'config.json');rows=[];ids=None;selected={}
 for g in cfg['groups']:
  a=Path(g['analysis']);selection=read(a/'checkpoint_selection.json');selected[g['name']]=selection['checkpoint_epoch'];p=a/'post300_validation_values.npz';S[str(p)]=sha(p)
  with np.load(p) as v:
   assert np.array_equal(v['epochs'],np.arange(300,401))
   if ids is None:ids=v['row_index'].copy()
   else:assert np.array_equal(ids,v['row_index'])
   assert len(ids)==15985 and v['qmin'].shape==(101,15985,2)
   for e,q in zip(v['epochs'],v['qmin']):
    assert np.isfinite(q).all();x=np.sort(q.astype(np.float64).ravel());p25,p75=np.interp([.25,.75],(np.arange(len(x))+.5)/len(x),x)
    rows.append(dict(group=g['name'],epoch=int(e),mean_Q=float(x.mean()),Q_IQR=float(p75-p25),Q_p25=float(p25),Q_p75=float(p75),states=len(ids),actions=2))
  c=a/'post300_adjacent_census.csv';S[str(c)]=sha(c)
  for d in csv.DictReader(c.open()):
   r=next(r for r in rows if r['group']==g['name'] and r['epoch']==int(d['from_epoch']))
   # Selection uses the average IQR of adjacent checkpoints.
   nxt=next(r for r in rows if r['group']==g['name'] and r['epoch']==int(d['to_epoch']))
   assert abs((r['Q_IQR']+nxt['Q_IQR'])/2-float(d['q_iqr_scale']))<1e-10,(r,d)
 with (OUT/'q_statistics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 for metric,title in [('mean_Q','Mean Q'),('Q_IQR','Q interquartile range')]:
  fig,ax=plt.subplots(figsize=(11,5.6));fig.subplots_adjust(left=.10,right=.97,bottom=.21,top=.83)
  for group,color,style in [('conservative','#2879B0','-'),('aggressive','#BD4B76','--')]:
   rr=[r for r in rows if r['group']==group];e=selected[group];chosen=next(r for r in rr if r['epoch']==e)
   ax.plot([r['epoch'] for r in rr],[r[metric] for r in rr],color=color,ls=style,lw=1.8,label=f'{group.capitalize()} (selected e{e})')
   ax.axvline(e,color=color,alpha=.35,ls=':',lw=1)
   ax.scatter([e],[chosen[metric]],color=color,marker='*',s=145,zorder=5)
  ax.set(xlabel='Epoch',ylabel=title,xlim=(300,400));ax.grid(alpha=.18);ax.legend()
  fig.suptitle('Mixed3500 Dynamics128 | '+title,fontsize=17,y=.95)
  fig.text(.1,.08,'Same 15,985 validation discretionary states × 2 actions; Q = min(Q1, Q2).',fontsize=10)
  fig.text(.1,.04,'Cached epochs 300–400 only. IQR = midpoint empirical P75 − P25. Stars: selected checkpoints.',fontsize=10)
  fig.canvas.draw();ren=fig.canvas.get_renderer()
  for t in fig.findobj(matplotlib.text.Text):
   if t.get_visible() and t.get_text():
    b=t.get_window_extent(ren);assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,t.get_text()
  for ext in ['png','svg']:fig.savefig(OUT/f'{metric}.{ext}',dpi=180,facecolor='white')
  plt.close(fig)
 chosen=[r for r in rows if r['epoch']==selected[r['group']]];print(json.dumps(chosen,indent=2))
 S[str(HERE/'plot.py')]=sha(HERE/'plot.py');assert all(sha(p)==h for p,h in S.items())
 (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',rows=202,epochs=[300,400],states=15985,actions=2,selected=chosen,same_population=True,adjacent_census_iqr='pass',source_sha256=S,text_bounds='pass',visual_inspection='pending',output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ['.png','.svg','.csv']}),indent=2)+'\n')
if __name__=='__main__':main()
