import sys,json,csv,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.font_manager import FontProperties,fontManager
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'increase500_trace20_v1'))
import plot as source
E=Path('/mnt/hdd/xiongyuxiang/tmp/exp');I=E/'ours21_increase_vbench20_v1';M=E/'ours21_increase500_iql2_v1';OUT=I/'analysis/paired_traces18'
BG={'Increase':'#FFF0CA','conservative':'#DCECF8','aggressive':'#F7E2EA'}
LABEL={'Increase':'Increase','conservative':'保守 e374','aggressive':'激进 e358'}
def draw(rr,name):
 n=len(rr);fig=plt.figure(figsize=(22,5+n*.32),dpi=150);ax=fig.add_axes([.025,.105,.95,.77]);ax.set(xlim=(0,125),ylim=(n,0));ax.axis('off')
 ink='#243243';grid='#B3BCC5';left=26;cell=1.12
 for i,r in enumerate(rr):
  ax.add_patch(Rectangle((0,i),125,1,facecolor=BG[r['method']],edgecolor='white',lw=.6))
  ax.text(.5,i+.5,f"V{r['sample_id'][-3:]}  {LABEL[r['method']]}",va='center',fontsize=11,color=ink)
  for t,a in enumerate(r['skip_path']):
   ax.add_patch(Rectangle((left+t*cell,i+.14),cell,.72,facecolor='white',edgecolor=grid,lw=.35))
   if a=='0':ax.add_patch(Rectangle((left+t*cell+.07,i+.21),cell-.14,.58,facecolor=ink,edgecolor='none'))
  for x,k in [(86,'skip'),(92,'late25'),(98,'longest_late25'),(105,'psnr_rgb_db'),(113,'yuv_611_db'),(121,'speedup')]:
   val=f"{r[k]:.2f}" if k in ['psnr_rgb_db','yuv_611_db'] else f"{r[k]:.3f}×" if k=='speedup' else str(r[k])
   ax.text(x,i+.5,val,ha='center',va='center',fontsize=10.5,color=ink)
  if i%3==2:ax.plot([0,125],[i+1,i+1],color='#77828C',lw=1)
 for t in [0,5,10,15,20,25,30,35,40,45,50]:ax.plot([left+t*cell]*2,[0,n],color=ink if t==25 else grid,ls='--' if t==25 else '-',lw=.8 if t==25 else .35)
 for t in [1,5,10,15,20,25,30,35,40,45,50]:ax.text(left+(t-.5)*cell,-.6,str(t),ha='center',fontsize=11)
 for x,label in [(1,'prompt / 方法'),(86,'skip'),(92,'后25'),(98,'最长¹'),(105,'RGB dB'),(113,'YUV dB'),(121,'实速')]:ax.text(x,-.6,label,ha='left' if x==1 else 'center',weight='bold',fontsize=10.5)
 fig.text(.025,.967,'1.8×档：Increase 与两个训练组的配对 trace',fontsize=22,weight='bold',color=ink)
 fig.text(.025,.942,f'同 prompt 三行相邻；本图 {n//3} 个 prompt / {n} 条 trace，完整保留 50 步。',fontsize=12,color=ink)
 handles=[Patch(facecolor=BG[k],edgecolor=grid,label=LABEL[k]) for k in BG]+[Patch(facecolor=ink,label='重算'),Patch(facecolor='white',edgecolor=grid,label='skip / 复用')]
 fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.02,.929),ncol=5,frameon=False,fontsize=11)
 fig.text(.025,.070,'按 Increase 实际 skip 数升序，再按 prompt ID；保留同 prompt 配对。模型为 K23，Increase 使用既定递增阈值。',fontsize=11,color=ink)
 fig.text(.025,.048,'¹ 后 25 步内最长连续 skip；虚线分隔前 / 后 25 步。步数显示 1–50（原索引 0–49）。',fontsize=11,color=ink)
 fig.text(.025,.026,'右侧为逐视频指标；YUV 为 BT.709 / 6:1:1 加权。所有视频、baseline 和质量评分均复用，不新增推理。',fontsize=11,color=ink)
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for t in fig.findobj(matplotlib.text.Text):
  if t.get_visible() and t.get_text():
   b=t.get_window_extent(renderer);assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,t.get_text()
 for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=150,facecolor='white')
 plt.close(fig)
def main():
 OUT.mkdir(exist_ok=True);(OUT/'README.md').write_text((HERE/'README.md').read_text())
 v=source.read(I/'analysis/all_methods20/VALIDATION.json');path=I/'analysis/all_methods20/per_video.csv';assert source.sha256(path)==v['outputs_sha256']['per_video.csv'];source.SOURCES[str(path)]=source.sha256(path)
 detail=list(csv.DictReader(path.open()));cfg=source.read(I/'config.json');rr=[]
 for p in cfg['prompts']:
  sid=p['sample_id']
  for method in BG:
   out=I/'candidates/1.8'/sid if method=='Increase' else M/f'evaluation/candidates/{method}/K23'/sid
   if method!='Increase':assert source.verified(out)
   for name in ['COMPLETE.json','generation.json','trace.json','timing.json']:
    f=out/name
    if str(f) in v['evidence_sha256']:assert source.sha256(f)==v['evidence_sha256'][str(f)]
    source.SOURCES[str(f)]=source.sha256(f)
   gen=source.read(out/'generation.json');assert gen['gpu_uuid']==p['baseline_gpu_uuid'] and gen['protocol']==cfg['protocol']
   job=gen['job'] if method=='Increase' else gen['identity']['job'];prompt=job['prompt']['prompt'] if method=='Increase' else job['prompt'];assert prompt==p['prompt']
   stats=source.parse(out/'trace.json',out/'timing.json',method+'_'+sid)
   if method!='Increase':assert stats['skip']==23
   q=next(d for d in detail if d['method']==method and d['sample_id']==sid and d['setting']==('target1.8' if method=='Increase' else 'K23'))
   rr.append(dict(method=method,sample_id=sid,prompt=p['prompt'],source_trace=str(out/'trace.json'),speedup=float(q['baseline_seconds'])/float(q['generate_seconds']),psnr_rgb_db=float(q['psnr_rgb_db']),yuv_611_db=float(q['yuv_611_db']),**stats))
 inc={r['sample_id']:r for r in rr if r['method']=='Increase'};rr.sort(key=lambda r:(inc[r['sample_id']]['skip'],r['sample_id'],list(BG).index(r['method'])))
 assert len(rr)==60 and len(source.BRANCHES)==6000
 for name,data in [('per_video.csv',rr),('branch_actions.csv',source.BRANCHES),('step_actions.csv',[dict(method=r['method'],sample_id=r['sample_id'],step=t+1,skip=int(a)) for r in rr for t,a in enumerate(r['skip_path'])])]:
  with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fontManager.addfont(font);plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'svg.fonttype':'path'})
 for name,chunk in [('paired_all20',rr),('paired_01_10',rr[:30]),('paired_11_20',rr[30:])]:draw(chunk,name)
 source.SOURCES[str(Path(source.__file__))]=source.sha256(source.__file__);source.SOURCES[str(HERE/'plot.py')]=source.sha256(HERE/'plot.py')
 assert all(source.sha256(p)==h for p,h in source.SOURCES.items())
 stats={m:dict(skip=st.fmean(r['skip'] for r in rr if r['method']==m),late25=st.fmean(r['late25'] for r in rr if r['method']==m),longest_late25=st.fmean(r['longest_late25'] for r in rr if r['method']==m)) for m in BG}
 (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',traces=60,step_cells=3000,cfg_calls=6000,same_prompt_gpu=True,stats=stats,text_bounds='pass',visual_inspection='pending',source_sha256=source.SOURCES,output_sha256={p.name:source.sha256(p) for p in OUT.iterdir() if p.suffix in ['.png','.svg','.csv']}),indent=2)+'\n');print(json.dumps(stats));print(OUT)
if __name__=='__main__':main()
