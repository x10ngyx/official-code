"""Export complete paired50-step action matrices from the frozen40-trajectory experiment."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import numpy as np

ROOT=Path('/mnt/hdd/xiongyuxiang/tmp/exp/seacache_wan21_linear_increase_matched_5x4_v1')
OUT=ROOT/'analysis/trace_comparison'
BLUE='#376CB1';INK='#202B38';GRAY='#647181';GRID='#CDD5DF'
SOURCES={}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    SOURCES[str(path)]=sha(path)
    return json.loads(Path(path).read_text())

def csvread(path):
    SOURCES[str(path)]=sha(path)
    with Path(path).open() as f:return list(csv.DictReader(f))

def dump(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def writecsv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def load():
    cfg=read(ROOT/'config.json');done=read(ROOT/'COMPLETE.json');match=read(ROOT/'matching.json')
    assert done['status']=='complete' and done['formal_trajectories']==40
    assert sha(ROOT/'analysis/results.csv')==done['results_sha256']
    validation=read(ROOT/'analysis/VALIDATION.json')
    assert sha(ROOT/'analysis/VALIDATION.json')==done['validation_sha256']
    read(ROOT/'VBENCH_SKIPPED_BY_USER.json')
    results={(int(r['strategy']),r['phase']):r for r in csvread(ROOT/'analysis/results.csv')}
    videos={(int(r['strategy']),r['phase'],int(r['gpu'])):r for r in csvread(ROOT/'analysis/per_video.csv')}
    assert len(results)==10 and len(videos)==40
    plots={};branches=[];steps=[];summary=[]
    for i in range(1,6):
        for g in range(4):
            sid=cfg['prompts'][str(g)]['sample_id'];prompt=cfg['prompts'][str(g)]['prompt_en']
            for phase in ('increase','fixed'):
                d=ROOT/'conditions'/f'{phase}_{i}'/f'gpu{g}'
                paths=[d/'traces'/f'{sid}.json',d/'timings'/f'{sid}.json',d/'components.json',d/'run.json']
                for p in paths:assert sha(p)==validation['evidence_sha256'][str(p.relative_to(ROOT))]
                tr,t,component,run=[read(p) for p in paths]
                path=cfg['strategies'][i-1]['path'] if phase=='increase' else [match['pairs'][i-1]['threshold']]*50
                assert run['threshold_path']==tr['threshold_path']==path
                assert run['gpu_uuid']==cfg['gpu_uuids'][str(g)] and run['prompt']['sample_id']==sid
                assert len(tr['decisions'])==len(t['calls'])==100 and tr['total_steps']==50
                vr=videos[i,phase,g];cr=component['rows'][0]
                assert vr['sample_id']==cr['sample_id']==sid
                assert float(vr['generate_seconds'])==cr['generate_seconds']==t['pipeline_generate_wall_seconds']
                decision={b:[] for b in ('cond','uncond')}
                for j,(a,c) in enumerate(zip(tr['decisions'],t['calls'])):
                    assert a['step_index']==j//2 and a['branch']==('cond' if j%2==0 else 'uncond')
                    assert a['requested_threshold']==path[j//2] and a['action']==a['execution']
                    assert c['blocks_executed']==(30 if a['action']=='recompute' else 0)
                    decision[a['branch']].append(int(a['action']=='recompute'))
                    branches.append(dict(strategy=i,phase=phase,gpu=g,sample_id=sid,prompt=prompt,step_index=j//2,display_step=j//2+1,branch=a['branch'],threshold=a['requested_threshold'],action=a['action'],blocks_executed=c['blocks_executed'],relative_l1=a['relative_l1'],accumulator_before=a['accumulator_before'],accumulator_after=a['accumulator_after'],generate_seconds=cr['generate_seconds'],dit_tflops=cr['dit_tflops'],psnr_rgb_db=float(vr['psnr_rgb_db'])))
                assert decision['cond']==decision['uncond'],'separate CFG rows required'
                action=decision['cond'];assert sum(action)*2==tr['recompute']
                assert 50-sum(action)==float(vr['reuse_steps']) and action[0]==action[-1]==1
                plots[i,phase,g]=dict(action=action,path=path,video=vr)
                for k,full in enumerate(action):steps.append(dict(strategy=i,phase=phase,gpu=g,sample_id=sid,prompt=prompt,step_index=k,display_step=k+1,threshold=path[k],action='recompute' if full else 'reuse',cfg_branches_merged=2,psnr_rgb_db=float(vr['psnr_rgb_db'])))
                summary.append(dict(strategy=i,phase=phase,gpu=g,sample_id=sid,prompt=prompt,recompute_steps=sum(action),reuse_steps=50-sum(action),recompute_first25=sum(action[:25]),recompute_last25=sum(action[25:]),first_reuse_step=next((k+1 for k,v in enumerate(action) if v==0),None),psnr_rgb_db=float(vr['psnr_rgb_db']),generate_seconds=cr['generate_seconds'],dit_tflops=cr['dit_tflops']))
    assert len(branches)==4000 and len(steps)==2000
    for i in range(1,6):
        for phase in ('increase','fixed'):
            rows=[r for r in summary if r['strategy']==i and r['phase']==phase];rr=results[i,phase]
            assert math.isclose(st.fmean(r['psnr_rgb_db'] for r in rows),float(rr['psnr_rgb_db']),abs_tol=1e-10)
            speed=sum(x['generate_seconds'] for x in cfg['baselines'].values())/sum(x['generate_seconds'] for x in rows)
            assert math.isclose(speed,float(rr['latency_speedup']),rel_tol=1e-12)
    writecsv(OUT/'branch_steps.csv',branches);writecsv(OUT/'step_actions.csv',steps);writecsv(OUT/'trace_summary.csv',summary)
    prompts=[dict(gpu=g,sample_id=cfg['prompts'][str(g)]['sample_id'],prompt=cfg['prompts'][str(g)]['prompt_en']) for g in range(4)]
    writecsv(OUT/'prompts.csv',prompts)
    (OUT/'PROMPTS.md').write_text('# 完整prompt\n\n'+'\n\n'.join(f"**{r['sample_id']} (GPU{r['gpu']})**\n\n{r['prompt']}" for r in prompts)+'\n')
    return cfg,results,plots,summary


def setup():
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path','figure.facecolor':'white','axes.labelcolor':INK,'text.color':INK,'xtick.color':GRAY,'ytick.color':INK})


def draw(group_ids,cfg,results,plots,filename):
    n=len(group_ids);fig=plt.figure(figsize=(22,6.2 if n==1 else 3.65*n+1.4))
    top=.72 if n==1 else .90;bottom=.17 if n==1 else .048
    gs=fig.add_gridspec(n,3,left=.045,right=.985,top=top,bottom=bottom,width_ratios=[2.55,12,2.7],hspace=.43,wspace=.19)
    fig.text(.045,.985,'SeaCache：线性递增与固定 threshold 的逐步动作对照',ha='left',va='top',fontsize=21,weight='bold')
    fig.text(.045,.90 if n==1 else .96,'同 4 条 prompt · seed 42 · 50 个采样步全部保留 · 右侧 PSNR 为各视频 81 帧均值',ha='left',va='top',fontsize=11.5,color=GRAY)
    fig.legend(handles=[Patch(facecolor=BLUE,edgecolor=BLUE,label='重算（30 blocks）'),Patch(facecolor='white',edgecolor=GRID,label='复用（0 blocks）')],loc='upper right',bbox_to_anchor=(.985,.985),frameon=False,ncol=2,fontsize=11)
    for row,i in enumerate(group_ids):
        ax_t=fig.add_subplot(gs[row,0]);ax=fig.add_subplot(gs[row,1]);ax_m=fig.add_subplot(gs[row,2])
        pos=ax.get_position();inc=results[i,'increase'];fix=results[i,'fixed'];a,b=cfg['strategies'][i-1]['start'],cfg['strategies'][i-1]['end']
        title=f"组 {i}   递增 {a:.2f} → {b:.2f}  /  固定 {float(fix['threshold_start']):.5f}     实测加速 {float(inc['latency_speedup']):.3f}× / {float(fix['latency_speedup']):.3f}×"
        fig.text(.045,pos.y1+.035 if n==1 else pos.y1+.010,title,fontsize=13.5,weight='bold',va='bottom')
        x=np.arange(1,51);ax_t.plot(x,plots[i,'increase',0]['path'],color=BLUE,lw=2,label='递增')
        ax_t.plot(x,plots[i,'fixed',0]['path'],color=GRAY,lw=1.7,ls='--',label='固定')
        ax_t.set(xlim=(1,50),ylim=(0,1.05),xticks=[1,25,50],yticks=[0,.25,.5,.75,1.],xlabel='采样步',ylabel='threshold')
        ax_t.grid(axis='y',color='#E4E8ED',lw=.55);ax_t.spines[['top','right']].set_visible(False)
        for sp in ['bottom','left']:ax_t.spines[sp].set_color(GRID)
        ax_t.legend(loc='upper left',frameon=False,fontsize=10)
        order=[(g,p) for g in range(4) for p in ('increase','fixed')]
        matrix=np.array([plots[i,p,g]['action'] for g,p in order])
        ax.pcolormesh(np.arange(51),np.arange(9),matrix,cmap=ListedColormap(['white',BLUE]),vmin=0,vmax=1,edgecolors=GRID,linewidth=.45,antialiased=True)
        ax.set_xlim(0,50);ax.set_ylim(8,0)
        ticks=[1,5,10,15,20,25,30,35,40,45,50]
        ax.set_xticks([t-.5 for t in ticks],ticks);ax.tick_params(axis='both',length=0,pad=6)
        labels=[f"{cfg['prompts'][str(g)]['sample_id'].split('_')[-1]}   {'递增' if p=='increase' else '固定'}" for g,p in order]
        ax.set_yticks(np.arange(8)+.5,labels);ax.set_xlabel('采样步（按实际 denoise 执行顺序，1 → 50）',labelpad=7)
        for y in [2,4,6]:ax.axhline(y,color=INK,lw=1.05)
        for spine in ax.spines.values():spine.set_visible(False)
        ax_m.set(xlim=(0,1),ylim=(8,0));ax_m.axis('off')
        for xpos,text in zip([.13,.40,.77],['重算','复用','PSNR (dB)']):ax_m.text(xpos,-.43,text,ha='center',va='center',fontsize=10.5,color=GRAY)
        for k,(g,p) in enumerate(order):
            rec=plots[i,p,g];full=sum(rec['action']);vals=[str(full),str(50-full),f"{float(rec['video']['psnr_rgb_db']):.2f}"]
            for xpos,val in zip([.13,.40,.77],vals):ax_m.text(xpos,k+.5,val,ha='center',va='center',fontsize=12)
        for y in [2,4,6]:ax_m.axhline(y,color=GRID,lw=.6)
    fig.text(.045,.025 if n==1 else .018,'双 CFG 分支动作逐步核对一致后合并；全部 4,000 次调用与实际 blocks 一致。完整 prompt 与原始逐步 CSV 随图保存。',fontsize=10,color=GRAY)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();fw,fh=fig.bbox.width,fig.bbox.height
    clipped=[]
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():continue
        bb=text.get_window_extent(renderer)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1>fw+1 or bb.y1>fh+1:clipped.append(text.get_text())
    assert not clipped,clipped
    for ext in ['png','svg']:fig.savefig(OUT/f'{filename}.{ext}',dpi=180,facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True);setup();cfg,results,plots,summary=load()
    draw(list(range(1,6)),cfg,results,plots,'trace_comparison_all')
    for i in range(1,6):draw([i],cfg,results,plots,f'trace_comparison_group_{i}')
    SOURCES[str(Path(__file__).resolve())]=sha(Path(__file__))
    for p,h in SOURCES.items():assert sha(p)==h
    movements=[]
    for i in range(1,6):
        r={p:[x for x in summary if x['strategy']==i and x['phase']==p] for p in ['increase','fixed']}
        movements.append(dict(strategy=i,**{p+'_recompute_first25_mean':st.fmean(x['recompute_first25'] for x in rows) for p,rows in r.items()},**{p+'_recompute_last25_mean':st.fmean(x['recompute_last25'] for x in rows) for p,rows in r.items()}))
    dump(OUT/'VALIDATION.json',dict(status='pass',trajectories=40,branch_records=4000,display_cells=2000,all_cfg_actions_equal=True,all_actions_match_executed_blocks=True,no_steps_omitted=True,source_sha256=SOURCES,figures_sha256={p.name:sha(p) for p in sorted(OUT.glob('*.png'))+sorted(OUT.glob('*.svg'))},recompute_distribution=movements))
    print(json.dumps(dict(status='pass',figures=12,branch_records=4000,display_cells=2000,recompute_distribution=movements)))
if __name__=='__main__':main()
