from pathlib import Path
import json,csv,hashlib,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RUN=Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_dynamics128_e391_online_offline800_8rounds_a25_v1')
OUT=RUN/'analysis/training_readout_r1_r4'
COLORS=['#376CB1','#C87524','#778548','#B8678A'];INK='#233345';MUTED='#596879';sources={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,digest=None):
    if digest:assert sha(p)==digest
    sources[str(p)]=sha(p);return json.loads(p.read_text())
def savecsv(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True);histories=[];selections=[];summary=[]
    for n in range(1,5):
        p=RUN/'rounds'/f'round_{n:03d}'/'training';sealed=js(p/'COMPLETE.json')
        for name,digest in sealed['files'].items():js(p/name,digest)
        h=js(p/'epoch_metrics.json');s=js(p/'selection.json');m=js(p/'metrics.json')
        assert [(x['phase'],x['epoch']) for x in h]==[('warmup',i) for i in range(1,6)]+[('joint',i) for i in range(1,21)]
        assert len(s['candidates'])==10 and [x['epoch'] for x in s['candidates']]==list(range(11,21))
        best=max(s['candidates'],key=lambda r:(r['actor_agreement'],-r['probability_drift'],r['epoch']))
        assert best['epoch']==m['selected_epoch']==s['selected_epoch']
        for row in h:
            assert all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
            row.update(round=n,plot_epoch=(n-1)*25+(row['epoch'] if row['phase']=='warmup' else row['epoch']+5))
        for row in s['candidates']:row.update(round=n,plot_epoch=(n-1)*25+5+row['epoch'])
        selected=next(x for x in h if x['phase']=='joint' and x['epoch']==s['selected_epoch'])
        summary.append(dict(round=n,selected_epoch=s['selected_epoch'],replay_transitions=m['replay_transitions'],expected_online_fraction=m['expected_online_fraction'],actor_agreement=best['actor_agreement'],probability_drift=best['probability_drift'],**{k:selected[k] for k in ['q_loss','v_loss','pi_loss','advantage_mean','actor_weight_mean']}))
        histories.append(h);selections.append(s)
    fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'font.size':11,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(20,11.5));fig.subplots_adjust(left=.055,right=.98,top=.82,bottom=.125,wspace=.23,hspace=.43)
    fig.text(.055,.965,'在线微调 R1–R4：训练指标与选点',fontsize=25,weight='bold',color=INK,va='top')
    fig.text(.055,.917,'每轮 5 个 critic warmup + 20 个 joint epoch；仅训练指标。横轴分轮展示，不连接不同轮的终点与起点。',fontsize=12,color=MUTED)
    labels=[f"R{i+1} · 选 joint e{s['selected_epoch']}" for i,s in enumerate(selections)]
    handles=[Line2D([0],[0],color=c,lw=2,label=t) for c,t in zip(COLORS,labels)]
    handles+=[Line2D([0],[0],marker='*',color=INK,ls='',markersize=12,label='选中 checkpoint'),Patch(facecolor='#EDEFF2',label='critic warmup')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.05,.895),ncol=6,frameon=False,fontsize=10.5)
    specs=[('q_loss','Q loss','训练均值',False),('v_loss','V loss','训练均值',False),('pi_loss','Actor loss','训练均值',True),('advantage_mean','优势 A(s,a) 均值','训练 batch 均值',True),('actor_weight_mean','Actor 权重均值','有效 actor 样本的权重均值',True),('actor_agreement','选点 Actor 一致率','固定状态集合上的相邻模型一致率（%）',True)]
    for ax,(key,title,ylabel,joint_only) in zip(axes.flat,specs):
        ax.set_title(title,loc='left',fontsize=15,weight='bold',pad=30,color=INK);ax.set_ylabel(ylabel,fontsize=10)
        for n,(h,s,color) in enumerate(zip(histories,selections,COLORS)):
            begin=25*n
            if key!='actor_agreement':ax.axvspan(begin+.5,begin+5.5,color='#EDEFF2',zorder=0)
            if n:ax.axvline(begin+.5,color='#BCC4CE',lw=.8)
            selected_epoch=begin+5+s['selected_epoch']
            rr=s['candidates'] if key=='actor_agreement' else [r for r in h if not joint_only or r['phase']=='joint']
            scale=100 if key=='actor_agreement' else 1
            ax.plot([r['plot_epoch'] for r in rr],[r[key]*scale for r in rr],color=color,lw=1.8,marker='o',ms=2.6)
            selected=next(r for r in rr if r['plot_epoch']==selected_epoch)
            ax.plot(selected_epoch,selected[key]*scale,marker='*',ms=13,color=color,markeredgecolor=INK,markeredgewidth=.6,zorder=5)
            ax.text((begin+13)/101,1.025,'R'+str(n+1),ha='center',transform=ax.transAxes,color=color,weight='bold',fontsize=11)
        ticks=[n*25+x for n in range(4) for x in (5,15,25)]
        ax.set_xticks(ticks,['W5','J10','J20']*4,fontsize=9)
        ax.set_xlim(.5,100.8);ax.grid(axis='y',color='#D8DDE3',alpha=.7,lw=.6);ax.set_axisbelow(True)
        ax.tick_params(colors=INK);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.set_xlabel('轮内 epoch（W = warmup，J = joint）',fontsize=10)
    fig.text(.055,.070,'优势 / 权重 / actor loss 只画 joint 阶段；warmup 的 actor 零占位不当作训练值。各面板独立线性纵轴，未平滑。',fontsize=10.5,color=MUTED)
    fig.text(.055,.043,'选点一致率是相邻模型动作稳定性，不是任务准确率。replay 随轮次增长，训练 loss 不能直接作为视频质量或验证集泛化指标。',fontsize=10.5,color=MUTED)
    fig.text(.055,.017,'Actor 权重 = min(exp(2.5 × z(A)), 75)，z(A) 按有效样本在 batch 内标准化；优势原始均值下降不等于 skip 倾向增强。',fontsize=10.5,color=MUTED)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    clipped=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            b=t.get_window_extent(renderer)
            if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height:clipped.append(t.get_text())
    assert not clipped,clipped
    for ext in ('png','svg'):fig.savefig(OUT/f'online_training_r1_r4.{ext}',dpi=160,facecolor='white')
    plt.close(fig)
    savecsv('epoch_metrics.csv',[r for h in histories for r in h]);savecsv('selection_candidates.csv',[r for s in selections for r in s['candidates']]);savecsv('selected_summary.csv',summary)
    (OUT/'README.md').write_text('# Online training R1–R4\n\nStandalone six-panel PNG/SVG, complete epoch metrics CSV (100 rows), selection candidates CSV (40 rows), selected summary CSV. The actor weight mean is the logged batch mean; definitions follow online_training.py and its actor loss helper. No validation loss was recorded. Four rounds continue from selected checkpoints, not necessarily the last computed epoch. Source hashes in VALIDATION.json.\n')
    sources[str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
    (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',epoch_rows=100,joint_rows=80,selection_candidates=40,selected_summary=summary,source_sha256=sources,text_bounds='pass',visual_qa='pending PNG inspection',figures_sha256={p.name:sha(p) for p in OUT.glob('online_training_r1_r4.*')}),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
