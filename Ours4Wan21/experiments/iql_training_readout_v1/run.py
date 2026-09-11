"""Render completed offline training evidence independently of video evaluation."""
import sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]/'iql_aggressiveness_2x4_v1'
sys.path.insert(0,str(BASE))
from common import *
from ours4wan21.contracts import create_nested_result
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'training_readout'
COLORS={'Original':'#454545','a1':'#84adcb','a2':'#4d93bb','a3':'#226b98','a4':'#123c5c'}
STYLES={'Original':':','a1':'--','a2':'-.','a3':'-','a4':(0,(5,1,1,1))}

def main():
    config=read(ROOT/'config.json');verify_sources(config)
    create_nested_result(OUT,ROOT,(Path(__file__).with_name('README.md')).read_text())
    source={};qrows=[];training=[];selected={};windows=[]
    for feature in FEATURES:
        groups=[g for g in config['groups'] if g['feature']==feature]
        cases=[('Original',Path(groups[0]['baseline']),Path(groups[0]['baseline_analysis']))]
        cases += [(g['level'],Path(g['training']),Path(g['analysis'])) for g in groups]
        for level,path,analysis in cases:
            s=read(analysis/'checkpoint_selection.json');selected[feature,level]=s['checkpoint_epoch']
            p=path/'epoch_metrics.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines()]
            assert [r['epoch'] for r in rows]==list(range(1,401))
            source[str(p)]=sha256(p);source[str(analysis/'checkpoint_selection.json')]=sha256(analysis/'checkpoint_selection.json')
            for r in rows:
                row=dict(feature=feature,level=level,epoch=r['epoch'])
                for split in ('train','val'):
                    assert all(math.isfinite(v) for v in r[split].values())
                    row.update({split+'_'+k:v for k,v in r[split].items()})
                training.append(row)
            for lo,hi in [(251,300),(301,350),(351,400)]:
                window=rows[lo-1:hi]
                windows.append(dict(feature=feature,level=level,start=lo,end=hi,
                    **{split+'_'+key:float(np.mean([r[split][key] for r in window]))
                       for split in ('train','val') for key in ('q_loss','v_loss','pi_loss')}))
        for g in groups:
            extra=Path(g['analysis'])/'extra';p=extra/'q_statistics.csv';m=read(extra/'COMPLETE.json')
            assert sha256(p)==m['statistics_sha256'];source[str(p)]=sha256(p)
            for r in csv.DictReader(p.open()):
                if r['method']=='Original' and g['level']!='a1':continue
                qrows.append(dict(feature=feature,level='Original' if r['method']=='Original' else g['level'],
                    epoch=int(r['epoch']),mean_Q=float(r['mean_Q']),Q_IQR=float(r['Q_IQR']),
                    Q_p25=float(r['Q_p25']),Q_p75=float(r['Q_p75']),states=int(r['states']),actions=int(r['actions'])))
    assert len(qrows)==len(training)==4000
    assert all(math.isfinite(r[k]) for r in qrows for k in ('mean_Q','Q_IQR'))
    writecsv(OUT/'q_statistics.csv',qrows);writecsv(OUT/'training_metrics.csv',training);writecsv(OUT/'late_windows.csv',windows)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig,name):
        fig.savefig(OUT/(name+'.png'),dpi=170);fig.savefig(OUT/(name+'.svg'));plt.close(fig)
    for feature in FEATURES:
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for level in ('Original',*LEVELS):
            rr=[r for r in training if r['feature']==feature and r['level']==level]
            assert len(rr)==400
            for i,split in enumerate(('train','val')):
                for j,(metric,title) in enumerate([('q_loss','Q loss'),('v_loss','V loss'),('pi_loss','Actor loss')]):
                    values=[r[split+'_'+metric] for r in rr];assert min(values)>0
                    ax=axes[i,j]
                    label=f"{level.upper() if level!='Original' else level} (e{selected[feature,level]})"
                    ax.plot(range(1,401),values,color=COLORS[level],ls=STYLES[level],lw=1.2,label=label)
                    ax.set(title=('Training' if split=='train' else 'Validation')+' | '+title,
                        xlabel='Epoch',ylabel='Loss (log scale)',yscale='log',xlim=(1,400))
                    ax.grid(alpha=.16)
        axes[0,0].legend(fontsize=8.5,ncol=1)
        fig.suptitle(('SEA7' if feature=='sea7' else 'Dynamics128')+' | Original and four IQL settings | all 400 epochs\n'
            'Logarithmic loss axes; changing objectives means raw losses do not rank video quality',fontsize=13)
        save(fig,feature+'_training')
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for col,feature in enumerate(FEATURES):
        for row,metric in enumerate(('mean_Q','Q_IQR')):
            ax=axes[row,col]
            for level in ('Original',*LEVELS):
                rr=[r for r in qrows if r['feature']==feature and r['level']==level]
                assert [r['epoch'] for r in rr]==list(range(1,401))
                label=f"{level.upper() if level!='Original' else level} (e{selected[feature,level]})"
                ax.plot(range(1,401),[r[metric] for r in rr],color=COLORS[level],ls=STYLES[level],lw=1.5,label=label)
                e=selected[feature,level];ax.scatter([e],[rr[e-1][metric]],s=30,color=COLORS[level],zorder=5)
            ax.set(title=('SEA7' if feature=='sea7' else 'Dynamics128')+' | '+metric,
                xlabel='Epoch',ylabel=metric,xlim=(1,405));ax.grid(alpha=.16)
            if row==0:ax.legend(fontsize=8.5,ncol=2,loc='lower right')
    fig.suptitle('Value estimates | Q = min(Q1, Q2) | same 13,738 validation states x 2 actions\n'
        'Q-IQR = midpoint P75 - P25; dots mark validation-selected checkpoints',fontsize=13)
    save(fig,'q_statistics')
    dump(OUT/'VALIDATION.json',dict(status='pass',training_rows=4000,q_rows=4000,epochs=400,
        source_sha256=source,visual_qa='pending',scope='completed offline training only; video metrics not used'))
    print(str(OUT))

if __name__=='__main__':main()
