"""Full 400-epoch Q diagnostics on the same discretionary validation population."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import PROJECT,ROOT,read
import argparse
from ours4wan21.contracts import MODEL_ROOT,dump,sha256
from ours4wan21.analysis import midpoint_iqr,write_csv
from ours4wan21.local_iql import IQLModelConfig,QNet,apply_normalizer




def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'config.json');args=parser.parse_args()
    config=read(args.config);g=next(g for g in config['groups'] if g['name']==args.group)
    BASELINE,TRAINING,ANALYSIS,WEIGHTS,CACHE=[Path(g[k]) for k in ('baseline','training','analysis','weights','cache')]
    OLD_NAME=Path(g['baseline_weights']).name
    title=g['feature']+' '+g['level'].upper()
    torch.set_num_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    out=ANALYSIS/'extra';out.mkdir()
    bundle=torch.load(CACHE/'transitions.pt',map_location='cpu',weights_only=False)
    positions=bundle['val_indices']
    positions=positions[bundle['tensors']['actor_mask'][positions]>.5]
    assert len(positions)==13738
    methods=[('Original',BASELINE,BASELINE.with_name(OLD_NAME+'_analysis'),MODEL_ROOT/OLD_NAME),
             ('Aggressive',TRAINING,ANALYSIS,WEIGHTS)]
    allrows=[];normalizer=None;selections={};hashes={};logs={}
    for label,training,analysis,weights in methods:
        selections[label]=json.loads((analysis/'checkpoint_selection.json').read_text())['checkpoint_epoch']
        logs[label]=[json.loads(s) for s in (training/'epoch_metrics.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in logs[label]]==list(range(1,401))
        cached=np.load(analysis/'post300_validation_values.npz')
        assert np.array_equal(cached['row_index'],positions.numpy())
        assert np.array_equal(cached['epochs'],np.arange(300,401))
        first=torch.load(weights/'checkpoints/epoch_001.pt',map_location='cpu',weights_only=False)
        if normalizer is not None:
            assert all(torch.equal(normalizer[k],first['normalizer'][k]) for k in ('mean','std'))
        normalizer=first['normalizer']
        states=apply_normalizer(bundle['tensors']['state'][positions],normalizer).cuda()
        mc=IQLModelConfig(**first['model_config'])
        nets={k:QNet(mc).cuda().eval() for k in ('q1_net','q2_net')}
        for epoch in range(1,401):
            if epoch<300:
                path=weights/'checkpoints'/f'epoch_{epoch:03d}.pt'
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                assert checkpoint['epoch']==epoch and checkpoint['train_config']==first['train_config']
                assert all(torch.equal(normalizer[k],checkpoint['normalizer'][k]) for k in ('mean','std'))
                for key,net in nets.items():net.load_state_dict(checkpoint[key])
                with torch.inference_mode():
                    values=np.concatenate([torch.minimum(nets['q1_net'](x),nets['q2_net'](x)).cpu().numpy()
                        for x in states.split(2048)])
                hashes[str(path)]=sha256(path)
            else:values=cached['qmin'][epoch-300]
            assert values.shape==(len(positions),2) and np.isfinite(values).all()
            ordered=np.sort(values.astype(np.float64).ravel());centers=(np.arange(len(ordered))+.5)/len(ordered)
            p25,p75=np.interp([.25,.75],centers,ordered)
            allrows.append(dict(method=label,epoch=epoch,mean_Q=float(ordered.mean()),
                Q_IQR=midpoint_iqr(values),Q_p25=float(p25),Q_p75=float(p75),states=len(positions),actions=2))
            if epoch%50==0:print(json.dumps(dict(method=label,epoch=epoch)),flush=True)
        hashes[str(analysis/'post300_validation_values.npz')]=sha256(analysis/'post300_validation_values.npz')
        cached.close()
    write_csv(out/'q_statistics.csv',allrows)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors={'Original':'#3167ad','Aggressive':'#df6a32'}
    for metric,title in [('mean_Q','Mean Q'),('Q_IQR','Q interquartile range')]:
        fig,ax=plt.subplots(figsize=(10,4.7),layout='constrained')
        for label,_,_,_ in methods:
            rr=[r for r in allrows if r['method']==label]
            ax.plot([r['epoch'] for r in rr],[r[metric] for r in rr],color=colors[label],label=label)
            selected=selections[label];r=rr[selected-1]
            ax.scatter([selected],[r[metric]],color=colors[label],s=45,zorder=5,label=f'{label} selected e{selected}')
        ax.set(xlabel='Training epoch',ylabel=metric,title=title+' | '+g['feature']+' '+g['level'].upper()+', same validation states',xlim=(1,405))
        ax.grid(alpha=.18);ax.legend(fontsize=9)
        fig.savefig(out/(metric+'.png'),dpi=180);fig.savefig(out/(metric+'.svg'));plt.close(fig)
    keys=['q_loss','v_loss','pi_loss','adv_mean','weight_mean','policy_skip_rate']
    available=list(logs['Original'][0]['val'])
    # Use the exact logged metric names, and fail clearly if a requested diagnostic disappeared.
    aliases={'adv_mean':['adv_mean','advantage_mean'],'weight_mean':['weight_mean','actor_weight_mean'],
             'policy_skip_rate':['skip_rate_policy']}
    keys=[next((v for v in aliases.get(k,[k]) if v in available),k) for k in keys]
    assert set(keys)<=set(available),(keys,available)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for key,ax in zip(keys,axes.flat):
        for label in logs:
            for split,style in [('train','--'),('val','-')]:
                ax.plot(range(1,401),[r[split][key] for r in logs[label]],style,color=colors[label],
                    alpha=.8,lw=1,label=f'{label} {split}')
        ax.set(title=key,xlabel='Epoch');ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle(g['feature']+' '+g['level'].upper()+' | changed objectives: raw losses do not rank policy quality')
    fig.savefig(out/'training_comparison.png',dpi=160);fig.savefig(out/'training_comparison.svg');plt.close(fig)
    lines=['# Full-400-epoch Q and training diagnostics','',
        'Q = min(Q1,Q2), pooled over the same 13,738 discretionary validation states × both actions.',
        'mean_Q is the population mean; Q-IQR uses midpoint empirical quartiles (P75−P25), matching checkpoint selection.',
        'Epochs 1–299 are recomputed from saved checkpoints; epochs 300–400 reuse audited Q caches.',
        'Original and aggressive use identical training-derived normalizers and evaluation rows. No test or video quality enters selection.',
        'Changed expectile/advantage weighting changes optimization objectives. Higher mean_Q or smaller Q-IQR alone does not establish better video quality.',
        '', '| Method | Selected epoch | mean_Q | Q-IQR |','|---|---:|---:|---:|']
    for label,e in selections.items():
        r=next(r for r in allrows if r['method']==label and r['epoch']==e)
        lines.append(f"| {label} | {e} | {r['mean_Q']:.6f} | {r['Q_IQR']:.6f} |")
    lines+=['','## Late training windows','',
        '| Method | Epochs | mean_Q | Q-IQR | Train actor loss | Validation actor loss | Validation Q loss |',
        '|---|---|---:|---:|---:|---:|---:|']
    for label in logs:
        for lo,hi in [(251,300),(301,350),(351,400)]:
            qr=[r for r in allrows if r['method']==label and lo<=r['epoch']<=hi]
            lr=[r for r in logs[label] if lo<=r['epoch']<=hi]
            vals=[np.mean([r[k] for r in qr]) for k in ('mean_Q','Q_IQR')]
            vals += [np.mean([r[s][k] for r in lr]) for s,k in [('train','pi_loss'),('val','pi_loss'),('val','q_loss')]]
            lines.append(f'| {label} | {lo}–{hi} | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    dump(out/'COMPLETE.json',dict(status='complete',epochs_per_method=400,methods=2,validation_states=len(positions),
        q_definition='min(Q1,Q2); pool both actions',selections=selections,sources_sha256=hashes,
        statistics_sha256=sha256(out/'q_statistics.csv')))


if __name__=='__main__':main()
