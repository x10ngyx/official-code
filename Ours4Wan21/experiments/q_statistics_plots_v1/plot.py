"""Reproducible mean_Q and Q-IQR figures from the saved full-validation census."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/mnt/hdd/xiongyuxiang/tmp/exp')
PREFIX = 'ours21_random3000_12groups_v1'
OUT = ROOT / (PREFIX + '_training_readout') / 'q_statistics'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    summary = ROOT / (PREFIX + '_training_readout') / 'suite_summary.csv'
    groups = list(csv.DictReader(summary.open()))
    assert len(groups) == 12
    rows, sources, common_rows = [], [], None
    for group in groups:
        directory = ROOT / f"{PREFIX}_{group['mode']}_analysis"
        path = directory / 'post300_validation_values.npz'
        with np.load(path) as data:
            epochs, q, indices = data['epochs'], data['qmin'], data['row_index']
        assert np.array_equal(epochs, np.arange(300, 401))
        assert q.shape == (101, len(indices), 2) and np.isfinite(q).all()
        if common_rows is None:
            common_rows = indices.copy()
        assert np.array_equal(indices, common_rows)
        selection = json.loads((directory / 'checkpoint_selection.json').read_text())
        assert len(indices) == selection['validation_actor_free_population']
        assert int(group['selected_epoch']) == selection['checkpoint_epoch']
        iqrs = []
        for epoch, values in zip(epochs, q):
            flat = values.astype(np.float64).reshape(-1)
            ordered = np.sort(flat)
            q25, q75 = np.interp([.25, .75], (np.arange(len(flat)) + .5) / len(flat), ordered)
            iqr = float(q75-q25)
            iqrs.append(iqr)
            rows.append(dict(mode=group['mode'], label=group['label'], kind=group['kind'],
                epoch=int(epoch), mean_Q=float(flat.mean()), Q_IQR=iqr,
                Q_p25=float(q25), Q_p75=float(q75), n_states=len(indices), n_values=len(flat),
                selected_epoch=selection['checkpoint_epoch'], source=str(path)))
        adjacent = list(csv.DictReader((directory / 'post300_adjacent_census.csv').open()))
        assert len(adjacent) == 100
        np.testing.assert_allclose((np.array(iqrs[:-1])+iqrs[1:])/2,
            [float(r['q_iqr_scale']) for r in adjacent], rtol=1e-12, atol=1e-12)
        sources.append(dict(path=str(path), sha256=sha(path)))
    with (OUT/'q_statistics.csv').open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.edgecolor':'#777777',
        'text.color':'#252525', 'axes.labelcolor':'#252525'})
    for key, title in [('mean_Q','mean_Q'), ('Q_IQR','Q-IQR')]:
        fig, axes = plt.subplots(4,3,figsize=(15,12),sharex=True,sharey=True)
        for ax, group in zip(axes.flat, groups):
            data=[r for r in rows if r['mode']==group['mode']]
            ax.plot([r['epoch'] for r in data], [r[key] for r in data],color='#3569A8',lw=1.5)
            selected=next(r for r in data if r['epoch']==r['selected_epoch'])
            ax.plot(selected['epoch'],selected[key],marker='o',ms=6,mfc='white',mec='#252525',ls='none')
            ax.set_title(group['label'],loc='left',fontsize=11)
            ax.grid(axis='y',color='#E5E5E5',lw=.6)
            ax.set_xlim(298,402); ax.set_xticks([300,325,350,375,400])
            ax.tick_params(labelbottom=True,labelleft=True)
        if key=='Q_IQR':
            axes.flat[0].set_ylim(bottom=0)
        fig.suptitle(f'{title} by training group | epochs 300–400',x=.06,ha='left',y=.985,fontsize=20)
        fig.text(.06,.945,f'Validation: {len(common_rows):,} discretionary states × 2 actions | Q = min(Q1, Q2) | shared axes',fontsize=12)
        fig.supxlabel('Epoch',y=.055); fig.supylabel('Q value' if key=='mean_Q' else 'Q value spread (P75 − P25)',x=.015)
        fig.text(.06,.017,'All 101 saved epochs; no smoothing. Open circle: selected checkpoint. IQR: midpoint empirical quantiles.\nSource: per-group post300_validation_values.npz. Offline Q diagnostics; not a video-quality score.',fontsize=10)
        fig.subplots_adjust(left=.07,right=.98,bottom=.12,top=.90,hspace=.48,wspace=.20)
        for ext in ('png','svg'):
            fig.savefig(OUT/f'{key}.{ext}',dpi=160,facecolor='white')
        plt.close(fig)
    (OUT/'README.md').write_text('# mean_Q and Q-IQR\n\nTwo 12-panel charts in PNG/SVG; q_statistics.csv retains all 1,212 group-epoch records.\nQ=min(Q1,Q2), pooled equally across both actions and all discretionary validation states.\nWindow: epochs 300–400 only; earlier Q values were not cached.\nQ-IQR=P75−P25 using midpoint empirical quantile interpolation from the original analysis.\nThe original adjacent census stores the average IQR of two consecutive epochs; these plots show each epoch separately.\nOpen circles mark existing selected checkpoints. No retraining or selection changes.\nReproduce with experiments/q_statistics_plots_v1/plot.py.\n')
    (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',rows=len(rows),groups=len(groups),
        epochs=[300,400],n_states=len(common_rows),all_finite=True,same_validation_rows=True,
        original_adjacent_iqr_reconciliation='rtol=atol=1e-12',sources=sources,
        summary_sha256=sha(summary),script_sha256=sha(Path(__file__))),indent=2)+'\n')
    print(json.dumps(dict(output=str(OUT),rows=len(rows),states=len(common_rows))))

if __name__=='__main__': main()
