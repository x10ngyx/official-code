"""Validate original per-video means and export all 13 methods' PSNR comparison."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

EXP = Path('/mnt/hdd/xiongyuxiang/tmp/exp')
ROOT = EXP / 'seacache_wan21_vbench5_ours_comparison_4gpu_v1'
OLD = EXP / 'ours21_random3000_12groups_v1_vbench5_speed_targets_v1'
TRAIN = EXP / 'ours21_random3000_12groups_v1_training_readout'
OUT = ROOT / 'analysis/psnr_13methods'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    groups = list(csv.DictReader((TRAIN / 'suite_summary.csv').open()))
    modes = {g['label']: g['mode'] for g in groups}
    methods = list(modes) + ['SeaCache']
    targets = (1.8, 2.4, 3.0)
    source = ROOT / 'analysis/comparison.csv'
    rows = list(csv.DictReader(source.open()))
    data = {(r['label'], float(r['target'])): r for r in rows}
    assert len(rows) == len(data) == 39 and len(methods) == 13
    assert set(data) == {(m,t) for m in methods for t in targets}
    assert json.loads((ROOT / 'COMPLETE.json').read_text())['comparison_sha256'] == sha(source)
    hashes = {str(source): sha(source)}
    details = []
    ids = {f'vbench200_{p}' for p in ('001','016','056','135','159')}
    for (method, target), r in data.items():
        ti = targets.index(target)
        path = (ROOT / f'target_{ti}/quality/per_video.csv' if method == 'SeaCache'
                else OLD / 'candidates' / modes[method] / f"K{r['k']}" / 'quality/video_metrics/per_video.csv')
        videos = list(csv.DictReader(path.open()))
        assert len(videos) == 5 and {v['video_id'] for v in videos} == ids
        assert all((int(v['frames']),int(v['width']),int(v['height'])) == (81,832,480) for v in videos)
        values = [float(v['psnr_rgb_db_mean']) for v in videos]
        assert all(math.isfinite(v) for v in values)
        assert math.isclose(fmean(values), float(r['psnr_rgb_db']), abs_tol=1e-10)
        hashes[str(path)] = sha(path)
        details.extend(dict(method=method,target=target,video_id=v['video_id'],
                            psnr_rgb_db=v['psnr_rgb_db_mean'],source=str(path)) for v in videos)
    OUT.mkdir(exist_ok=True)
    for name, records in [('comparison.csv', rows), ('per_video.csv', details)]:
        with (OUT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'svg.fonttype':'path','text.color':'#222B38'})
    fig, ax=plt.subplots(figsize=(12,10))
    fig.subplots_adjust(left=.30,right=.90,top=.80,bottom=.13)
    cmap=LinearSegmentedColormap.from_list('blue',['#F1F5FC','#3B6FD4','#173E82'])
    norm=Normalize(20,30)
    best={t:max(methods,key=lambda m:float(data[m,t]['psnr_rgb_db'])) for t in targets}
    for i,m in enumerate(methods):
        for j,t in enumerate(targets):
            r=data[m,t];value=float(r['psnr_rgb_db']);color=cmap(norm(value))
            ax.add_patch(Rectangle((j-.48,i-.46),.96,.92,facecolor=color,edgecolor='white'))
            ink='white' if value>=25 else '#222B38'
            ax.text(j,i-.13,f"{value:.3f}"+(' ★' if m==best[t] else ''),ha='center',va='center',
                    fontsize=13,color=ink,weight='bold' if m==best[t] else 'normal')
            ax.text(j,i+.20,f"实测 {float(r['speedup']):.3f}×",ha='center',va='center',fontsize=9,color=ink)
    ax.set_xlim(-.5,2.5);ax.set_ylim(12.5,-.5)
    ax.set_xticks(range(3),['名义 1.8×\n训练组 K=23','名义 2.4×\n训练组 K=29','名义 3.0×\n训练组 K=35'],fontsize=11)
    ax.xaxis.tick_top();ax.tick_params(length=0,pad=10)
    labels=[m+(' [对照]' if m in ('Scalar5','SEA7') else ' [固定阈值]' if m=='SeaCache' else '') for m in methods]
    ax.set_yticks(range(13),labels,fontsize=11)
    for tick,m in zip(ax.get_yticklabels(),methods):
        if m=='SeaCache':tick.set_weight('bold')
    for s in ax.spines.values():s.set_visible(False)
    fig.text(.055,.955,'12 个训练组与 SeaCache：PSNR 对比',fontsize=21,weight='bold')
    fig.text(.055,.917,'同 5 条 prompt · seed=42 · 每视频 81 帧 PSNR 均值，再对 5 个视频等权平均',fontsize=11)
    fig.text(.055,.888,'单位 dB，越高越好；★ 为该名义档位最高均值。每格同时标注实际推理加速比。',fontsize=11)
    cax=fig.add_axes([.925,.20,.018,.53])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,label='PSNR / dB')
    fig.text(.055,.072,'注意：中档 SeaCache 实测 2.464×，训练组约 2.22×，不是等速比较。',fontsize=11)
    fig.text(.055,.044,'所有数值采用同一 VideoMetrics PSNR 口径；仅为五提示词单 seed 诊断，不代表正式 VBench200 结论。',fontsize=10)
    for suffix in ('png','svg'):
        fig.savefig(OUT/f'psnr_13methods.{suffix}',dpi=160,facecolor='white')
    plt.close(fig)
    validation=dict(status='pass',conditions=39,video_rows=len(details),frame_pairs=len(details)*81,
                    mean_recomputed_from_per_video=True,source_sha256=hashes,script_sha256=sha(Path(__file__)),
                    winners={str(t):best[t] for t in targets})
    (OUT/'VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
    for t in targets:
        ranked=sorted(methods,key=lambda m:float(data[m,t]['psnr_rgb_db']),reverse=True)
        print(t,[(m,round(float(data[m,t]['psnr_rgb_db']),3)) for m in ranked])
    print('VALIDATION pass: 39 conditions / 195 videos / 15,795 frame pairs')


if __name__=='__main__':
    main()
