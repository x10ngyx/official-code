"""Read completed R1 artifacts without disturbing the running online pipeline."""
import importlib.util
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Patch
import numpy as np

HELPER=Path(__file__).resolve().parents[1]/'iql_increase_traces_v1/plot.py'
spec=importlib.util.spec_from_file_location('trace_helpers',HELPER)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
RUN=h.BASE/'ours21_dynamics128_e391_online_offline800_8rounds_a25_v1'
R1=RUN/'rounds/round_001'
OUT=RUN/'analysis/r1_increase_traces'
h.OUT=OUT

def draw(rr,page):
    n=len(rr);fig=plt.figure(figsize=(20,12.5))
    ax=fig.add_axes([.22,.11,.54,.72]);mt=fig.add_axes([.775,.11,.212,.72])
    fig.text(.025,.967,f'在线第一轮 R1 与 Increase：实测 trace（{page}/3）',fontsize=22,weight='bold',va='top')
    fig.text(.025,.92,f'100条R1 + Increase五档各1条 · 按实际skip数全局升序 · 本页排序 {(page-1)*35+1}–{page*35}',fontsize=12,color=h.GRAY)
    fig.legend(handles=[Patch(facecolor=h.BLUE,label='R1采集（起始e391采样）：重算'),Patch(facecolor=h.ORANGE,label='Increase：重算'),
        Patch(facecolor='white',edgecolor=h.GRID,label='白色：skip / 复用')],loc='upper left',bbox_to_anchor=(.02,.891),frameon=False,ncol=3,fontsize=11)
    a=np.array([[0 if c=='1' else (2 if r['family']=='Increase' else 1) for c in r['skip_path']] for r in rr])
    ax.pcolormesh(np.arange(51),np.arange(n+1),a,cmap=ListedColormap(['white',h.BLUE,h.ORANGE]),vmin=0,vmax=2,edgecolors=h.GRID,linewidth=.42)
    ax.set(xlim=(0,50),ylim=(n,0));ticks=[1,5,10,15,20,25,30,35,40,45,50]
    ax.set_xticks([x-.5 for x in ticks],ticks);ax.set_yticks(np.arange(n)+.5,[r['label'] for r in rr],fontsize=10.5)
    ax.tick_params(length=0,pad=6);ax.axvline(25,color=h.INK,ls='--',lw=1.1);ax.spines[:].set_visible(False)
    ax.set_xlabel('采样步（实际执行顺序，1 → 50）；虚线右侧为后25步',labelpad=10,fontsize=11)
    mt.set(xlim=(0,1),ylim=(n,0));mt.axis('off')
    cols=[(.07,'skip','skip_steps'),(.28,'后25','skip_last25'),(.50,'最长¹','longest_last25'),(.73,'PSNR','psnr'),(.95,'加速','speedup')]
    for x,title,key in cols:mt.text(x,-.85,title,ha='center',fontsize=11,weight='bold')
    for i,r in enumerate(rr):
        for x,title,key in cols:
            value=f"{r[key]:.2f}" if key=='psnr' else f"{r[key]:.2f}×" if key=='speedup' else str(r[key])
            mt.text(x,i+.5,value,ha='center',va='center',fontsize=10.5)
        if i and r['skip_steps']!=rr[i-1]['skip_steps']:
            ax.axhline(i,color=h.INK,lw=1.05);mt.axhline(i,color=h.GRID,lw=.7)
    fig.text(.025,.055,'R1来自OpenVid训练prompt；Increase固定VBench155。不同prompt，只比较路径结构，不作配对画质结论。',fontsize=11,color=h.GRAY)
    fig.text(.025,.029,'¹ 后25步内最长连续skip；PSNR单位dB。R1采集发生在本轮训练之前，使用e391动作采样；不是R1训练后checkpoint的评测。',fontsize=10.5,color=h.GRAY)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,t.get_text()
    for ext in ['png','svg']:fig.savefig(OUT/f'r1_increase_page{page}.{ext}',dpi=180,facecolor='white')
    plt.close(fig)

def main():
    manifest=h.js(RUN/'manifest.json');plan=h.js(R1/'plan.json');assert len(plan['rows'])==100 and plan['round']==1
    assert plan['parent']['path']==manifest['paths']['start_checkpoint']
    assert h.sha(plan['parent']['path'])==plan['parent']['sha256']
    qseal=h.js(R1/'quality/COMPLETE.json')
    for name,checksum in qseal['files'].items():h.read(R1/'quality'/name,checksum)
    quality={r['video_id']:r for r in h.rows(R1/'quality/metrics/per_video.csv')}
    assert set(quality)=={r['trajectory_id'] for r in plan['rows']}
    rr=[]
    for row in plan['rows']:
        uid=row['trajectory_id'];p=R1/'collection'/uid;seal=h.js(p/'COMPLETE.json')
        for name in ['trace.json','timing.json','measurement.json','generation.json']:h.read(p/name,seal['files'][name])
        job=seal['identity']['job'];assert job['checkpoint']==plan['parent'] and job['sampling_seed']==row['sampling_seed']
        assert job['sample_id']==row['sample_id'] and job['skip_budget']==row['skip_budget'] and job['kind']=='collection'
        trace=h.js(p/'trace.json');expected=''.join('1' if i in trace['per_branch']['cond']['reuse_path'] else '0' for i in range(50))
        parsed=h.audit(p/'trace.json',p/'timing.json',uid,expected);assert parsed['skip_steps']==row['skip_budget']
        q=quality[uid];assert q['candidate_sha256']==seal['files']['video.mp4']
        assert (int(q['frames']),int(q['width']),int(q['height']))==(81,832,480)
        baseline=Path(manifest['paths']['training_bundle'])/'baselines'/row['sample_id'];bs=h.js(baseline/'COMPLETE.json')
        assert q['reference_sha256']==bs['files']['video.mp4']
        h.read(baseline/'measurement.json',bs['files']['measurement.json'])
        seconds=h.js(p/'measurement.json')['generate_seconds'];base_seconds=h.js(baseline/'measurement.json')['generate_seconds']
        rr.append(dict(trace_id=uid,family='R1',sample_id=row['sample_id'],prompt=row['prompt'],sampling_seed=row['sampling_seed'],
            label=f"R1 #{uid[-4:]} / OpenVid {row['sample_id'][-4:]}",psnr=float(q['psnr_rgb_db_mean']),speedup=base_seconds/seconds,
            source_trace=str(p/'trace.json'),**parsed))
    oldval=h.js(h.PREV/'VALIDATION.json')
    for p,checksum in oldval['source_sha256'].items():h.read(p,checksum)
    old=[r for r in h.rows(h.PREV/'trace_summary_sorted.csv') if r['method']=='increase' and r['sample_id']=='vbench200_155']
    assert len(old)==5
    for row in old:
        p=Path(row['source_trace']);uid='increase_'+row['level']+'_v155'
        parsed=h.audit(p,p.parent.parent/'timings'/p.name,uid,row['skip_path'])
        rr.append(dict(trace_id=uid,family='Increase',sample_id=row['sample_id'],prompt=row['prompt_en'],sampling_seed='',
            label=row['label']+' / V155',psnr=float(row['psnr_rgb_db']),speedup=float(row['latency_speedup']),source_trace=str(p),**parsed))
    rr.sort(key=lambda r:(r['skip_steps'],r['family'],r['trace_id']))
    assert len(rr)==105 and len(h.RAW)==10500
    for i,r in enumerate(rr):r['display_rank']=i+1
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT.parent/'README.md').write_text('# Online analysis\n\nr1_increase_traces/: completed R1 measured traces against archived manual Increase. CPU-only readout; pipeline artifacts remain unchanged.\n')
    (OUT/'README.md').write_text(Path(__file__).with_name('README.md').read_text())
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'svg.fonttype':'path','font.size':11,'text.color':h.INK,'xtick.color':h.GRAY,'ytick.color':h.INK})
    for page in range(1,4):draw(rr[(page-1)*35:page*35],page)
    h.writecsv('traces_sorted.csv',rr);h.writecsv('branch_steps.csv',h.RAW)
    h.writecsv('step_actions.csv',[dict(trace_id=r['trace_id'],step=i+1,skip=int(v)) for r in rr for i,v in enumerate(r['skip_path'])])
    h.writecsv('prompts.csv',[dict(trace_id=r['trace_id'],sample_id=r['sample_id'],prompt=r['prompt']) for r in rr])
    h.SOURCES[str(Path(__file__))]=h.sha(Path(__file__));h.SOURCES[str(HELPER)]=h.sha(HELPER)
    assert all(h.sha(p)==s for p,s in h.SOURCES.items())
    (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',r1_traces=100,increase_traces=5,cfg_calls=10500,cells=5250,
        parent_checkpoint=plan['parent'],r1_sampling=True,paired_prompt_comparison=False,source_sha256=h.SOURCES,
        figures_sha256={p.name:h.sha(p) for ext in ['png','svg'] for p in OUT.glob('*.'+ext)},visual_qa='text bounds pass; actual PNG review pending'),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='pass',traces=105,skip_range=[min(r['skip_steps'] for r in rr),max(r['skip_steps'] for r in rr)],output=str(OUT)),ensure_ascii=False))

if __name__=='__main__':main()
