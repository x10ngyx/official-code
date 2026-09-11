"""Plot selected, audited measured traces from eight IQL groups and manual Increase."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Patch
import numpy as np

BASE = Path('/mnt/hdd/xiongyuxiang/tmp/exp')
SUITE = BASE/'ours21_iql_aggressiveness_2x4_v1'
PREV = BASE/'wan21_increase_e391_trace_comparison_v1'
INC = BASE/'seacache_wan21_linear_increase_matched_5x4_v1'
OUT = SUITE/'trace_comparison'
SOURCES = {}
RAW = []
BLUE, ORANGE, INK, GRAY, GRID = '#376CB1','#C87524','#202B38','#647181','#CDD5DF'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p, expected=None):
    data = Path(p).read_bytes(); h = hashlib.sha256(data).hexdigest()
    if expected is not None: assert h == expected, p
    SOURCES[str(p)] = h
    return data.decode()
def js(p, expected=None): return json.loads(read(p,expected))
def rows(p): return list(csv.DictReader(read(p).splitlines()))
def writecsv(name, rr):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)

def audit(tracepath,timingpath,uid,expected):
    tr,tm=js(tracepath),js(timingpath)
    assert tr['total_steps']==50 and len(tr['decisions'])==len(tm['calls'])==100
    assert tm['status']=='success'
    both=[]
    for offset,(branch,cfgbranch) in enumerate([('cond','condition'),('uncond','uncondition')]):
        ds=[d for d in tr['decisions'] if d['branch']==branch]
        assert [d['step_index'] for d in ds]==list(range(50))
        values=[]
        for step,d in enumerate(ds):
            assert d['action'] in ('reuse','recompute') and d['action']==d['execution']
            skip=int(d['action']=='reuse');values.append(skip)
            call=tm['calls'][2*step+offset]
            assert call['cfg_branch']==cfgbranch and call['blocks_executed']==(0 if skip else 30)
            RAW.append(dict(trace_id=uid,branch=branch,step=step+1,skip=skip,blocks_executed=call['blocks_executed']))
        both.append(values)
    assert both[0]==both[1]
    p=''.join(map(str,both[0]));assert p==expected
    return dict(skip_path=p,skip_steps=p.count('1'),skip_last25=p[25:].count('1'),
        longest_last25=max(map(len,p[25:].split('0'))))

def draw(rr,name,title,subtitle,footnote):
    rr=sorted(rr,key=lambda r:(r['skip_steps'],r['family'],r['level'],r['sample_id']))
    n=len(rr); fig=plt.figure(figsize=(20,2.5+n*.285))
    ax=fig.add_axes([.225,.105,.535,.72]);mt=fig.add_axes([.775,.105,.212,.72])
    fig.text(.025,.967,title,fontsize=22,weight='bold',va='top')
    fig.text(.025,.922,subtitle,fontsize=11.5,color=GRAY,va='top')
    fig.legend(handles=[Patch(facecolor=BLUE,label='SEA7 / Dynamics128：重算'),
        Patch(facecolor=ORANGE,label='Increase：重算'),Patch(facecolor='white',edgecolor=GRID,label='白色：skip / 复用')],
        loc='upper left',bbox_to_anchor=(.02,.889),frameon=False,ncol=3,fontsize=11)
    matrix=np.array([[0 if s=='1' else (2 if r['family']=='Increase' else 1) for s in r['skip_path']] for r in rr])
    ax.pcolormesh(np.arange(51),np.arange(n+1),matrix,cmap=ListedColormap(['white',BLUE,ORANGE]),
        vmin=0,vmax=2,edgecolors=GRID,linewidth=.42)
    ax.set(xlim=(0,50),ylim=(n,0))
    ticks=[1,5,10,15,20,25,30,35,40,45,50]
    ax.set_xticks([t-.5 for t in ticks],ticks);ax.tick_params(length=0,pad=6)
    ax.set_yticks(np.arange(n)+.5,[r['label']+'  |  '+r['sample_id'][-3:]+(' *' if r['sample_id'].endswith('155') else '') for r in rr],fontsize=10.5)
    ax.set_xlabel('采样步（实际执行顺序，1 → 50）；虚线右侧为后25步',fontsize=11,labelpad=10)
    ax.axvline(25,color=INK,ls='--',lw=1.1)
    ax.spines[:].set_visible(False)
    mt.set(xlim=(0,1),ylim=(n,0));mt.axis('off')
    columns=[(.07,'skip','skip_steps'),(.28,'后25','skip_last25'),(.50,'最长¹','longest_last25'),(.73,'PSNR','psnr'),(.95,'加速','speedup')]
    for x,label,key in columns: mt.text(x,-.85,label,ha='center',fontsize=10.5,weight='bold')
    for i,r in enumerate(rr):
        for x,label,key in columns:
            value=f"{r[key]:.2f}" if key=='psnr' else f"{r[key]:.2f}×" if key=='speedup' else str(r[key])
            mt.text(x,i+.5,value,ha='center',va='center',fontsize=10.5)
        if i and (r['skip_steps'],r['family'],r['level']) != (rr[i-1]['skip_steps'],rr[i-1]['family'],rr[i-1]['level']):
            heavy=r['skip_steps']!=rr[i-1]['skip_steps']
            ax.axhline(i,color=INK if heavy else GRAY,lw=1.1 if heavy else .65)
            mt.axhline(i,color=GRID,lw=.6)
    fig.text(.025,.048,footnote,fontsize=10.5,color=GRAY)
    fig.text(.025,.023,'¹ 后25步内最长连续 skip；PSNR单位dB。按实际skip数升序；双CFG动作一致并核对实际blocks。全部为既有实测，seed=42。',fontsize=10.5,color=GRAY)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();clipped=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:clipped.append(t.get_text())
    assert not clipped,clipped
    for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=180,facecolor='white')
    plt.close(fig)
    return [dict(figure=name,display_rank=i+1,trace_id=r['trace_id']) for i,r in enumerate(rr)]

def main():
    config=js(SUITE/'config.json');done=js(SUITE/'COMPLETE.json')
    js(SUITE/'VALIDATION.json',done['validation_sha256']);read(SUITE/'results.csv',done['results_sha256'])
    oldval=js(PREV/'VALIDATION.json')
    for p,h in oldval['source_sha256'].items():read(p,h)
    old=[r for r in rows(PREV/'trace_summary_sorted.csv') if r['method']=='increase']
    prompts={p['sample_id']:p['prompt'] for p in config['prompts']}
    incprompts={r['sample_id']:r['prompt_en'] for r in old}
    common=sorted(prompts.keys() & incprompts.keys());assert common==['vbench200_155']
    chosen=sorted(common+sorted(prompts.keys()-set(common))[:3]);assert len(chosen)==4
    for p in common:assert prompts[p]==incprompts[p]
    groupmap={g['name']:g for g in config['groups']}
    rr=[]
    for r in rows(SUITE/'per_video.csv'):
        if r['sample_id'] not in chosen:continue
        g=groupmap[r['group']];sid=r['sample_id'];k=int(r['k']);uid=f"{r['group']}_K{k}_{sid}"
        path=SUITE/f"evaluation/candidates/{r['group']}/K{k}/{sid}";seal=js(path/'COMPLETE.json')
        for filename in ['trace.json','timing.json','generation.json']:
            read(path/filename,seal['files'][filename])
        generation=js(path/'generation.json');assert generation['protocol']==config['protocol']
        assert seal['identity']['job']['sample_id']==sid and seal['identity']['job']['skip_budget']==k
        selected=js(Path(g['analysis'])/'checkpoint_selection.json');epoch=selected['checkpoint_epoch']
        assert seal['identity']['job']['checkpoint']['sha256']==selected['checkpoint_sha256']
        family='SEA7' if g['feature']=='sea7' else 'Dynamics128'
        parsed=audit(path/'trace.json',path/'timing.json',uid,r['skip_path']);assert parsed['skip_steps']==k
        rr.append(dict(trace_id=uid,family=family,level=int(g['level'][1:]),epoch=epoch,k=k,
            label=f"{family} {g['level'].upper()} e{epoch} / K{k}",sample_id=sid,prompt=prompts[sid],
            psnr=float(r['psnr_rgb_db']),speedup=float(r['baseline_seconds'])/float(r['generate_seconds']),
            source_trace=str(path/'trace.json'),**parsed))
    assert len(rr)==96
    reference_hash=None
    for r in old:
        p=Path(r['source_trace']);folder=p.parent.parent;sid=r['sample_id'];level=int(r['level'])
        uid=f'increase_{level}_{sid}'
        parsed=audit(p,folder/'timings'/p.name,uid,r['skip_path'])
        if sid in common:
            q=next(x for x in rows(folder/'quality/per_video.csv') if x['video_id']==sid)
            native=Path(config['reference'])/'baselines'/sid/'video.mp4'
            assert sha(native)==q['reference_sha256'];SOURCES[str(native)]=sha(native)
            reference_hash=q['reference_sha256']
        rr.append(dict(trace_id=uid,family='Increase',level=level,epoch='',k='',label=r['label'],sample_id=sid,
            prompt=r['prompt_en'],psnr=float(r['psnr_rgb_db']),speedup=float(r['latency_speedup']),source_trace=str(p),**parsed))
    assert len(rr)==116 and len(RAW)==11600 and len({r['trace_id'] for r in rr})==116
    OUT.mkdir(exist_ok=True)
    (OUT/'README.md').write_text(Path(__file__).with_name('README.md').read_text())
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'svg.fonttype':'path','font.size':11,'text.color':INK,'xtick.color':GRAY,'ytick.color':INK})
    paired=[r for r in rr if r['sample_id'] in common];assert len(paired)==29
    order=draw(paired,'matched_155','同一 prompt：八组训练策略与手工 Increase',
        'Prompt 155：A panda cooking in the kitchen · 8组 × 3档 + Increase 5档 = 29条trace',
        '全部为同prompt、同seed、同GPU及相同native参考视频；Increase的实际skip数不同，不能直接视为等速画质比较。')
    for k,levels in [(23,[1]),(29,[2,3]),(35,[4,5])]:
        subset=[r for r in rr if r['k']==k or (r['family']=='Increase' and r['level'] in levels)]
        order+=draw(subset,f'K{k}_four_traces',f'K{k}：每个训练组4条trace，与相邻skip数量的 Increase 对比',
            '训练组固定008 / 023 / 024 / 155；Increase沿用040 / 041 / 144 / 155 · 星号 * 标记唯一公共prompt 155',
            '每个训练组都使用相同4条prompt，未按结果挑选；Increase其余3条是不同prompt，仅比较路径形状，不作配对画质结论。')
    writecsv('traces.csv',rr);writecsv('figure_rows.csv',order);writecsv('branch_steps.csv',RAW)
    writecsv('step_actions.csv',[dict(trace_id=r['trace_id'],step=i+1,skip=int(v)) for r in rr for i,v in enumerate(r['skip_path'])])
    writecsv('prompts.csv',[dict(sample_id=s,prompt=prompts.get(s,incprompts.get(s)),in_training_selection=s in chosen,in_increase=s in incprompts) for s in sorted(set(chosen)|incprompts.keys())])
    SOURCES[str(Path(__file__))]=sha(Path(__file__))
    assert all(sha(p)==h for p,h in SOURCES.items())
    (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',unique_traces=116,training_traces=96,increase_traces=20,
        display_cells_unique=5800,cfg_calls=11600,matched_prompt=common,training_selected_prompts=chosen,
        selection='common prompt then lowest three other IDs, outcome-blind',matched_native_sha256=reference_hash,
        source_sha256=SOURCES,figure_sha256={p.name:sha(p) for ext in ['png','svg'] for p in OUT.glob('*.'+ext)},
        visual_qa='text bounds checked; image inspection pending'),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='pass',traces=116,output=str(OUT)),ensure_ascii=False))

if __name__=='__main__':main()
