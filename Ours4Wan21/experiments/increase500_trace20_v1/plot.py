"""Twenty sealed traces: paired three-model conditions and five increase references."""
import csv
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.font_manager import FontProperties,fontManager
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT))
from ours4wan21.online_common import verified
from ours4wan21.contracts import sha256
EXP=Path('/mnt/hdd/xiongyuxiang/tmp/exp')
RUN=EXP/'ours21_increase500_iql2_v1'
INC=EXP/'ours21_increase500_v1'
REF=EXP/'ours21_dynamics128_e391_online_offline800_8rounds_a25_v1/evaluation_reference'
OUT=RUN/'analysis/trace20_comparison'
BG={'Conservative':'#DCECF8','Aggressive':'#F7E2EA','e391':'#E7EDD9','Increase':'#FFF0CA'}
ORDER={k:i for i,k in enumerate(BG)}
LABEL={'Conservative':'保守 e374','Aggressive':'激进 e358','e391':'原始 e391','Increase':'Increase'}
SOURCES={}
BRANCHES=[]


def read(p):
    p=Path(p);SOURCES[str(p)]=sha256(p)
    return json.loads(p.read_text())


def csvout(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def parse(trace_path,timing_path,tid):
    t=read(trace_path);timing=read(timing_path)
    assert timing['status']=='success'
    assert len(t['decisions'])==len(timing['calls'])==100
    paths=[]
    for branch in ('cond','uncond'):
        ds=[d for d in t['decisions'] if d['branch']==branch]
        assert [d['step_index'] for d in ds]==list(range(50))
        paths.append([int(d['action']=='reuse') for d in ds])
    assert paths[0]==paths[1] and paths[0][0]==paths[0][-1]==0
    for i,(d,c) in enumerate(zip(t['decisions'],timing['calls'])):
        assert d['step_index']==i//2 and d['branch']==('cond','uncond')[i%2]
        assert c['blocks_executed']==(0 if d['action']=='reuse' else 30)
        BRANCHES.append(dict(trace_id=tid,call_index=i,step_index=i//2,branch=d['branch'],action=d['action'],actual_blocks=c['blocks_executed']))
    path=paths[0];run=best=0
    for a in path[25:]:
        run=run+1 if a else 0;best=max(best,run)
    return dict(skip=sum(path),late25=sum(path[25:]),longest_late25=best,skip_path=''.join(map(str,path)))


def freeze_selection():
    p=OUT/'selection.json'
    if p.exists():return read(p)
    cfg=read(RUN/'config.json');jobs=[]
    for f in sorted((RUN/'jobs').glob('gpu?.json')):
        jobs.extend(read(f))
    completed=[j for j in jobs if (Path(j['output'])/'COMPLETE.json').exists()]
    sets={g:{(j['skip_budget'],j['sample_id']) for j in completed if j['group']==g} for g in ('conservative','aggressive')}
    common=sets['conservative']&sets['aggressive']
    conditions=[]
    for k,count in [(23,3),(29,2)]:
        chosen=sorted([x for x in common if x[0]==k])[:count]
        assert len(chosen)==count,'insufficient sealed paired conditions'
        conditions.extend(chosen)
    manifest=INC/'manifests/increase_runnable.jsonl';SOURCES[str(manifest)]=sha256(manifest)
    allinc=[json.loads(s) for s in manifest.read_text().splitlines()]
    selected=[]
    for target in (1.5,2.,2.5,3.,3.5):
        row=min(allinc,key=lambda r:(abs(r['target_speedup']-target),r['trajectory_id']))
        assert row['trajectory_id'] not in [r['trajectory_id'] for r in selected]
        selected.append(row)
    value=dict(captured_at=datetime.now().astimezone().isoformat(),completed_candidates=len(completed),
        completed_counts={f'{g}_K{k}':sum(j['group']==g and j['skip_budget']==k for j in completed)
            for g in ('conservative','aggressive') for k in (23,29,35)},
        conditions=conditions,selected_jobs=[j for j in completed if (j['skip_budget'],j['sample_id']) in conditions],
        increase_rows=selected,selection_rule='common sealed K23 first3 and K29 first2 by prompt ID; five increase rows nearest target1.5/2/2.5/3/3.5, tie trajectory ID; no quality/path-based selection')
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return read(p)


def collect(selection):
    cfg=read(RUN/'config.json');ref=Path(cfg['reference']);result=[]
    source=PROJECT/'experiments/increase500_iql2_v1'
    sys.path.insert(0,str(source))
    from generate_worker import job_identity
    conditions=[tuple(c) for c in selection['conditions']]
    for j in selection['selected_jobs']:
        p=Path(j['output']);identity=job_identity(j,cfg)
        assert verified(p,identity)
        read(p/'COMPLETE.json');gen=read(p/'generation.json')
        chosen=read(RUN/j['group']/'SELECTED.json')
        assert j['checkpoint']['sha256']==chosen['checkpoint_sha256']
        assert gen['gpu_uuid']==j['expected_gpu_uuid']
        base=ref/'baselines'/j['sample_id'];assert verified(base)
        assert read(base/'generation.json')['gpu_uuid']==gen['gpu_uuid']
        m=read(p/'measurement.json');bm=read(base/'measurement.json')
        family='Conservative' if j['group']=='conservative' else 'Aggressive'
        tid=f"{family}_K{j['skip_budget']}_{j['sample_id']}"
        stats=parse(p/'trace.json',p/'timing.json',tid);assert stats['skip']==j['skip_budget']
        result.append(dict(trace_id=tid,family=family,sample_id=j['sample_id'],prompt=j['prompt'],
            pair_id=f"K{j['skip_budget']}_{j['sample_id']}",nominal_speed=0,
            speedup=bm['generate_seconds']/m['generate_seconds'],source_trace=str(p/'trace.json'),
            label=f"{LABEL[family]}  V{j['sample_id'][-3:]}",**stats))
    for k,sid in conditions:
        tag={23:'1.8x',29:'2.4x',35:'3x'}[k];p=REF/f'target_{tag}'/'candidates'/sid
        assert verified(p);read(p/'COMPLETE.json');gen=read(p/'generation.json')
        base=ref/'baselines'/sid;bg=read(base/'generation.json')
        assert gen['gpu_uuid']==bg['gpu_uuid']
        # Original e391 identity is retained by the source adapter.
        job=gen['identity']['job'];ckpt=job['checkpoint']
        assert '391' in Path(ckpt['path']).stem
        assert sha256(ckpt['path'])==ckpt['sha256']
        SOURCES[ckpt['path']]=ckpt['sha256']
        bm=read(base/'measurement.json');m=read(p/'measurement.json');tid=f'e391_K{k}_{sid}'
        stats=parse(p/'trace.json',p/'timing.json',tid);assert stats['skip']==k
        result.append(dict(trace_id=tid,family='e391',sample_id=sid,prompt=job['prompt'],pair_id=f'K{k}_{sid}',nominal_speed=0,
            speedup=bm['generate_seconds']/m['generate_seconds'],source_trace=str(p/'trace.json'),label=f'原始 e391  V{sid[-3:]}',**stats))
    for row in selection['increase_rows']:
        tid=row['trajectory_id'];c=read(INC/'completed'/f'{tid}.json');r=c['trajectory_row']
        assert r['policy_family']=='linear_increase_seacache_threshold'
        trace=Path(r['trace_json']);t=read(trace)
        assert t['threshold_path']==row['threshold_path']
        stats=parse(trace,trace.parent/'timing.json',tid)
        assert stats['skip']==r['actual_both_reuse_steps']
        bm=read(Path(r['baseline_video']).parent/'performance.json');m=read(trace.parent/'performance.json')
        speed=bm['pipeline_generate_wall_seconds']/m['pipeline_generate_wall_seconds']
        assert abs(speed-r['inference_latency_speedup'])<1e-8
        result.append(dict(trace_id=tid,family='Increase',sample_id=r['sample_id'],prompt=r['prompt'],pair_id='',
            nominal_speed=r['target_speedup'],speedup=speed,source_trace=str(trace),
            label=f"Increase  OV{r['sample_id'][-4:]}  ~{r['target_speedup']:.1f}x",**stats))
    for pair in {r['pair_id'] for r in result if r['pair_id']}:
        matched=[r for r in result if r['pair_id']==pair]
        assert len(matched)==3 and len({r['prompt'] for r in matched})==1
        assert len({r['skip'] for r in matched})==1
    assert len(result)==20 and Counter(r['family'] for r in result)=={g:5 for g in BG}
    assert len({r['trace_id'] for r in result})==20 and len(BRANCHES)==2000
    result.sort(key=lambda r:(r['skip'],r['sample_id'],ORDER[r['family']],r['trace_id']))
    for i,r in enumerate(result):r['rank']=i+1
    return result


def draw(rows):
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'svg.fonttype':'path','font.size':11})
    fig=plt.figure(figsize=(21,12),dpi=160)
    ax=fig.add_axes([.025,.125,.95,.65]);ax.set(xlim=(0,110),ylim=(20,0));ax.axis('off')
    ink='#243243';grid='#B3BCC5';left=32;cell=1.10
    for i,r in enumerate(rows):
        ax.add_patch(Rectangle((0,i),110,1,facecolor=BG[r['family']],edgecolor='white',lw=.8))
        ax.text(.6,i+.5,f"{i+1:02d}  {r['label']}",va='center',fontsize=11,color=ink)
        for t,a in enumerate(r['skip_path']):
            ax.add_patch(Rectangle((left+t*cell,i+.14),cell,.72,facecolor='white',edgecolor=grid,lw=.4))
            if a=='0':ax.add_patch(Rectangle((left+t*cell+.07,i+.21),cell-.14,.58,facecolor=ink,edgecolor='none'))
        for x,key in [(91,'skip'),(97,'late25'),(103,'longest_late25'),(108,'speedup')]:
            val=f"{r[key]:.2f}×" if key=='speedup' else str(r[key])
            ax.text(x,i+.5,val,ha='center',va='center',fontsize=11,color=ink)
    for t in [0,5,10,15,20,25,30,35,40,45,50]:
        ax.plot([left+t*cell]*2,[0,20],color=ink if t==25 else grid,lw=1 if t==25 else .45,ls='--' if t==25 else '-')
    for t in [1,5,10,15,20,25,30,35,40,45,50]:ax.text(left+(t-.5)*cell,-.42,str(t),ha='center',fontsize=11)
    for x,label in [(1,'来源 / prompt ID'),(91,'skip'),(97,'后25'),(103,'最长¹'),(108,'实速')]:
        ax.text(x,-.42,label,ha='left' if x==1 else 'center',weight='bold',fontsize=11)
    fig.text(.025,.957,'混合数据训练与原始 e391 / Increase：20 条 trace 对比',fontsize=23,weight='bold',color=ink)
    fig.text(.025,.918,'按实际 skip 数升序；每条完整 50 步。保守 e374、激进 e358、原始 e391、Increase 各 5 条。',fontsize=13,color=ink)
    handles=[Patch(facecolor=BG[g],edgecolor=grid,label=LABEL[g]+' 底色') for g in BG]
    handles += [Patch(facecolor=ink,label='深色块：重算'),Patch(facecolor='white',edgecolor=grid,label='白格：skip / 复用')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.019,.897),ncol=6,frameon=False,fontsize=11.5)
    fig.text(.025,.831,'三模型同 prompt / 同 K 配对；K23 取 3 对、K29 取 2 对。右侧为实际路径统计和完整推理加速。',fontsize=12,color=ink)
    fig.text(.025,.080,'¹ 后 25 步内最长连续 skip。虚线分隔前 / 后 25 步；步数显示为 1–50（原始索引 0–49）。',fontsize=11,color=ink)
    fig.text(.025,.049,'Increase 取本次 OpenVid 数据的五档近邻样本，与 VBench prompt 不同，只比较路径结构。',fontsize=11,color=ink)
    fig.text(.025,.020,'仅使用已封存结果；选样不参考质量或动作路径。当前质量评测尚未完成，本图不评价画质收益。',fontsize=11,color=ink)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            bb=text.get_window_extent(renderer)
            assert bb.x0>=0 and bb.y0>=0 and bb.x1<=fig.bbox.width and bb.y1<=fig.bbox.height,text.get_text()
    for ext in ('png','svg'):fig.savefig(OUT/f'trace20_sorted.{ext}',dpi=160,facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'README.md').write_text((HERE/'README.md').read_text())
    if not (OUT.parent/'README.md').exists():(OUT.parent/'README.md').write_text('# Analysis\n\nSource-backed figures and readouts.\n')
    selection=freeze_selection();rows=collect(selection);draw(rows)
    csvout('traces_sorted.csv',rows);csvout('branch_actions.csv',BRANCHES)
    csvout('step_actions.csv',[dict(trace_id=r['trace_id'],rank=r['rank'],family=r['family'],step=t+1,skip=int(a)) for r in rows for t,a in enumerate(r['skip_path'])])
    SOURCES[str(HERE/'plot.py')]=sha256(HERE/'plot.py')
    assert all(sha256(p)==digest for p,digest in SOURCES.items())
    (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',traces=20,step_cells=1000,cfg_calls=2000,
        source_sha256=SOURCES,counts=dict(Counter(r['family'] for r in rows)),sort='actual skip ascending; prompt ID then family',
        figure_sha256={f'trace20_sorted.{e}':sha256(OUT/f'trace20_sorted.{e}') for e in ('png','svg')},
        text_bounds='pass',visual_inspection='pending'),ensure_ascii=False,indent=2)+'\n')
    print(OUT)

if __name__=='__main__':main()
