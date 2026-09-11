from pathlib import Path
import importlib.util,json,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
from matplotlib.patches import Rectangle,Patch
from matplotlib.collections import PatchCollection

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit',HERE.parent/'iql_increase_traces_v1/plot.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
RUN=h.BASE/'ours21_dynamics128_e391_online_offline800_8rounds_a25_v1';EVAL=RUN/'rounds/round_004/evaluation';OUT=RUN/'analysis/r4_e391_paired_traces';h.OUT=OUT
BG={'e391':'#E5EFFB','R4':'#F9E6EF'};INK='#243345';GRID='#C6CDD5';MUTED='#596879'

def draw(rr,k,target):
    fig=plt.figure(figsize=(20,13));ax=fig.add_axes([.02,.115,.96,.725]);ax.set(xlim=(0,100),ylim=(40,0));ax.axis('off')
    fig.text(.025,.963,f'离线 e391 与在线 R4：K{k} / 名义 {target:g}×',fontsize=24,weight='bold',color=INK,va='top')
    fig.text(.025,.916,'同20个 prompt、同skip预算、同seed、同baseline；每对上行为 e391，下行为 R4（joint e11）。按prompt ID排序。',fontsize=11.5,color=MUTED)
    handles=[Patch(facecolor=BG[x],edgecolor=GRID,label=x+' 底色') for x in BG]+[Patch(facecolor=INK,label='深色块：重算'),Patch(facecolor='white',edgecolor=GRID,label='留空：skip / 复用')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.02,.898),ncol=4,frameon=False,fontsize=11)
    left=20;cell=1;cols=[(74,'总skip','skip_steps'),(79,'后25','skip_last25'),(84,'最长¹','longest_last25'),(90,'PSNR','psnr'),(97,'加速','speedup')]
    ax.text(.6,-.5,'Prompt / 模型 / 动作差异',fontsize=11,weight='bold',color=INK)
    for s in [1,5,10,15,20,25,30,35,40,45,50]:ax.text(left+s-.5,-.5,str(s),ha='center',fontsize=10,color=MUTED)
    for x,title,key in cols:ax.text(x,-.5,title,ha='center',fontsize=11,weight='bold',color=INK)
    cells=[];marks=[]
    for i,r in enumerate(rr):
        ax.add_patch(Rectangle((0,i),100,1,facecolor=BG[r['model']],edgecolor='none'))
        label=r['sample_id'][-3:]+'   '+r['model']+(f"   Δ{r['different_steps']}步" if r['model']=='R4' else '')
        ax.text(.6,i+.5,label,va='center',fontsize=11.5,color=INK)
        for j,v in enumerate(r['skip_path']):
            cells.append(Rectangle((left+j,i+.12),1,.76))
            if v=='0':marks.append(Rectangle((left+j+.055,i+.18),.89,.64))
        for x,title,key in cols:
            val=f"{r[key]:.2f}" if key=='psnr' else f"{r[key]:.2f}×" if key=='speedup' else str(r[key])
            ax.text(x,i+.5,val,ha='center',va='center',fontsize=11,color=INK)
        if i%2:ax.axhline(i+1,color='#AAB5C3',lw=.8,zorder=4)
    ax.add_collection(PatchCollection(cells,facecolors='none',edgecolors=GRID,linewidths=.35,zorder=2));ax.add_collection(PatchCollection(marks,facecolors=INK,edgecolors=INK,linewidths=.1,zorder=3))
    ax.plot([45,45],[0,40],color=INK,lw=.9,ls=(0,(3,3)),zorder=4)
    for s in [0,5,10,15,20,30,35,40,45,50]:ax.plot([left+s]*2,[0,40],lw=.35,color='#AFB9C5',zorder=3)
    online=[r for r in rr if r['model']=='R4'];diff=statistics.fmean(r['different_steps'] for r in online)
    delta=statistics.fmean(r['delta_psnr'] for r in online)
    fig.text(.025,.075,f'本档每对平均 {diff:.2f}/50 步动作不同；R4 − e391 的平均 PSNR = {delta:+.4f} dB。Δ统计不同位置数，不是额外skip数。',fontsize=11,color=MUTED)
    fig.text(.025,.048,'¹ 第26–50步内最长连续skip；后25为该窗口skip总数。PSNR单位dB；加速为逐视频baseline/candidate完整推理耗时之比。',fontsize=10.5,color=MUTED)
    fig.text(.025,.021,'全部为固定20prompt的确定性评测trace；不是在线随机采集轨迹。每对50步均保留，双CFG分支动作与实际blocks已核验。',fontsize=10.5,color=MUTED)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();clipped=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            b=t.get_window_extent(renderer)
            if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height:clipped.append(t.get_text())
    assert not clipped,clipped
    for ext in ('png','svg'):fig.savefig(OUT/f'r4_e391_K{k}_20pairs.{ext}',dpi=160,facecolor='white')
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True);done=h.js(EVAL/'COMPLETE.json')
    for name,digest in done['files'].items():h.read(EVAL/name,digest)
    metrics=h.js(EVAL/'metrics.json');prompts={r['sample_id']:r for r in done['identity']['prompts']};allrows=[];pairs=[]
    for condition in metrics['per_target']:
        k=condition['skip_budget'];target=condition['target_speedup'];label=f'target_{target:g}x';rr=[]
        maps={method:{r['sample_id']:r for r in condition[method]['rows']} for method in ('offline','online')}
        assert maps['offline'].keys()==maps['online'].keys()==prompts.keys() and len(prompts)==20
        for sid in sorted(prompts):
            pair=[]
            for method,model,base,ckpt in [('offline','e391',RUN/'evaluation_reference',done['identity']['reference']),('online','R4',EVAL,done['identity']['checkpoint'])]:
                p=base/label/'candidates'/sid;seal=h.js(p/'COMPLETE.json');job=seal['identity']['job']
                assert job['checkpoint']==ckpt and job['skip_budget']==k and job['sample_id']==sid
                assert job['expected_gpu_uuid']==prompts[sid]['baseline_gpu_uuid']
                for name in ['trace.json','timing.json','measurement.json','generation.json']:h.read(p/name,seal['files'][name])
                row=maps[method][sid];q=row['quality'];assert q['candidate_sha256']==seal['files']['video.mp4']
                trace=h.js(p/'trace.json')
                if method=='online':assert trace['action_mode']=='policy_argmax'
                assert all(d['reason']=='policy_argmax' for d in trace['decisions'] if d['policy_queried'])
                path=''.join('1' if i in trace['per_branch']['cond']['reuse_path'] else '0' for i in range(50));uid=f'K{k}_{model}_{sid}'
                audited=h.audit(p/'trace.json',p/'timing.json',uid,path);assert audited['skip_steps']==k
                m=h.js(p/'measurement.json');assert abs(m['generate_seconds']-row['candidate']['generate_seconds'])<1e-9
                pair.append(dict(trace_id=uid,k=k,target=target,model=model,sample_id=sid,prompt=prompts[sid]['prompt'],psnr=float(q['psnr_rgb_db_mean']),speedup=row['baseline']['generate_seconds']/m['generate_seconds'],**audited))
            assert maps['offline'][sid]['quality']['reference_sha256']==maps['online'][sid]['quality']['reference_sha256']
            diff=sum(a!=b for a,b in zip(pair[0]['skip_path'],pair[1]['skip_path']));delta=pair[1]['psnr']-pair[0]['psnr'];assert diff%2==0
            for r in pair:r.update(different_steps=diff,delta_psnr=delta)
            pairs.append(dict(k=k,target=target,sample_id=sid,different_steps=diff,delta_psnr=delta,offline_late25=pair[0]['skip_last25'],online_late25=pair[1]['skip_last25'],offline_run25=pair[0]['longest_last25'],online_run25=pair[1]['longest_last25']))
            rr.extend(pair)
        draw(rr,k,target);allrows.extend(rr)
    assert len(allrows)==120 and len(h.RAW)==12000
    h.writecsv('traces.csv',allrows);h.writecsv('pairs.csv',pairs);h.writecsv('branch_steps.csv',h.RAW)
    h.writecsv('prompts.csv',[dict(sample_id=sid,prompt=prompts[sid]['prompt']) for sid in sorted(prompts)])
    summary=[dict(k=k,pairs=20,mean_different_steps=statistics.fmean(r['different_steps'] for r in pairs if r['k']==k),identical_pairs=sum(r['different_steps']==0 for r in pairs if r['k']==k),delta_psnr=statistics.fmean(r['delta_psnr'] for r in pairs if r['k']==k)) for k in (23,29,35)]
    sources=h.SOURCES;sources[str(Path(__file__).resolve())]=h.sha(Path(__file__).resolve())
    (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',traces=120,pairs=60,cells=6000,cfg_calls=12000,summary=summary,source_sha256=sources,text_bounds='pass',visual_qa='pending actual PNG inspection',figures_sha256={p.name:h.sha(p) for ext in ('png','svg') for p in OUT.glob('*.'+ext)}),ensure_ascii=False,indent=2)+'\n')
    (OUT/'README.md').write_text('# R4 / e391 paired traces\n\nThree PNG/SVG figures for K23/K29/K35, 20 pairs each; traces.csv contains full 120 traces and metrics, pairs.csv action differences and quality deltas, prompts.csv full prompts, branch_steps.csv all 12000 CFG calls. VALIDATION.json holds sealed source hashes. Offline references are reused original e391 results; online is R4 selected joint e11.\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'svg.fonttype':'path'});main()
