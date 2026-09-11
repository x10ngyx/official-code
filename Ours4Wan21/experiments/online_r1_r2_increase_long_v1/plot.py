"""One long figure, 100 R1 + 100 R2 + five original Increase traces."""
import importlib.util
from pathlib import Path
import json
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.patches import Rectangle,Patch
from matplotlib.collections import PatchCollection

HELPER=Path(__file__).resolve().parents[1]/'iql_increase_traces_v1/plot.py'
spec=importlib.util.spec_from_file_location('trace_helpers',HELPER);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
RUN=h.BASE/'ours21_dynamics128_e391_online_offline800_8rounds_a25_v1'
OUT=RUN/'analysis/r1_r2_increase_long';h.OUT=OUT
BG={'R1':'#E5EFFB','R2':'#F9E6EF','Increase':'#FFF0D0'}
INK='#243345';GRID='#C6CDD5';MUTED='#566577'
ORDER={'R1':0,'R2':1,'Increase':2}


def collect():
    manifest=h.js(RUN/'manifest.json');rr=[];parents={}
    for number in (1,2):
        family=f'R{number}';root=RUN/'rounds'/f'round_{number:03d}';plan=h.js(root/'plan.json')
        assert len(plan['rows'])==100 and plan['round']==number
        expected=(dict(path=manifest['paths']['start_checkpoint'],sha256=manifest['inputs']['start_checkpoint']['sha256']) if number==1
                  else {k:v for k,v in h.js(RUN/'rounds/round_001/training/selected.json').items() if k in ('path','sha256')})
        assert plan['parent']==expected and h.sha(expected['path'])==expected['sha256'];parents[family]=expected
        h.SOURCES[expected['path']]=expected['sha256']
        qs=h.js(root/'quality/COMPLETE.json')
        for name,digest in qs['files'].items():h.read(root/'quality'/name,digest)
        quality={r['video_id']:r for r in h.rows(root/'quality/metrics/per_video.csv')}
        assert set(quality)=={r['trajectory_id'] for r in plan['rows']}
        for row in plan['rows']:
            uid=row['trajectory_id'];p=root/'collection'/uid;seal=h.js(p/'COMPLETE.json')
            for name in ('trace.json','timing.json','measurement.json','generation.json'):h.read(p/name,seal['files'][name])
            job=seal['identity']['job'];assert job['checkpoint']==plan['parent'] and job['sampling_seed']==row['sampling_seed']
            assert job['sample_id']==row['sample_id'] and job['skip_budget']==row['skip_budget'] and job['kind']=='collection'
            tr=h.js(p/'trace.json');path=''.join('1' if i in tr['per_branch']['cond']['reuse_path'] else '0' for i in range(50))
            parsed=h.audit(p/'trace.json',p/'timing.json',uid,path);assert parsed['skip_steps']==row['skip_budget']
            q=quality[uid];assert q['candidate_sha256']==seal['files']['video.mp4']
            assert (int(q['frames']),int(q['width']),int(q['height']))==(81,832,480)
            base=Path(manifest['paths']['training_bundle'])/'baselines'/row['sample_id'];bs=h.js(base/'COMPLETE.json')
            assert q['reference_sha256']==bs['files']['video.mp4']
            h.read(base/'measurement.json',bs['files']['measurement.json'])
            seconds=h.js(p/'measurement.json')['generate_seconds'];base_seconds=h.js(base/'measurement.json')['generate_seconds']
            rr.append(dict(trace_id=uid,family=family,sample_id=row['sample_id'],prompt=row['prompt'],sampling_seed=row['sampling_seed'],
                parent_checkpoint_sha256=plan['parent']['sha256'],label=f"{family} #{uid[-4:]}   OV{row['sample_id'][-4:]}",
                psnr=float(q['psnr_rgb_db_mean']),speedup=base_seconds/seconds,source_trace=str(p/'trace.json'),**parsed))
    previous=h.js(h.PREV/'VALIDATION.json')
    for p,digest in previous['source_sha256'].items():h.read(p,digest)
    old=[r for r in h.rows(h.PREV/'trace_summary_sorted.csv') if r['method']=='increase' and r['sample_id']=='vbench200_155'];assert len(old)==5
    for row in old:
        p=Path(row['source_trace']);uid='increase_'+row['level']+'_v155'
        parsed=h.audit(p,p.parent.parent/'timings'/p.name,uid,row['skip_path'])
        rr.append(dict(trace_id=uid,family='Increase',sample_id=row['sample_id'],prompt=row['prompt_en'],sampling_seed='',parent_checkpoint_sha256='',
            label=f"Increase {row['level']}   V155",psnr=float(row['psnr_rgb_db']),speedup=float(row['latency_speedup']),source_trace=str(p),**parsed))
    rr.sort(key=lambda r:(r['skip_steps'],ORDER[r['family']],r['trace_id']))
    assert len(rr)==205 and len({r['trace_id'] for r in rr})==205 and len(h.RAW)==20500
    for i,r in enumerate(rr):r['display_rank']=i+1
    return rr,parents


def draw(rows):
    groups=sorted({r['skip_steps'] for r in rows});n=len(rows);header_height=.76
    total_units=n+header_height*len(groups)
    row_inches=.225;body_inches=total_units*row_inches;top_inches=2.45;bottom_inches=.9;height=body_inches+top_inches+bottom_inches
    fig=plt.figure(figsize=(20,height),dpi=150)
    ax=fig.add_axes([.02,bottom_inches/height,.96,body_inches/height]);ax.set(xlim=(0,100),ylim=(total_units,0));ax.axis('off')
    text=lambda y,t,**kw:fig.text(.025,1-y/height,t,va='top',**kw)
    text(.22,'在线 R1 / R2 与 Increase：按实际 skip 数排序',fontsize=22,weight='bold',color=INK)
    text(.69,'205条完整trace：R1 100条 + R2 100条 + Increase五档各1条；每条50步，实际skip数升序。',fontsize=11.5,color=MUTED)
    text(1.02,'R1：起始Dynamics128 e391采样；R2：R1选中的joint e12采样。同skip数内依次R1、R2、Increase。',fontsize=11,color=MUTED)
    handles=[Patch(facecolor=BG[f],edgecolor=GRID,label=f+' 底色') for f in BG]
    handles += [Patch(facecolor=INK,label='深色块：重算'),Patch(facecolor='white',edgecolor=GRID,label='留空：skip / 复用')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.02,1-1.35/height),frameon=False,ncol=5,fontsize=11)
    text(1.94,'左：来源 / 轨迹编号 / prompt ID      中：步1 → 50（虚线为后25步边界）      右：路径统计与原始质量 / 速度',fontsize=10.4,color=MUTED)
    left=24.;cell=.99;right=left+50*cell
    columns=[(76.0,'skip','skip_steps'),(81.0,'后25','skip_last25'),(86.0,'最长¹','longest_last25'),(91.5,'PSNR','psnr'),(97.3,'加速','speedup')]
    fills=[];marks=[];current=0.;matrix_count=0;locations=[]
    for k in groups:
        subset=[r for r in rows if r['skip_steps']==k]
        ax.add_patch(Rectangle((0,current),100,header_height,facecolor='#F2F4F7',edgecolor='none'))
        ax.text(.6,current+header_height/2,f'实际 skip = {k}   ·   {len(subset)}条',va='center',fontsize=10.5,weight='bold',color=INK)
        for step in (1,5,10,15,20,25,30,35,40,45,50):ax.text(left+(step-.5)*cell,current+header_height/2,str(step),ha='center',va='center',fontsize=9,color=MUTED)
        for x,title,key in columns:ax.text(x,current+header_height/2,title,ha='center',va='center',fontsize=9.4,weight='bold',color=INK)
        current+=header_height;start=current
        for r in subset:
            ax.add_patch(Rectangle((0,current),100,1,facecolor=BG[r['family']],edgecolor='none'))
            ax.text(.6,current+.5,f"{r['display_rank']:03d}   {r['label']}",va='center',fontsize=10.5,color=INK)
            for step,value in enumerate(r['skip_path']):
                fills.append(Rectangle((left+step*cell,current+.12),cell,.76))
                if value=='0':marks.append(Rectangle((left+step*cell+.055,current+.19),cell-.11,.62))
                matrix_count+=1
            for x,title,key in columns:
                value=f"{r[key]:.2f}" if key=='psnr' else f"{r[key]:.2f}×" if key=='speedup' else str(r[key])
                ax.text(x,current+.5,value,ha='center',va='center',fontsize=10.3,color=INK)
            locations.append(dict(rank=r['display_rank'],trace_id=r['trace_id'],family=r['family'],y_unit=current+.5));current+=1
        for step in (0,5,10,15,20,30,35,40,45,50):ax.plot([left+step*cell]*2,[start,current],color='#AFB9C5',lw=.35,zorder=3)
        ax.plot([left+25*cell]*2,[start,current],color=INK,ls=(0,(3,3)),lw=.8,zorder=4)
        ax.axhline(current,color='#AEB9C8',lw=.65)
    ax.add_collection(PatchCollection(fills,facecolors='none',edgecolors=GRID,linewidths=.32,zorder=2))
    ax.add_collection(PatchCollection(marks,facecolors=INK,edgecolors=INK,linewidths=.2,zorder=3))
    assert matrix_count==10250
    fig.text(.025,.63/height,'¹ 后25步内最长连续skip；PSNR单位dB。R1/R2为该轮训练前的采集轨迹，不是该轮训练后模型的评测。',fontsize=10.5,color=MUTED,va='top')
    fig.text(.025,.34/height,'R1/R2来自OpenVid训练prompt；Increase沿用VBench155。这里只对照路径结构，不作跨prompt配对画质结论。',fontsize=10.5,color=MUTED,va='top')
    fig.canvas.draw();renderer=fig.canvas.get_renderer();clipped=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height:clipped.append(t.get_text())
    assert not clipped,clipped
    for ext in ('png','svg'):fig.savefig(OUT/f'r1_r2_increase_sorted_long.{ext}',dpi=150,facecolor='white')
    dims=[int(fig.bbox.width),int(fig.bbox.height)];plt.close(fig)
    return dict(pixel_dimensions=dims,data_rows=n,matrix_cells=matrix_count,sort_groups=groups,palette=BG,
        row_locations=locations,body_units=total_units,body_inches=body_inches,top_inches=top_inches,bottom_inches=bottom_inches,height_inches=height,text_bounds='pass')


def main():
    rows,parents=collect();OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'README.md').write_text(Path(__file__).with_name('README.md').read_text())
    parent_readme=OUT.parent/'README.md'
    if not parent_readme.exists():parent_readme.write_text('# Online analysis\n')
    original=parent_readme.read_text()
    if 'r1_r2_increase_long/' not in original:parent_readme.write_text(original+'\nr1_r2_increase_long/: all R1/R2 collection traces and the same five Increase references in one sorted long figure.\n')
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'svg.fonttype':'path','font.size':10.5})
    layout=draw(rows)
    h.writecsv('traces_sorted.csv',rows);h.writecsv('branch_steps.csv',h.RAW)
    h.writecsv('step_actions.csv',[dict(trace_id=r['trace_id'],family=r['family'],step=i+1,skip=int(v)) for r in rows for i,v in enumerate(r['skip_path'])])
    h.writecsv('prompts.csv',[dict(trace_id=r['trace_id'],family=r['family'],sample_id=r['sample_id'],prompt=r['prompt']) for r in rows])
    for p in (Path(__file__).resolve(),HELPER.resolve()):h.SOURCES[str(p)]=h.sha(p)
    assert all(h.sha(p)==digest for p,digest in h.SOURCES.items())
    (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',counts=dict(Counter(r['family'] for r in rows)),unique_traces=205,cfg_calls=20500,cells=10250,
        parent_checkpoints=parents,sort='skip_steps ascending, R1/R2/Increase, trajectory_id',source_sha256=h.SOURCES,
        figures_sha256={p.name:h.sha(p) for ext in ('png','svg') for p in OUT.glob('*.'+ext)},layout=layout,
        visual_qa='text bounds pass; actual PNG inspection pending'),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='pass',traces=205,cells=10250,dimensions=layout['pixel_dimensions'],output=str(OUT)),ensure_ascii=False))

if __name__=='__main__':main()
