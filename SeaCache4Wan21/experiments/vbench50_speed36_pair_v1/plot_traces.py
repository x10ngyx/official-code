"""Lossless paired50 SeaCache/e391 nominal3.6x action matrices; no GPU work."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
import numpy as np

ROOT=Path('/mnt/hdd/xiongyuxiang/tmp/exp/wan21_vbench50_speed36_seacache_dynamics128_e391_v1')
OUT=ROOT/'analysis/trace_comparison_50'
SOURCES={}
METHODS=('seacache','dynamics128')
LABELS={'seacache':'SeaCache','dynamics128':'e391'}
COLORS={'seacache':'#2563A6','dynamics128':'#C87524'}


def read(path):
    data=path.read_bytes(); SOURCES[str(path)]=hashlib.sha256(data).hexdigest()
    return data.decode()


def js(path): return json.loads(read(path))


def csv_rows(path): return list(csv.DictReader(read(path).splitlines()))


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def longest(values):
    best=run=0
    for v in values:
        run=run+1 if v else 0; best=max(best,run)
    return best


def main():
    cfg=js(ROOT/'config.json'); done=js(ROOT/'COMPLETE.json')
    results={r['method']:r for r in csv_rows(ROOT/'analysis/results.csv')}
    assert SOURCES[str(ROOT/'analysis/results.csv')]==done['results_sha256']
    assert done['status']=='complete' and cfg['target']==3.6 and cfg['k']==38
    prompts=[json.loads(s) for s in read(Path(cfg['previous'])/'prompts/selected.jsonl').splitlines()]
    ids=[r['sample_id'] for r in prompts]; assert ids==sorted(set(ids)) and len(ids)==50
    details=csv_rows(ROOT/'analysis/per_video.csv')
    assert len(details)==len({(r['method'],r['sample_id']) for r in details})==100
    quality={(r['method'],r['sample_id']):r for r in details}
    paths={}; rows=[]; raw=[]; comparison=[]
    for prompt in prompts:
        sid=prompt['sample_id']
        for method in METHODS:
            q=quality[method,sid]; gpu=q['gpu']; folder=ROOT/'shards'/f'gpu{gpu}'/method
            assert sid in cfg['shard_ids'][gpu]
            run=js(folder/'run.json'); assert run['gpu_uuid']==cfg['gpu_uuids'][gpu] and run['protocol']==cfg['protocol']
            if method=='seacache': assert run['threshold']==cfg['threshold']
            else: assert run['policy_sha256']==cfg['checkpoint_sha256'] and run['skip_budget']==38
            trace=js(folder/'traces'/f'{sid}.json'); timing=js(folder/'timings'/f'{sid}.json')
            assert trace['total_steps']==50 and len(trace['decisions'])==len(timing['calls'])==100
            assert timing['status']=='success'
            assert abs(timing['pipeline_generate_wall_seconds']-float(q['candidate_seconds']))<1e-9
            saved=next(r for r in csv_rows(folder/'quality/per_video.csv') if r['video_id']==sid)
            assert abs(float(saved['psnr_rgb_db_mean'])-float(q['psnr_rgb_db']))<1e-10
            branches=[]
            for offset,(branch,tbranch) in enumerate((('cond','condition'),('uncond','uncondition'))):
                decisions=[d for d in trace['decisions'] if d['branch']==branch]
                assert [d['step_index'] for d in decisions]==list(range(50))
                assert all(d['action'] in ('reuse','recompute') for d in decisions)
                values=[int(d['action']=='reuse') for d in decisions]
                assert [i for i,v in enumerate(values) if v]==trace['per_branch'][branch]['reuse_path']
                assert [i for i,v in enumerate(values) if not v]==trace['per_branch'][branch]['recompute_path']
                assert sum(values)==trace['per_branch'][branch]['reuse']
                if method=='dynamics128': assert sum(values)==38
                for step,(v,d) in enumerate(zip(values,decisions)):
                    call=timing['calls'][2*step+offset]
                    assert call['cfg_branch']==tbranch and call['blocks_executed']==(0 if v else 30)
                    raw.append(dict(sample_id=sid,prompt_en=prompt['prompt_en'],method=method,branch=branch,step=step,
                        action=d['action'],reason=d.get('reason',''),blocks_executed=call['blocks_executed'],
                        psnr_rgb_db=q['psnr_rgb_db'],source_trace=str(folder/'traces'/f'{sid}.json')))
                branches.append(values)
            assert branches[0]==branches[1], f'CFG mismatch: {method} {sid}'
            values=branches[0]; paths[method,sid]=values
            rows.append(dict(sample_id=sid,prompt_en=prompt['prompt_en'],method=method,psnr_rgb_db=q['psnr_rgb_db'],
                reuse_steps=sum(values),recompute_steps=50-sum(values),first_reuse=values.index(1),
                longest_reuse=longest(values),trace_0_to_49=''.join('R' if v else 'C' for v in values)))
        a,b=(paths[m,sid] for m in METHODS)
        comparison.append(dict(sample_id=sid,prompt_en=prompt['prompt_en'],disagree_steps=sum(x!=y for x,y in zip(a,b))))
    assert len(raw)==10000
    for method in METHODS:
        qs=[q for (m,sid),q in quality.items() if m==method]
        assert abs(statistics.mean(float(q['psnr_rgb_db']) for q in qs)-float(results[method]['psnr_rgb_db']))<1e-10
        assert abs(sum(float(q['baseline_seconds']) for q in qs)/sum(float(q['candidate_seconds']) for q in qs)-float(results[method]['latency_speedup']))<1e-10
    OUT.mkdir(exist_ok=True)
    write_csv('branch_step_traces.csv',raw); write_csv('prompt_method_traces.csv',rows); write_csv('paired_differences.csv',comparison)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'svg.fonttype':'path','text.color':'#20252B'})
    for start,stop,name in ((0,25,'trace_01_25'),(25,50,'trace_26_50'),(0,50,'trace_all50')):
        count=stop-start; height=3+count*.59
        fig,ax=plt.subplots(figsize=(17,height))
        fig.subplots_adjust(left=.16,right=.91,top=1-1.85/height,bottom=.95/height)
        yticks=[]; ylabels=[]
        for index,sid in enumerate(ids[start:stop]):
            for offset,method in enumerate(METHODS):
                y=2.25*index+offset; values=paths[method,sid]
                ax.pcolormesh(np.arange(51)-.5,[y-.5,y+.5],np.array([values]),vmin=0,vmax=1,
                    cmap=ListedColormap(['#FFFFFF',COLORS[method]]),edgecolors='#D3D7DD',linewidth=.4)
                yticks.append(y); ylabels.append(f'{sid[-3:]}  {LABELS[method]}')
                q=quality[method,sid]
                ax.text(50.1,y,f"{float(q['psnr_rgb_db']):.2f}",va='center',fontsize=9,clip_on=False)
        ax.set(xlim=(-.5,49.5),ylim=(2.25*(count-1)+1.5,-.5))
        ax.set_xticks(range(50)); ax.set_yticks(yticks,ylabels)
        ax.tick_params(axis='x',top=True,labeltop=True,bottom=True,labelbottom=True,length=0,labelsize=8,pad=5)
        ax.tick_params(axis='y',length=0,labelsize=9,pad=8)
        ax.set_xlabel('采样 step（0–49）',fontsize=11,labelpad=9)
        ax.text(50.1,-1.2,'PSNR\n(dB)',va='bottom',fontsize=10,clip_on=False)
        fig.text(.16,1-.28/height,'名义 3.6×：SeaCache4Wan21 / Dynamics128 e391 的逐步 trace',fontsize=17,va='top')
        fig.text(.16,1-.70/height,'SeaCache：threshold 0.592402，实测 3.6466×  |  e391：K38，实测 3.6339×',fontsize=11,va='top')
        fig.text(.16,1-1.03/height,f'同 50 条 prompt · seed 42 · 当前第 {start+1}–{stop} 条 · 每对上行为 SeaCache，下行为 e391',fontsize=10,va='top')
        fig.legend(handles=[Patch(facecolor=COLORS[m],edgecolor='#D3D7DD',label=f'{LABELS[m]} 复用') for m in METHODS]+
                   [Patch(facecolor='white',edgecolor='#A0A6AF',label='重算')],loc='upper left',bbox_to_anchor=(.155,1-1.3/height),ncol=3,frameon=False,fontsize=10)
        fig.text(.16,.12/height,'各方法 cond / uncond 动作一致，已核对实际 blocks 后合并显示。左侧为 prompt ID 尾号；PSNR 为视频指标，非逐 step 指标。',fontsize=8)
        for ext in ('png','svg'): fig.savefig(OUT/f'{name}.{ext}',dpi=170,facecolor='white')
        plt.close(fig)
    summary={}
    for method in METHODS:
        selected=[r for r in rows if r['method']==method]
        summary[method]=dict(unique_traces=len({tuple(paths[method,sid]) for sid in ids}),
            **{field+'_range':[min(r[field] for r in selected),max(r[field] for r in selected)] for field in ('reuse_steps','first_reuse','longest_reuse')})
    qa=dict(status='pass',prompts=50,methods=2,raw_branch_steps=10000,displayed_cells=5000,
        cfg_branches_identical=True,actual_blocks_match=True,summary=summary,
        disagree_steps_range=[min(r['disagree_steps'] for r in comparison),max(r['disagree_steps'] for r in comparison)],
        disagree_steps_mean=statistics.mean(r['disagree_steps'] for r in comparison),source_sha256=SOURCES)
    (OUT/'VALIDATION.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in qa.items() if k!='source_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
