"""Build a canonical portable report and a stdlib-executed audit notebook."""
from pathlib import Path
import csv,json,datetime,io,contextlib,sqlite3

OUT=Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_dynamics128_e391_online_offline800_8rounds_a25_v1/analysis/r1_r4_policy_diagnostics')
def rows(name):return list(csv.DictReader((OUT/name).open()))
def dump(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    s=json.loads((OUT/'SUMMARY.json').read_text());valid=json.loads((OUT/'VALIDATION.json').read_text())
    summary=s['rounds'];standard=s['standardized'];title='R1–R4 在线采集策略变化'
    source=dict(id='collection_audit',label='四轮封存采集、计时与质量审计',path='SUMMARY.json',query=dict(
        engine='Python, read-only sealed experiment artifacts',language='Python',
        description='analyze.py逐条核对400份collection trace/timing/measurement及四轮quality/metrics/per_video.csv，计算50步路径特征和同K标准化；40000次CFG调用与实际blocks一致。',
        tables_used=['trajectories.csv','round_summary.csv','by_k.csv','standardized.csv','budget_bands.csv','step_profile.csv','run_histogram.csv'],
        filters=['R1–R4 completed collection only; 100 trajectories per round','50 denoising steps, shared CFG decisions; skip=1, recompute=0','No evaluation20 candidates; no new inference or quality computation'],
        metric_definitions=['speedup=mean(baseline generate_seconds / candidate generate_seconds)','PSNR=mean(VideoMetrics per-video psnr_rgb_db_mean), dB','early=sum(skip[0:25]); late=sum(skip[25:50]); K=early+late','longest=max consecutive skip run on all 50 steps; late_longest clips runs at step 25 boundary','late_share=sum(late)/sum(K)','standardized=sum_K(pooled weight_K * per-round mean at exact K), K18–37, 397/400 support; no prompt control']))
    sources=[source];blocks=[]
    def md(id,body):blocks.append(dict(id=id,type='markdown',body=body,**({} if id=='title' else dict(sourceId='collection_audit'))))
    def table_block(id):blocks.append(dict(id=id+'_block',type='table',tableId=id))
    def chart_block(id):blocks.append(dict(id=id+'_block',type='chart',chartId=id))
    md('title','# '+title)
    md('summary','## 主要结论\n\n**四轮速度基本不变，采集质量没有持续提升；路径先向后段集中，再回到略偏前、较分散的复用方式。** R4采集、质量和训练已完成，R4均值为2.4905×、23.5524dB，相对R1为+0.0029×、−0.0955dB，相对R2为−0.0079×、−0.8100dB。\n\nR2后25步skip最多，R3/R4回落。R4相对R1平均把约0.44次skip从后半段移出，最长连续skip均值从7.66降至7.11步。该变化主要来自低预算K17–23，而高预算仍保持较强后段复用。以上是随机采集集合的描述统计，尚不能确认策略质量改善或退化。')
    md('scope','## 比较的是四轮采集集合，每轮100条\n\n加速比取每条baseline完整generate耗时除以候选完整generate耗时，再做算术平均；PSNR取每视频原始PSNR均值的算术平均，单位dB。前段为第1–25步，后段为第26–50步，首尾强制重算保留。\n\nR1由原e391采样，R2由R1选点e12采样，R3由R2选点e20采样，R4由R3选点e16采样。**R4采集指标不是R4训练后e11模型的固定测试集评测。** 后者正在单独进行。')
    table_block('means')
    md('allocation','## R2最偏后段，R3/R4前移\n\n平均总skip数仅29.92–29.98，速度均值仅2.4876–2.4984×，因此主要变化发生在预算分配位置。后25步skip从R1的19.06升至R2的19.44，再降至R3的18.75和R4的18.62。后段占全部skip的比例为63.68%/64.97%/62.54%/62.23%。四轮分别有100/99/96/98条轨迹的后段skip多于前段，整体仍明显偏后。\n\n下图比较同长度的两个窗口，柱高为每轨迹skip次数；并非后段所有步都会连续复用。R4相对R2前段+0.82、后段−0.82，平均总K恰好相同。')
    chart_block('halves')
    md('profile_text','## 变化表现为中前段多复用、后段少复用\n\n逐步曲线显示每轮100条轨迹中该步实际skip的比例，包含Exact-K强制决策。R4相对R1在11–20、21–30步平均分别多0.24、0.18次skip，在31–40、41–50步分别少0.30、0.17次。skip平均发生位置从30.20前移至29.72；R2则为30.54。\n\n这是实际路径分布，不能将曲线直接当作actor在相同状态上的skip概率。')
    chart_block('profile')
    md('runs_text','## 连续复用略缩短，未出现逐轮延长\n\n每条轨迹先求最长连续skip，再在轮内平均：7.66→7.79→7.16→7.11步。后25步内最长连续skip的均值为7.60→7.78→7.13→7.08步；前段对应3.09→3.07→3.24→3.23步。最长段通常位于后半程。\n\n达到至少10步连续skip的轨迹占比为29%/33%/26%/24%；达到14步的数量为0/2/3/0条。每条轨迹平均skip段数从R1的10.43变为R4的10.76，每段平均长度从3.020降至2.950步。下图按每轨迹最长段分类，分母始终100条，避免把长轨迹的段数当作更多独立样本。')
    chart_block('run_hist')
    table_block('runs')
    md('bands_text','## 前移主要集中在K17–23\n\n三个预算区间每轮恰好都是17/29/54条，所以区间占比变化为零。R4相对R1的后段skip均值变化，在K17–23为−2.118步、K24–30为+0.034步、K31–37为−0.167步；对全体均值贡献分别−0.36、+0.01、−0.09，精确合计−0.44步，低预算贡献约82%。\n\n低预算首次skip平均从第18.41步提前至14.82步，最长连续skip从4.65降至3.29步；高预算后段skip仍为20.50→20.33，最长连续段9.56→8.83步。因而不能概括成所有预算都发生大幅前移。区间内部K和prompt仍有差异，下一节进一步统一精确K分布。')
    table_block('bands')
    md('robustness','## 统一精确K后，位置与连续长度方向仍在\n\n将各轮按相同K重新加权，权重来自四轮合并的预算分布。四轮共有K18–37，覆盖397/400条；R3没有K17，因此各轮共3条K17不参与此敏感性计算。调整后后段skip为19.125/19.511/18.739/18.719，最长连续skip为7.695/7.827/7.148/7.142。R4相对R1分别−0.406和−0.554步，路径方向未消失。\n\n但质量结论很弱：统一K后R4相对R1的PSNR差为+0.132dB，与未调整的−0.096dB符号相反；相对R2仍约−0.619dB。低K少量轨迹和prompt难度会影响均值，不能把采集PSNR小幅差异解释为确定的学习收益。')
    table_block('standard')
    md('limits','## 路径变化已观察到，原因与质量收益仍未确定\n\n这不是同prompt同K的配对实验；每轮随机抽样、策略采样seed不同，且Exact-K在预算受限时强制执行动作。相同K重加权只控制预算组合，不控制prompt难度，也不能消除动作采样波动。各K单轮仅少量样本，本报告不作统计显著性或因果声明。\n\n后段窗口缩为最后20或15步时，R4相对R1的skip数量差仍分别为−0.47/−0.28步，支持总体位置前移的描述，但幅度不大。不能仅因后段更集中就判定质量更好，也不能因R4平均PSNR低于R2就确认策略退化。')
    md('next','## 以既定固定测试集判断训练收益\n\n接下来按已有计划读取R4/R8的固定20prompt、K23/K29/K35结果，与复用的离线e391对照比较PSNR、SSIM、LPIPS和实际速度。核心问题是：相同prompt和预算下，路径分配变化是否带来稳定的质量收益。R4正式评测正在执行，本次不新增测试或改动训练。')
    means=[dict(round='R'+str(r['round']),n=r['n'],speedup=r['speedup'],psnr=r['psnr'],k=r['k'],early=r['early'],late=r['late']) for r in summary]
    halves=[dict(round='R'+str(r['round']),window=label,skip=r[key],n=100,total_k=r['k']) for r in summary for label,key in [('前25步','early'),('后25步','late')]]
    profile=[dict(round='R'+x['round'],step=int(x['step']),rate=float(x['rate']),skips=int(x['skips']),n=100) for x in rows('step_profile.csv')]
    hist=[dict(round='R'+x['round'],length_bin=x['bin'],count=int(x['count']),n=100) for x in rows('run_histogram.csv')]
    runs=[dict(round='R'+str(r['round']),**{k:r[k] for k in ['longest','early_longest','late_longest','longest_p90','longest_max','ge10','ge14','runs','mean_run']}) for r in summary]
    bands=[dict(round='R'+str(r['round']),band=r['band'],n=r['n'],late=r['late'],longest=r['longest'],first_skip=r['first_skip'],psnr=r['psnr']) for r in s['bands']]
    std=[dict(round='R'+str(r['round']),**{k:r[k] for k in ['psnr','speedup','early','late','longest']}) for r in standard]
    # Materialize reviewed aggregates and execute the exact SQL served as chart provenance.
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
    for name,records in [('halves',halves),('profile',profile),('run_hist',hist),('means',means),('runs',runs),('bands',bands),('standard',std)]:
        fields=list(records[0]);db.execute('CREATE TABLE '+name+' ('+', '.join('"'+k+'" '+('TEXT' if isinstance(records[0][k],str) else 'REAL') for k in fields)+')')
        db.executemany('INSERT INTO '+name+' VALUES ('+','.join('?' for _ in fields)+')',[[r[k] for k in fields] for r in records])
        query='SELECT '+', '.join('"'+k+'"' for k in fields)+' FROM '+name+' ORDER BY rowid'
        checked=[dict(r) for r in db.execute(query)];assert checked==records
        (OUT/(name+'.sql')).write_text(query+';\n')
        sources.append(dict(source,id=name+'_source',path='chart_data.sqlite',query=dict(source['query'],engine='SQLite',language='SQL',sql=query,tables_used=[name],description='analyze.py从已审计原始轨迹生成聚合表；本SQL读取全部已核验聚合行，保留样本分母。')))
    db.commit()
    with sqlite3.connect(OUT/'chart_data.sqlite') as dest:db.backup(dest)
    db.close()
    def table(id,title,fields):return dict(id=id,title=title,dataset=id,sourceId=id+'_source',defaultSort=dict(field='round',direction='asc'),columns=[dict(field=k,label=label,format=fmt) for k,label,fmt in fields])
    tables=[table('means','四轮采集均值',[('round','轮次','string'),('n','轨迹数','number'),('speedup','加速比（×）','number'),('psnr','PSNR（dB）','number'),('k','总skip','number'),('early','前25步skip','number'),('late','后25步skip','number')]),
        table('runs','连续skip统计',[('round','轮次','string'),('longest','最长段均值','number'),('early_longest','前段最长均值','number'),('late_longest','后段最长均值','number'),('longest_p90','最长段P90','number'),('longest_max','最长段最大值','number'),('ge10','≥10步条数','number'),('ge14','≥14步条数','number'),('runs','平均段数','number')]),
        table('bands','分预算路径统计',[('round','轮次','string'),('band','K区间','string'),('n','轨迹数','number'),('late','后25步skip','number'),('longest','最长段均值','number'),('first_skip','首次skip步号','number'),('psnr','PSNR（dB）','number')]),
        table('standard','相同K分布重加权',[('round','轮次','string'),('speedup','加速比（×）','number'),('psnr','PSNR（dB）','number'),('early','前25步skip','number'),('late','后25步skip','number'),('longest','最长段均值','number')])]
    def chart(id,title,type,x,y,c,xlabel,ylabel,percent=False):
        return dict(id=id,title=title,dataset=id,type=type,sourceId=id+'_source',intent='distribution' if id=='run_hist' else 'comparison',valueFormat='percent' if percent else 'number',
            palette=dict(kind='categorical',name='blue-orange-neutral'),legend=dict(position='bottom'),
            encodings=dict(x=dict(field=x,type='quantitative' if x=='step' else 'nominal',label=xlabel),y=dict(field=y,type='quantitative',label=ylabel),color=dict(field=c,type='nominal',label='窗口' if c=='window' else '轮次')))
    charts=[chart('halves','前后25步平均skip数量（每轮100条）','bar','round','skip','window','采集轮次','平均skip步数'),chart('profile','50步实际skip频率（每轮100条）','line','step','rate','round','去噪步号1–50','该步skip轨迹占比',True),chart('run_hist','每条轨迹最长连续skip的分布','bar','length_bin','count','round','最长连续skip步数区间','轨迹数量 / 每轮100条')]
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    artifact=dict(surface='report',manifest=dict(version=1,surface='report',title=title,generatedAt=now,description='四轮采集路径、质量和预算控制诊断',sources=sources,blocks=blocks,charts=charts,tables=tables,cards=[]),snapshot=dict(version=1,generatedAt=now,status='ready',datasets=dict(means=means,halves=halves,profile=profile,run_hist=hist,runs=runs,bands=bands,standard=std)),sources=sources)
    dump('artifact.json',artifact)
    (OUT/'RESULTS.md').write_text('\n\n'.join(b['body'] for b in blocks if b['type']=='markdown')+'\n')
    # Jupyter packages are absent. Build valid nbformat 4 JSON and execute code cells
    # sequentially in a clean Python namespace; record that this is not kernel QA.
    cells=[]
    def nbmd(id,text):cells.append(dict(id=id,cell_type='markdown',metadata={},source=text))
    def nbcode(id,text):cells.append(dict(id=id,cell_type='code',metadata={},source=text,execution_count=None,outputs=[]))
    nbmd('summary','# R1–R4策略统计复核\n\n## tl;dr\nR4加速2.4905×、PSNR23.5524dB；R2最偏后段，R3/R4回落。同K调整保留路径方向，不证明质量改善。')
    nbmd('methods','## Context & Methods\n每轮100条；前/后25步。读取封存源审计生成的完整CSV，独立用循环复算连续长度，再重算均值和预算标准化。\n\n### Key Assumptions\n各集合不同prompt，结果是描述性。共同K18–37，397/400条。')
    nbmd('data','## Data\n### 1. Load audited trajectories')
    nbcode('load',f"import csv, json, statistics\nfrom pathlib import Path\nroot=Path({str(OUT)!r})\ndata=list(csv.DictReader((root/'trajectories.csv').open()))\nexpected=json.loads((root/'SUMMARY.json').read_text())\nassert len(data)==400\nprint('400 unique trajectories:',len({{r['trace_id'] for r in data}})==400)")
    nbmd('checks','### 2. Recompute action features')
    nbcode('features',"for r in data:\n    p=r['skip_path'];best=run=0\n    for v in p:\n        run=run+1 if v=='1' else 0\n        best=max(best,run)\n    assert best==int(r['longest'])\n    assert p[:25].count('1')==int(r['early']) and p[25:].count('1')==int(r['late'])\n    assert len(p)==50 and p.count('1')==int(r['k'])\nprint('400 path features independently verified')")
    nbmd('results','## Results\n### 3. Recompute round means')
    nbcode('means',"for n in range(1,5):\n    rr=[r for r in data if int(r['round'])==n]\n    vals={k:statistics.fmean(float(r[k]) for r in rr) for k in ['speedup','psnr','early','late','longest']}\n    for k,v in vals.items(): assert abs(v-expected['rounds'][n-1][k])<1e-10\n    print(n,vals)")
    nbmd('adjust','### 4. Recompute common-K standardized means')
    nbcode('standardize',"common=set.intersection(*[{int(r['k']) for r in data if int(r['round'])==n} for n in range(1,5)])\npool=[r for r in data if int(r['k']) in common]\nassert len(pool)==397\nweights={k:sum(int(r['k'])==k for r in pool)/len(pool) for k in common}\nfor n in range(1,5):\n    vals={m:sum(w*statistics.fmean(float(r[m]) for r in data if int(r['round'])==n and int(r['k'])==k) for k,w in weights.items()) for m in ['psnr','late','longest']}\n    for m,v in vals.items(): assert abs(v-expected['standardized'][n-1][m])<1e-10\n    print(n,vals)")
    nbmd('takeaways','## Takeaways\n总预算基本不变，后段集中和长连续复用没有逐轮增强。采集集合差异和预算重加权会影响PSNR差值，等待已有R4/R8固定prompt评测判定质量收益。')
    namespace={};count=0
    for c in cells:
        if c['cell_type']=='code':
            count+=1;stream=io.StringIO()
            with contextlib.redirect_stdout(stream):exec(compile(c['source'],c['id'],'exec'),namespace)
            c['execution_count']=count;c['outputs']=[dict(output_type='stream',name='stdout',text=stream.getvalue())]
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=cells)
    assert len({c['id'] for c in cells})==len(cells) and all('source' in c and 'metadata' in c for c in cells)
    dump('analysis.ipynb',nb)
    dump('REPORT_NOTES.json',dict(audience='technical',delivery='portable HTML',structure='summary; scope before evidence; allocation; step profile; run distribution; budget bands; exact-K sensitivity; limitations; existing evaluation and open question',
        notebook='Four code cells executed sequentially in clean Python namespace and passed; nbformat/nbclient/ipykernel unavailable, no Jupyter kernel execution or external schema validation',
        chart_map=['halves: grouped bars, 8 rows, 2 window categories','profile: line, 50 ordered steps x 4 rounds, actual per-step skip frequency','run_hist: grouped bars, 6 bins x 4 rounds, 100 trajectory denominator'],
        omitted_trend='Only four rounds; use exact tables and grouped comparisons, no four-point time trend',palette='categorical blue-orange-neutral with explicit legends',uncertainty='descriptive, no p-values; common-K support 397/400; no prompt control',source_validation='400 unique sealed traces, 40000 actual CFG calls; R1–R3 quality/time means reconcile with prior report'))
    print('Canonical report and four sequentially executed notebook cells ready.')

if __name__=='__main__':main()
