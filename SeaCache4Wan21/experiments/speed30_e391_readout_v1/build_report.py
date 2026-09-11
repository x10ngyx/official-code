"""Package the audited comparison through the canonical analytics report reader."""
import csv
import json
import sqlite3
from pathlib import Path

ROOT=Path('/mnt/hdd/xiongyuxiang/tmp/exp/seacache_wan21_vbench50_speed30_4gpu_v1/analysis/e391_comparison')
audit=json.loads((ROOT/'AUDIT.json').read_text())
assert audit['status']=='pass'
with (ROOT/'comparison.csv').open() as f:
    data=list(csv.DictReader(f))
sea,ours=data
def n(r,k): return float(r[k])
title='SeaCache 3.0× 与 Dynamics128 e391：同随机50条 prompt 对比'
lpips_reduction=(1-n(sea,'lpips_alex_v0_1_spatial')/n(ours,'lpips_alex_v0_1_spatial'))*100
time_reduction=(1-n(sea,'candidate_generate_seconds_mean')/n(ours,'candidate_generate_seconds_mean'))*100
source=dict(id='paired_source',label='SeaCache 3.0× 与 e391 K35 已验收的配对结果',
    path='seacache_wan21_vbench50_speed30_4gpu_v1/analysis/e391_comparison/comparison.csv',
    query=dict(engine='Python standard library',language='Python',
        description='compare.py 校验完整标记、来源哈希、同GPU和baseline视频身份、质量视频哈希与逐视频均值，并计算SeaCache减e391的配对差值。',
        tables_used=['seacache_wan21_vbench50_speed30_4gpu_v1/analysis/results.csv',
            'seacache_wan21_vbench50_speed30_4gpu_v1/analysis/per_video.csv',
            'ours21_dynamics128_e391_vbench50_random42_4gpu_v1/analysis/results.csv',
            'ours21_dynamics128_e391_vbench50_random42_4gpu_v1/analysis/per_video.csv'],
        filters=['同一冻结随机50条prompt，四卡13/13/12/12分片','e391只取K35','seed=42，81帧，832×480，50-step UniPC'],
        metric_definitions=['完整推理时间为pipeline.generate wall time，排除模型加载、warmup及视频保存。',
            '实际加速比=sum(baseline_seconds)/sum(candidate_seconds)，50条共同baseline均值254.481844秒。',
            'PSNR、SSIM、LPIPS先按帧汇总为视频均值，再对50条视频等权平均。',
            'DiT TFLOPs按实际执行的branch blocks计数；T5/VAE为固定shape profile估算。',
            '分组件CUDA计时与predictor/feature诊断包含在完整generate之内，不能重复相加。']))
definitions=[('完整推理时间（秒/视频，↓）','candidate_generate_seconds_mean',3),
    ('实际加速（×，↑）','latency_speedup',5),('DiT TFLOPs/视频','candidate_dit_tflops_mean',3),
    ('PSNR（dB，↑）','psnr_rgb_db',5),('SSIM（↑）','ssim_rgb',6),
    ('LPIPS（↓）','lpips_alex_v0_1_spatial',6)]
summary=[dict(order=i,metric=label,seacache=f'{n(sea,key):.{p}f}',e391=f'{n(ours,key):.{p}f}',
    delta=f'{n(sea,key)-n(ours,key):+.{p}f}') for i,(label,key,p) in enumerate(definitions)]
components=[]
for i,(label,key,p) in enumerate([
    ('T5 CUDA秒/视频','candidate_t5_cuda_seconds_mean',6),
    ('DiT CUDA秒/视频','candidate_dit_cuda_seconds_mean',6),
    ('VAE decode CUDA秒/视频','candidate_vae_decode_cuda_seconds_mean',6),
    ('T5 TFLOPs/视频','candidate_estimated_t5_tflops_per_video_mean',6),
    ('VAE decode TFLOPs/视频','candidate_estimated_vae_decode_tflops_per_video_mean',6),
    ('Predictor TFLOPs/视频','candidate_predictor_tflops_mean',10),
    ('Predictor network CUDA秒/视频','candidate_predictor_network_cuda_seconds_mean',6),
    ('Predictor decision wall秒/视频','candidate_predictor_decision_wall_seconds_mean',6),
    ('Latent feature wall秒/视频','candidate_latent_feature_wall_seconds_mean',6)]):
    components.append(dict(order=i,metric=label,seacache=f'{n(sea,key):.{p}f}',e391=f'{n(ours,key):.{p}f}'))
pair_rows=[]
for i,(label,key) in enumerate([('PSNR','psnr_rgb_db'),('SSIM','ssim_rgb'),('LPIPS','lpips_alex_v0_1_spatial')]):
    r=audit['paired'][key]
    pair_rows.append(dict(order=i,metric=label,seacache_better=r['seacache_better'],e391_better=r['e391_better'],
        median=f"{r['median_delta']:+.6f}",range=f"{r['min_delta']:+.6f} 至 {r['max_delta']:+.6f}"))
def table(id,title,subtitle,cols):
    return dict(id=id,title=title,subtitle=subtitle,dataset=id,sourceId='paired_source',density='normal',
        defaultSort=dict(field='order',direction='asc'),columns=[dict(field='order',label='序号',format='number')]+
        [dict(field=k,label=v,type='text') for k,v in cols])
tables=[table('summary','3.0×档汇总','50条视频等权均值；差值为SeaCache − e391',
    [('metric','指标'),('seacache','SeaCache'),('e391','e391 K35'),('delta','差值')]),
    table('paired','逐prompt质量胜负','同50个配对；PSNR/SSIM越高越好，LPIPS越低越好；差值为SeaCache − e391',
    [('metric','指标'),('seacache_better','SeaCache更好条数'),('e391_better','e391更好条数'),('median','差值中位数'),('range','差值范围')]),
    table('components','分组件时间与计算量','每视频均值；predictor与feature开销已包含于完整推理时间',
    [('metric','指标'),('seacache','SeaCache'),('e391','e391 K35')])]
def md(id,body): return dict(id=id,type='markdown',body=body,sourceId='paired_source')
blocks=[md('title','# '+title),
    md('answer',f'## 在相同DiT计算量、接近等速下，SeaCache的三项质量均值略优\n\nSeaCache 3.0×补测于2026-09-10 00:51（北京时间）完成50候选和4,050对帧评测。相比e391 K35，PSNR高0.228365 dB、SSIM高0.015517，LPIPS低{lpips_reduction:.2f}%。两者实测分别3.01234×和3.00325×，耗时仅差0.255454秒（{time_reduction:.2f}%）。本档结果没有显示e391质量优势。'),
    md('scope','## 同50条prompt、同卡baseline，固定3.0×参数\n\n两者使用同一批从VBench200随机抽取的50条prompt及同物理GPU的native baseline。Wan2.1-T2V-1.3B，batch=1，832×480、81帧、16fps、50-step UniPC、shift=5、CFG=5、seed=42、DiT BF16；全部模型驻GPU，无offload或prompt扩写。SeaCache固定threshold=0.4384497621471517；Dynamics128固定e391、K35。质量为相对同一baseline的PSNR/SSIM/LPIPS，先逐帧再逐视频等权平均。加速比为baseline总推理时间除以candidate总推理时间，baseline均值254.481844秒。'),
    md('quality','## 均值差距不大，逐prompt各有胜负\n\nSeaCache的PSNR高0.228365 dB，但配对差值中位数只有0.070321 dB。50条中SeaCache在PSNR/SSIM/LPIPS上分别有32/35/31条更好，e391分别有18/15/19条更好。结果支持本批样本的均值排序，不表示SeaCache对所有prompt都更好。'),
    dict(id='summary_block',type='table',tableId='summary'),
    dict(id='paired_block',type='table',tableId='paired'),
    md('compute','## 每视频均35步复用、15步重算，DiT预算完全相同\n\n两方法全部100个视频的10,000条CFG branch-step动作已与实际blocks执行核对；每条均35步复用、15步重算。DiT计算量均为8,551.315670 TFLOPs/视频，相对baseline计算量缩减3.33273×。完整推理耗时分别84.479905和84.735358秒；实验分时执行，0.30%的差值不足以证明稳定速度优势，也不能单独用predictor/feature分项解释全部时间差。'),
    dict(id='components_block',type='table',tableId='components'),
    md('validation','## 完整性与配对验收通过\n\n复核SeaCache及e391最终完成标记、汇总与验证文件哈希、20/36份结果证据文件、50个共同prompt与baseline视频身份、同物理GPU、checkpoint及固定参数；逐视频质量与全部组件均值重新汇总一致。视频质量引用的MP4哈希一致，本次没有重新解码计算质量或重新推理。所有比较只取e391 K35，不混入其他档位。'),
    md('limits','## 结论限于当前50条prompt与单个生成seed\n\n这是描述性比较，未进行显著性检验或多seed重复，不能据此推断所有prompt上的稳定排名。VBench已按该实验既有用户要求跳过，不能由三项参考质量推导VBench分数。Latent feature只有wall时间，没有完整算子TFLOPs估算；因此这里严格称相同DiT预算，不称完整pipeline所有运算量完全相同。'),
    md('interpretation','## 当前读出的用途与未回答问题\n\n现有证据足以完成3.0×补测汇报，并将本档记录为SeaCache均值略优、两者近似等速。是否在更多prompt或其他seed下维持排序尚无证据；本次不据此新增实验计划。')]
db=sqlite3.connect(':memory:')
db.row_factory=sqlite3.Row
db.execute('CREATE TABLE paired_results(sample_id TEXT, psnr_delta REAL, ssim_delta REAL, lpips_delta REAL)')
with (ROOT/'paired.csv').open() as f:
    for r in csv.DictReader(f):
        db.execute('INSERT INTO paired_results VALUES (?,?,?,?)',(r['sample_id'],float(r['delta_psnr_rgb_db']),
            float(r['delta_ssim_rgb']),float(r['delta_lpips_alex_v0_1_spatial'])))
queries=[]
for metric,col,direction in [('PSNR','psnr_delta','>'),('SSIM','ssim_delta','>'),('LPIPS','lpips_delta','<')]:
    for method,op in [('SeaCache',direction),('e391 K35','<' if direction=='>' else '>')]:
        queries.append(f"SELECT '{metric}' AS metric, '{method}' AS method, SUM(CASE WHEN {col} {op} 0 THEN 1 ELSE 0 END) AS better_prompts, COUNT(*) AS paired_prompts FROM paired_results")
sql=' UNION ALL '.join(queries)
win_rows=[dict(r) for r in db.execute(sql)]
(ROOT/'wins_query.sql').write_text(sql+';\n')
win_source=dict(id='wins_source',label='50个逐prompt配对的三项质量胜负',
    path='seacache_wan21_vbench50_speed30_4gpu_v1/analysis/e391_comparison/paired.csv',
    query=dict(engine='SQLite in-memory',language='SQL',sql=sql,
        description='paired_results从paired.csv的sample_id和SeaCache减e391三项质量差值逐行加载，按更好方向计数。',
        tables_used=['paired_results'],filters=['all 50 paired prompts, e391 K35 vs SeaCache nominal3.0x'],
        metric_definitions=['PSNR/SSIM差值为正计SeaCache胜；LPIPS差值为负计SeaCache胜；反向计e391胜。']))
charts=[dict(id='quality_wins',title='同50条prompt的三项质量胜负',
    subtitle='每项指标分别比较；纵轴为质量更好的prompt条数（共50条），不表示质量改善幅度',
    type='bar',intent='comparison',dataset='wins',sourceId='wins_source',
    palette=dict(kind='categorical',name='blue-orange-neutral'),legend=dict(position='bottom'),
    labels=dict(values='auto'),encodings=dict(
        x=dict(field='metric',type='nominal',label='质量指标'),
        y=dict(field='better_prompts',type='quantitative',label='更好的prompt条数'),
        color=dict(field='method',type='nominal',label='方法'),
        tooltip=[dict(field='paired_prompts',type='quantitative',label='配对总数')]),
    comparisonContext=dict(grain='one method per metric over the same 50 paired prompts',unit='prompt count'))]
blocks.insert(next(i for i,b in enumerate(blocks) if b['id']=='paired_block'),
    dict(id='wins_block',type='chart',chartId='quality_wins'))
datasets=dict(summary=summary,paired=pair_rows,components=components,wins=win_rows)
table_sources=[]
for t in tables:
    dataset=t['dataset']
    rr=datasets[dataset]
    cols=list(rr[0])
    db.execute('CREATE TABLE '+dataset+'_reviewed ('+', '.join('"'+k+'"' for k in cols)+')')
    db.executemany('INSERT INTO '+dataset+'_reviewed VALUES ('+','.join('?' for _ in cols)+')',
        [tuple(r[k] for k in cols) for r in rr])
    query='SELECT * FROM '+dataset+'_reviewed ORDER BY "order"'
    datasets[dataset]=[dict(r) for r in db.execute(query)]
    ts=dict(source,id=dataset+'_source')
    ts['query']=dict(source['query'],engine='SQLite in-memory',language='SQL',sql=query,
        description=source['query']['description']+' build_report.py将上述Python算出的精确结果行加载为'+dataset+'_reviewed；SQL仅提供展示顺序，原始计算与来源保留在compare.py及AUDIT.json。',
        tables_used=[dataset+'_reviewed',*source['query']['tables_used']])
    t['sourceId']=ts['id']
    table_sources.append(ts)
db.close()
all_sources=[source,win_source,*table_sources]
artifact=dict(surface='report',manifest=dict(version=1,surface='report',title=title,
    description='固定推理协议下的质量、实际速度与分组件计量配对读出。',generatedAt=audit['checked_at'],
    cards=[],charts=charts,tables=tables,sources=all_sources,blocks=blocks),
    snapshot=dict(version=1,generatedAt=audit['checked_at'],status='ready',
        datasets=datasets,accessIssues=[]),sources=all_sources)
(ROOT/'artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
(ROOT/'SOURCE_NOTES.md').write_text('# 读出说明\n\n选择technical报告：实验配对与组件计量。summary与scope前置，质量/计算/验收/限制分段。\n两种方法的混单位质量幅度使用精确对照表。图选Comparison分组柱状：三项质量指标×两方法，\n以相同的prompt条数为单位、零基线、蓝橙两色，使用全部50个配对，无额外指标可扩展。\n图仅用于补充均值解释，不表示改善幅度或显著性，邻近说明明确计数定义。\n建议与开放问题合并在最后一节，未新增实验计划。\n来源绝对路径和SHA保留在AUDIT.json；HTML中用精确结果目录名和文件名作为可移植来源身份。\n')
print(ROOT/'artifact.json')
