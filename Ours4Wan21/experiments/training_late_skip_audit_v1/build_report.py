"""Package audited late-skip coverage as a canonical technical report and notebook."""
import csv
import hashlib
import io
import json
import sqlite3
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import torch

OUT=Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_training_late_skip_audit_v1')
EXP=OUT.parent


def rows(name):
    return list(csv.DictReader((OUT/name).open()))


def dump(name, data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    summary=json.loads((OUT/'SUMMARY.json').read_text())
    valid=json.loads((OUT/'VALIDATION.json').read_text());assert valid['status']=='pass'
    cfgpath=EXP/'ours21_dynamics128_e391_vbench50_random42_4gpu_v1/config.json'
    cfg=json.loads(cfgpath.read_text());checkpoint=Path(cfg['checkpoint'])
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();assert digest==cfg['checkpoint_sha256']
    checkpoint_data=torch.load(checkpoint,map_location='cpu',weights_only=False)
    manifest=EXP/'ours21_random3000_12groups_v1_sea7_dynamics_raw_sea128_cache/manifest.json'
    assert checkpoint_data['epoch']==391 and checkpoint_data['dataset_manifest']==json.loads(manifest.read_text())
    valid.update(e391_checkpoint_dataset_manifest_matches=True,e391_checkpoint_sha256=digest)
    valid['source_sha256'][str(cfgpath)]=hashlib.sha256(cfgpath.read_bytes()).hexdigest()
    valid['source_sha256'][str(checkpoint)]=digest
    dump('VALIDATION.json',valid)
    coverage=summary['matched'];criteria={r['criterion']:r for r in summary['criteria']}
    for r in coverage:
        r['budget']=f"K{r['k']}"
    bars=[]
    for r in coverage:
        for label,key in (('后25步skip数达到参照','match_late25'),('同时达到连续skip长度','match_both25')):
            bars.append(dict(budget=r['budget'],k=r['k'],criterion=label,rate=r[key]/r['n'],
                count=r[key],n=r['n'],reference_n=r['reference_n'],ref_late25=r['ref_late25'],ref_run25=r['ref_run25']))
    hist=[dict(late25=int(r['late25']),count=int(r['count'])) for r in rows('late25_histogram.csv')]
    sensitivity=[dict(k=r['k'],window=w,reference_late=r[f'ref_late{w}'],
        match=r[f'match_late{w}'],n=r['n'],rate=r[f'match_late{w}']/r['n']) for r in coverage for w in (25,20,15)]
    db=sqlite3.connect(':memory:')
    for table,filename in [('training_trajectories','training_trajectories.csv'),('increase_references','increase_references.csv')]:
        data=rows(filename);fields=list(data[0])
        db.execute('CREATE TABLE '+table+' ('+', '.join('"'+f+'" TEXT' for f in fields)+')')
        db.executemany('INSERT INTO '+table+' VALUES ('+','.join('?' for _ in fields)+')',[[r[f] for f in fields] for r in data])
    coverage_sql='''WITH ref AS (
      SELECT CAST(k AS INTEGER) AS k, COUNT(*) AS reference_n,
             MIN(CAST(late25 AS INTEGER)) AS ref_late25,
             MIN(CAST(run25 AS INTEGER)) AS ref_run25
      FROM increase_references GROUP BY CAST(k AS INTEGER)
    ) SELECT ref.k, ref.reference_n, ref.ref_late25, ref.ref_run25,
        COUNT(*) AS n,
        SUM(CASE WHEN CAST(t.late25 AS INTEGER)>=ref.ref_late25 THEN 1 ELSE 0 END) AS match_late25,
        SUM(CASE WHEN CAST(t.late25 AS INTEGER)>=ref.ref_late25 AND CAST(t.run25 AS INTEGER)>=ref.ref_run25 THEN 1 ELSE 0 END) AS match_both25
      FROM ref JOIN training_trajectories t ON CAST(t.k AS INTEGER)=ref.k
      WHERE t.split='train' GROUP BY ref.k ORDER BY ref.k'''
    cursor=db.execute(coverage_sql)
    checked=[dict(zip([c[0] for c in cursor.description],row)) for row in cursor.fetchall()]
    for a,b in zip(checked,coverage):
        assert all(a[k]==b[k] for k in a)
    histogram_sql='''WITH RECURSIVE bins(late25) AS (
      SELECT 0 UNION ALL SELECT late25+1 FROM bins WHERE late25<25
    ) SELECT bins.late25, COUNT(t.trajectory_id) AS count FROM bins
      LEFT JOIN training_trajectories t ON CAST(t.late25 AS INTEGER)=bins.late25 AND t.split='train'
      GROUP BY bins.late25 ORDER BY bins.late25'''
    assert db.execute(histogram_sql).fetchall()==[(r['late25'],r['count']) for r in hist]
    db.commit()
    with sqlite3.connect(OUT/'analysis.sqlite') as disk:db.backup(disk)
    db.close()
    (OUT/'coverage.sql').write_text(coverage_sql+';\n')
    (OUT/'histogram.sql').write_text(histogram_sql+';\n')
    source=dict(id='audit',label='e391训练tensor与原始trace全量审计',path='SUMMARY.json',
        query=dict(engine='Python / NumPy / PyTorch, CPU',language='Python',
            sql=coverage_sql,
            description='analyze.py读取冻结训练tensor并逐条核对3000份trace；同K比较后段动作。build_report.py核对e391 checkpoint的数据manifest。',
            tables_used=['training_trajectories.csv','increase_references.csv','matched_k_coverage.csv','criteria_sensitivity.csv'],
            filters=['train only: 2400 trajectories / 800 prompts','same realized K, no interpolation','steps displayed 1–50; late half is steps 26–50'],
            metric_definitions=['late25=sum(action[25:50]); action=1 means measured reuse, shared cond/uncond',
                'run25=longest consecutive reuse run wholly within steps 26–50',
                'reference thresholds=min observed metric among increase traces at exact same K',
                'coverage=count(train metric >= reference threshold)/count(train at same K)',
                'centroid=sum(display_step * action)/sum(action); cross-prompt comparisons describe action support, not quality']))
    coverage_source=dict(source,id='coverage_source',path='analysis.sqlite',query=dict(source['query'],
        engine='SQLite',language='SQL',sql=coverage_sql,tables_used=['training_trajectories','increase_references'],
        description='由Python从真实动作提取每条后段指标，SQLite在严格相同K下独立汇总训练覆盖率，和Python结果一致。'))
    hist_source=dict(source,id='hist_source',path='analysis.sqlite',query=dict(source['query'],
        engine='SQLite',language='SQL',sql=histogram_sql,tables_used=['training_trajectories'],
        description='全部2400条train轨迹的后25步skip数量分布，保留0–25所有整数bin。'))
    sources=[source,coverage_source,hist_source]
    title='Late-Skip Coverage in e391 Training Data'
    technical_summary=(
        '## 核心结论\n\n'
        '**高skip档位的强后段集中轨迹确实稀缺，尤其Increase 5的程度完全没有覆盖；低档位则不能说缺少。** '
        'e391真正用于参数更新的是2400条轨迹（800个prompt），另外300条验证、300条测试不计入训练分母。\n\n'
        '- 2299/2400（95.8%）已经是后半段skip多于前半段：一般的后期偏向并不缺。\n'
        '- 全训练集只有16/2400（0.67%）同时满足后25步至少22次skip、其中连续skip至少14步；来自16个prompt。\n'
        '- 后25步至少23次skip的训练轨迹为0；全3000条也为0，而Increase 5的4条均达到23次。\n'
        '- 这支持高档位的数据覆盖不足解释，但未证明补数据必然改善质量，也不是模型能力上限的证明。')
    definition=(
        '## 对比总skip数相同的真实动作\n\n'
        '后段主口径固定为50步中的**第26–50步**，第一步和最后一步的强制重算均保留。'
        '以同K的increase实测轨迹为参照，分别统计后段skip数，以及后段最长连续skip；'
        '“同时达到”要求两项都不低于参照。参照取同K已测轨迹的最小值，属于较宽松门槛。'
        'K21有3条increase参照、K22有1条，其余各4条。\n\n'
        '只比较相同总skip数，不把高K天然较多的后段skip当成额外激进；没有对未测K插值或外推。'
        '柱图展示各K内部的比例，下降表示达到对应increase程度的数据支持越来越弱。')
    matched_text=(
        '## 同K比较：从约四成降至零\n\n'
        'K21/22约40%的训练轨迹达到increase的后段skip数；K27为24.7%、K31为19.5%、K34为10.8%、K36为0。'
        '如果还要求连续skip长度不低于increase，K31降至15/133（11.3%），K34只有5/212（2.4%），K36仍为0。'
        '这说明缺口集中在高档位的强集中模式，而不是所有档位都没有后期skip示例。')
    all_text=(
        '## 全训练集也没有后25步skip 23次的轨迹\n\n'
        '全训练集后25步skip最多22次。后25步至少20、21、22次skip分别有1219（50.8%）、766（31.9%）、228（9.5%）条；'
        '达到22次且连续至少14步只剩16条，说明后段总量与连续长度需要分开衡量。'
        '下面的分布使用全部2400条训练轨迹，不控制K，用来检查支持范围，不能替代前面的同K比较。')
    mechanism=(
        '## 采集范围与训练目标能够解释这一缺口\n\n'
        '训练数据的threshold被限制在0.04–0.70，观察到的峰值也为0.70；Increase 4与5则分别递增至0.80、1.00。'
        '训练集中其实有278条（11.6%）threshold全程单调不减，1201条（50.0%）后半段平均threshold高于前半段。'
        '因此不能把原因概括为“完全没采到递增路径”；高阈值范围和最终动作的强度覆盖更值得关注。'
        '这些范围差异是已核实事实，但目前没有通过对照采集证明阈值上限是唯一原因。\n\n'
        '本地IQL实现对已记录动作计算优势加权的log-prob损失，只在actor_mask有效的自主决策行更新actor；'
        'Exact-K强制动作不直接参与actor损失。上述16条强集中轨迹共提供286个后段自主skip示例。'
        '重复训练可以重用这些示例，但不会增加不同prompt与轨迹的覆盖。此处没有估计这些示例在各epoch的优势权重或有效训练贡献。')
    robustness=(
        '## 结论对后段窗口有边界\n\n'
        '同K36比较，后25步口径是0/237；如果只看最后20步，则有9/237（3.8%）达到Increase 5的18次skip；'
        '只看最后15步则为60/237（25.3%）达到13次。'
        '所以“没有覆盖”特指整个后半程的集中程度，不能扩展成任意更短后段都缺少。'
        '以skip发生位置的平均步号作为另一口径，K36同样0/237达到increase参照，支持整体位置分配确有差异。\n\n'
        '训练prompt来自OpenVidHD，increase参照来自4条VBench prompt；同K控制了动作预算，却没有控制内容难度。'
        '本次结论是动作分布覆盖诊断，不用跨prompt的PSNR证明因果，也不把0条样本等同于策略不可能泛化。')
    validation=(
        '## 数据来源已核对到e391实际训练输入\n\n'
        '已核验e391 checkpoint SHA及其dataset manifest与本次读取的cache完全一致；'
        'transitions.pt与完成标记SHA一致。3000份原始trace、300000次CFG决策均与150000条训练transition动作一致，'
        '每条50步、两个CFG分支一致，轨迹ID唯一、每prompt三条，train/validation/test按prompt隔离。'
        '原始complete/trace/quality文件SHA均与训练manifest记录一致，训练终局reward与原始PSNR核对一致。'
        '本次未启动采集、训练或视频评测。')
    recommendations=(
        '## 下一次验证应优先针对高档位覆盖\n\n'
        '建议优先验证补充K34–36附近、后半段22–23次skip及长连续skip的真实轨迹是否有效，'
        '同时保留相同prompt/相同K的其他分配作为质量对照。'
        '补充后应分别观察轨迹覆盖、actor实际使用的自主动作，以及同K闭环质量；仅增加轨迹数量不能保证它们具有足够高的奖励。'
        '这些是基于本次诊断的建议，本次没有修改数据集或启动补采。\n\n'
        '仍待验证：这些强集中轨迹在训练prompt上是否优于较分散的轨迹，以及模型对它们的优势估计是否会给予足够训练权重。')
    blocks=[]
    def md(i,body):blocks.append(dict(id=i,type='markdown',body=body,**({} if i=='title' else {'sourceId':'audit'})))
    md('title','# '+title);md('summary',technical_summary);md('definition',definition)
    blocks.append(dict(id='coverage_plot',type='chart',chartId='coverage'))
    md('matched_text',matched_text);blocks.append(dict(id='coverage_table',type='table',tableId='matched'))
    md('all_text',all_text);blocks.append(dict(id='hist_plot',type='chart',chartId='hist'))
    md('mechanism',mechanism);md('robustness',robustness)
    blocks.append(dict(id='sensitivity_table',type='table',tableId='sensitivity'))
    md('validation',validation);md('recommendations',recommendations)
    charts=[dict(id='coverage',title='同K下达到Increase后段集中程度的训练占比',type='bar',
        dataset='coverage',sourceId='coverage_source',intent='comparison',valueFormat='percent',
        palette=dict(kind='categorical',name='blue-orange-neutral'),legend=dict(position='bottom'),
        encodings=dict(x=dict(field='budget',type='nominal',label='总skip预算'),
            y=dict(field='rate',type='quantitative',label='同K训练轨迹占比'),
            color=dict(field='criterion',type='nominal',label='判定口径'))),
        dict(id='hist',title='训练轨迹后25步skip数分布',type='bar',dataset='hist',sourceId='hist_source',
            intent='distribution',valueFormat='number',palette=dict(kind='sequential',name='blue'),
            encodings=dict(x=dict(field='late25',type='ordinal',label='第26–50步的skip数量'),
                y=dict(field='count',type='quantitative',label='训练轨迹数')))]
    table=dict(id='matched',title='严格同K的覆盖数量',dataset='matched',sourceId='audit',
        defaultSort=dict(field='k',direction='asc'),columns=[
            dict(field='k',label='K',format='number'),dict(field='n',label='训练条数',format='number'),
            dict(field='ref_late25',label='参照后25步skip',format='number'),
            dict(field='match_late25',label='达到条数',format='number'),dict(field='late25_rate',label='比例',format='percent'),
            dict(field='ref_run25',label='参照连续长度',format='number'),
            dict(field='match_both25',label='两项均达到',format='number'),dict(field='both25_rate',label='两项比例',format='percent')])
    sens_table=dict(id='sensitivity',title='后段窗口敏感性（仅skip数量）',dataset='sensitivity',sourceId='audit',
        defaultSort=dict(field='k',direction='asc'),columns=[dict(field=f,label=l,format=fmt) for f,l,fmt in
            [('k','K','number'),('window','最后多少步','number'),('reference_late','参照skip数','number'),
             ('match','达到条数','number'),('n','同K训练条数','number'),('rate','占比','percent')]])
    now=datetime.now(timezone.utc).isoformat()
    artifact=dict(surface='report',manifest=dict(version=1,surface='report',title=title,
        generatedAt=now,description='e391训练数据后段集中skip覆盖诊断',sources=sources,blocks=blocks,
        charts=charts,tables=[table,sens_table],cards=[]),
        snapshot=dict(version=1,generatedAt=now,status='ready',datasets=dict(coverage=bars,matched=coverage,hist=hist,sensitivity=sensitivity)),sources=sources)
    dump('artifact.json',artifact)
    (OUT/'RESULTS.md').write_text('\n\n'.join(b['body'] for b in blocks if b['type']=='markdown')+'\n')
    code="""import csv
from pathlib import Path
root = Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_training_late_skip_audit_v1')
data = list(csv.DictReader((root / 'training_trajectories.csv').open()))
refs = list(csv.DictReader((root / 'increase_references.csv').open()))
train = [r for r in data if r['split'] == 'train']
assert len(train) == 2400
def longest(path):
    best = run = 0
    for value in path:
        run = run + 1 if value == '1' else 0
        best = max(best, run)
    return best
print('K | n | ref late25 | match late25 | ref run25 | match both')
for k in sorted({int(r['k']) for r in refs}):
    local = [r for r in train if int(r['k']) == k]
    ref = [r for r in refs if int(r['k']) == k]
    late = min(r['skip_path'][25:].count('1') for r in ref)
    run = min(longest(r['skip_path'][25:]) for r in ref)
    match = sum(r['skip_path'][25:].count('1') >= late for r in local)
    both = sum(r['skip_path'][25:].count('1') >= late and longest(r['skip_path'][25:]) >= run for r in local)
    print(k, len(local), late, match, run, both)
assert all(r['skip_path'][25:].count('1') < 23 for r in data)
"""
    stream=io.StringIO()
    with redirect_stdout(stream):exec(code,{})
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        cells=[dict(id='scope',cell_type='markdown',metadata={},source=['# e391训练数据后段skip审计\n',
            '本notebook从完整逐轨迹CSV独立重算关键覆盖率；原始tensor和3000份trace的完整审计入口是同目录对应实验代码 `analyze.py`。\n',
            '主口径：同K下，第26–50步skip数量；补充口径再要求后段最长连续skip不少于increase参照。\n']),
            dict(id='audit',cell_type='code',metadata={},execution_count=1,source=code.splitlines(keepends=True),
                outputs=[dict(output_type='stream',name='stdout',text=stream.getvalue().splitlines(keepends=True))])])
    dump('analysis.ipynb',nb)
    (OUT/'README.md').write_text('# e391训练数据后段skip覆盖\n\n'
        '- `report.html`：技术报告；`artifact.json`：canonical源；`RESULTS.md`：文字结论。\n'
        '- `analysis.ipynb`：从CSV独立复核主结论的已执行notebook。\n'
        '- `training_trajectories.csv`：全部3000条，含split、50步动作、后段统计、prompt、threshold和actor mask统计。\n'
        '- `matched_k_coverage.csv`：同K参照覆盖，含所有split和后25/20/15步。\n'
        '- `increase_references.csv`：20条increase实测参照。\n'
        '- `criteria_sensitivity.csv` / `train_k_profile.csv` / `late25_histogram.csv`：全量、分K与敏感性统计。\n'
        '- `SUMMARY.json` / `VALIDATION.json`：机器可读结果及原始来源SHA。\n\n'
        '代码：Ours4Wan21/experiments/training_late_skip_audit_v1/，先运行analyze.py再运行build_report.py；'
        '最后使用Data Analytics插件的deliver_portable_artifact.mjs将artifact.json打包为report.html。\n')
    dump('CHART_QA.json',dict(surface='portable HTML',audience='technical',
        plots=['same-K grouped bar, 6 budgets × 2 criteria, fraction 0–1','late25 histogram, 26 integer bins, zero-based count axis'],
        palette='blue/orange + neutrals',non_color_encoding='legend labels and exact table',
        sample_scope='2400 train trajectories only; reference minima at same K, no interpolation',
        notebook_independent_recount=stream.getvalue(),
        structure='technical summary; definitions before evidence; same-K findings; global distribution; collection/training mechanism; sensitivity/limits; source verification; suggested validation and open questions'))
    print(stream.getvalue())


if __name__=='__main__':main()
