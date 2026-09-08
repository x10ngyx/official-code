#!/usr/bin/env python3
"""Independently verify the frozen timing audit and author its portable report input."""
import argparse
import contextlib
import io
import json
import os
import sqlite3
from pathlib import Path


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def materialize_chart_queries(out, artifact):
    """Use real, inspectable SQL to select the already derived display datasets."""
    for row in artifact['snapshot']['datasets']['estimates']:
        row['coefficient']=f"r={row['multiplier']:.6f}"
    for row in artifact['snapshot']['datasets']['preview']:
        for key in ('raw_speedup','estimated_corrected_speedup'):
            row[key+'_display']=f"{row[key]:.6f}×"
    for table in artifact['manifest']['tables']:
        for column in table['columns']:
            if column['field']=='coefficient':
                column['type']='text'
            elif column['field'] in ('raw_speedup','estimated_corrected_speedup'):
                column['field']+='_display'
                column['type']='text'
    db=sqlite3.connect(out/'audit.sqlite')
    db.row_factory=sqlite3.Row
    for name, rows in artifact['snapshot']['datasets'].items():
        fields=list(rows[0])
        declarations=', '.join('"'+k+'" '+('TEXT' if isinstance(rows[0][k],str) else 'REAL') for k in fields)
        db.execute('DROP TABLE IF EXISTS "'+name+'"')
        db.execute('CREATE TABLE "'+name+'" ('+declarations+')')
        db.executemany('INSERT INTO "'+name+'" VALUES ('+','.join('?' for _ in fields)+')',[[r[k] for k in fields] for r in rows])
        order={'estimates':'"order"','temporal':'gpu, bin','preview':'cohort'}[name]
        query='SELECT * FROM "'+name+'" ORDER BY '+order
        artifact['snapshot']['datasets'][name]=[dict(r) for r in db.execute(query)]
        artifact['manifest']['sources']=[src for src in artifact['manifest']['sources'] if src['id']!=name+'_query']
        artifact['manifest']['sources'].append(dict(id=name+'_query',label=name+'：已验证统计的展示数据',path='audit.sqlite',query={'engine':'SQLite','language':'SQL','sql':query,'tables_used':[name],'description':'build_evidence.py将已核对的summary.json统计写入派生表，再执行此SQL选择展示数据；原始计时来源为source_manifest.json列明的正式实验文件。'}))
    for item in artifact['manifest']['charts']+artifact['manifest']['tables']:
        item['sourceId']=item['dataset']+'_query'
    artifact['sources']=artifact['manifest']['sources']
    db.commit()
    db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    out = parser.parse_args().result_dir.resolve(strict=True)
    assert out.is_relative_to(Path('/all/yiran07-disk3/huteng_data/exp'))
    assert all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
    sql = '''WITH healthy AS (
  SELECT cohort, schedule, COUNT(*) AS n, AVG(wall_seconds) AS expected,
         AVG(dit_seconds) AS expected_dit, AVG(non_dit_seconds) AS expected_other
  FROM videos WHERE gpu IN (1,2,3) GROUP BY cohort, schedule
)
SELECT v.cohort, COUNT(*) AS n, MIN(h.n) AS min_peers,
       AVG(v.wall_seconds) AS observed, AVG(h.expected) AS expected,
       SUM(v.wall_seconds)/SUM(h.expected) AS multiplier,
       (SUM(v.dit_seconds)-SUM(h.expected_dit)) /
       (SUM(v.wall_seconds)-SUM(h.expected)) AS dit_share,
       SUM(v.wall_seconds-h.expected-v.dit_seconds+h.expected_dit-v.non_dit_seconds+h.expected_other) AS residual
FROM videos v JOIN healthy h ON v.cohort=h.cohort AND v.schedule=h.schedule
WHERE v.gpu=0 GROUP BY v.cohort ORDER BY v.cohort'''
    (out/'verify.sql').write_text(sql+';\n')
    code = [
'''import os, csv, json, sqlite3, hashlib, math, statistics
from collections import Counter
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key) == '1'
root = Path.cwd()
summary = json.loads((root/'summary.json').read_text())
rows = list(csv.DictReader((root/'videos.csv').open()))
assert len(rows) == 1400
assert len({(r['cohort'], r['sample_id']) for r in rows}) == 1400
counts = Counter((r['cohort'], int(r['gpu'])) for r in rows)
assert len(counts) == 28 and set(counts.values()) == {50}
assert all(len(r['schedule']) == 100 and set(r['schedule']) <= {'F','R'} for r in rows)
assert all(r['schedule'].count('F') == int(r['full_calls']) for r in rows)
print('PASS: 1400 distinct videos, 7 cohorts × 4 GPUs × 50 videos, 140000 calls.')''',
'''db = sqlite3.connect(':memory:')
db.row_factory = sqlite3.Row
db.execute('CREATE TABLE videos (cohort TEXT, gpu INTEGER, sample_id TEXT, schedule TEXT, wall_seconds REAL, dit_seconds REAL, non_dit_seconds REAL)')
db.executemany('INSERT INTO videos VALUES (?,?,?,?,?,?,?)', [(r['cohort'],int(r['gpu']),r['sample_id'],r['schedule'],float(r['wall_seconds']),float(r['dit_seconds']),float(r['non_dit_seconds'])) for r in rows])
verified = {r['cohort']:dict(r) for r in db.execute((root/'verify.sql').read_text())}
errors=[]
for result in summary['summaries']:
    v = verified[result['cohort']]
    assert v['n'] == 50 and v['min_peers'] >= 15
    errors.append(abs(v['multiplier'] - result['multiplier']))
    assert math.isclose(v['multiplier'],result['multiplier'],abs_tol=1e-12)
    assert math.isclose(v['observed'],result['gpu0_mean_seconds'],abs_tol=1e-9)
    assert math.isclose(v['expected'],result['reference_matched_mean_seconds'],abs_tol=1e-9)
    assert abs(v['residual']) < 1e-8
    assert math.isclose(100*v['dit_share'],result['dit_share_of_excess_pct'],abs_tol=1e-8)
    assert math.isclose(100*(1-1/v['multiplier']),result['throughput_loss_pct'],abs_tol=1e-10)
    print(f"{result['label']}: r={v['multiplier']:.8f}, time +{100*(v['multiplier']-1):.4f}%, n={v['n']}")
print('Maximum independent SQL coefficient error:', max(errors))''',
'''manifest = json.loads((root/'source_manifest.json').read_text())
for source in manifest['sources']:
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256'], source['path']
print('PASS:', len(manifest['sources']), 'source SHA256 hashes unchanged.')
spreads=[]
for cohort in verified:
    ratios=[r['matched_ratio'] for r in summary['per_gpu'] if r['cohort']==cohort and r['gpu']!=0]
    spreads.append(100*(max(ratios)/min(ratios)-1))
print('Maximum healthy GPU matched-mean spread:',max(spreads),'%')
factors={r['cohort']:r['multiplier'] for r in summary['summaries']}
corrected={c:statistics.mean(float(r['wall_seconds'])/(factors[c] if int(r['gpu'])==0 else 1) for r in rows if r['cohort']==c) for c in verified}
assert math.isclose(corrected['baseline'],summary['shared_baseline_estimated_corrected_mean'],abs_tol=1e-9)
for p in summary['correction_preview']:
    assert math.isclose(corrected['baseline']/corrected[p['cohort']],p['estimated_corrected_speedup'],abs_tol=1e-12)
print('PASS: all correction previews independently recomputed; raw times unchanged.')''',
'''paired=list(csv.DictReader((root/'same_prompt_baselines.csv').open()))
assert len(paired)==11 and len({r['sample_id'] for r in paired})==11
p0=[r for r in paired if int(r['old_gpu'])==0]
ph=[r for r in paired if int(r['old_gpu'])!=0]
assert (len(p0),len(ph))==(4,7)
ratio=lambda rr:sum(float(r['old_seconds']) for r in rr)/sum(float(r['new_gpu123_mean_seconds']) for r in rr)
assert math.isclose(ratio(p0)/ratio(ph),summary['paired_baseline']['date_adjusted_gpu0_ratio'],abs_tol=1e-12)
print('Same-prompt check: GPU0',100*(ratio(p0)-1),'%; old healthy',100*(ratio(ph)-1),'%; date-adjusted',100*(ratio(p0)/ratio(ph)-1),'%.')
validation=dict(status='passed',video_count=1400,cfg_calls=140000,source_hashes_unchanged=len(manifest['sources']),independent_sql_max_coefficient_error=max(errors),max_peer_spread_pct=max(spreads),raw_timing_modified=False,notebook_execution='Python code cells executed top-to-bottom in wan2.2; Jupyter kernel unavailable',thermal_causality='Consistent with contemporaneous 91–92 C SW thermal notes; no continuous per-video telemetry')
(root/'VALIDATION.json').write_text(json.dumps(validation,indent=2,ensure_ascii=False)+'\\n')'''
    ]
    cells=[dict(cell_type='markdown',metadata={},source='# GPU0 slowdown audit\n\nIndependent SQLite reconstruction of the exact 100-call matching estimator. Run from this results directory with the four BLAS thread variables set to 1. The wan2.2 environment has no nbformat, nbclient or ipykernel: the saved code cells were executed in order by Python; this is not a Jupyter-kernel execution. Raw source SHA values are checked without modifying measurements.')]
    previous=Path.cwd()
    os.chdir(out)
    namespace={'__name__':'__main__'}
    try:
        for i, source in enumerate(code,1):
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):
                exec(compile(source,f'gpu0_audit_cell_{i}','exec'),namespace)
            output=stream.getvalue()
            print(output,end='')
            cells.append(dict(cell_type='code',metadata={'execution_method':'sequential_python'},execution_count=i,source=source,outputs=[dict(output_type='stream',name='stdout',text=output)]))
    finally:
        os.chdir(previous)
    dump(out/'audit.ipynb',dict(nbformat=4,nbformat_minor=4,cells=cells,metadata={'kernelspec':{'display_name':'Python (wan2.2)','language':'python','name':'python3'},'language_info':{'name':'python'},'execution_note':'Executed using Python exec in wan2.2; no Jupyter kernel installed.'}))
    s=json.loads((out/'summary.json').read_text())
    validation=json.loads((out/'VALIDATION.json').read_text())
    estimates=[]
    for i,r in enumerate(s['summaries']):
        flat={k:v for k,v in r.items() if not isinstance(v,dict)}
        flat.update({'bootstrap_'+k:v for k,v in r['bootstrap'].items()})
        estimates.append(dict(flat,order=i,ci95=f"{r['ci95_low_pct']:.2f}–{r['ci95_high_pct']:.2f}%",coefficient=round(r['multiplier'],6),gpu0_seconds=round(r['gpu0_mean_seconds'],2),reference_seconds=round(r['reference_matched_mean_seconds'],2),inflation_percent=round(r['time_inflation_pct'],2),loss_percent=round(r['throughput_loss_pct'],2)))
    temporal=[dict(r,series=r['label']+' · GPU'+str(r['gpu'])) for r in s['temporal_bins'] if r['cohort']=='teacache_0p68']
    sources=[dict(id='analysis',label='1400条计时的计算路径匹配统计与bootstrap',path='summary.json',query={'language':'Python','description':'audit.py：按方法/阈值及完整100-call full/reuse序列，以GPU1/2/3均值匹配GPU0；基于原始文件的SHA清单见source_manifest.json。'}),dict(id='validation',label='独立SQLite复算及原始来源完整性验证',path='VALIDATION.json',query={'engine':'SQLite','language':'SQL','sql':sql,'tables_used':['videos'],'description':'audit.ipynb将videos.csv导入内存SQLite videos表并执行verify.sql；独立于原始Python统计实现。'}),dict(id='raw',label='正式baseline、六档candidate及历史温度观察的文件SHA清单',path='source_manifest.json'),dict(id='preview',label='只归一化GPU0后的计时与速度估算',path='correction_preview.csv')]
    charts=[dict(id='slowdown',title='各批次GPU0耗时增幅',subtitle='完整generate；GPU1/2/3匹配参照=0%；95%区间见下表',intent='comparison',question='GPU0在七个批次分别慢多少？',rationale='单一度量的七个离散批次，用零起点柱图比较；不按方法人为合并。',comparisonContext={'baseline':'GPU1/2/3 exact-schedule mean','grain':'batch-specific GPU0 50 videos vs 150 references','unit':'%','semanticFamily':'generation time inflation'},type='bar',dataset='estimates',sourceId='analysis',encodings={'x':{'field':'label','type':'nominal','label':'批次'},'y':{'field':'time_inflation_pct','type':'quantitative','label':'耗时增加','format':'number','unit':'%'},'tooltip':[{'field':'gpu0_mean_seconds','type':'quantitative','label':'GPU0秒/视频'},{'field':'reference_matched_mean_seconds','type':'quantitative','label':'匹配参照秒/视频'},{'field':'ci95','type':'nominal','label':'95%区间'}]},palette={'kind':'sequential','name':'blue'},labels={'values':'all'},valueFormat='number',unit='%',layout='full'),
        dict(id='temporal',title='TeaCache 0.68按运行顺序的变化',subtitle='每点5条视频；各GPU按各自运行次序分箱，不表示同一时刻',intent='trend',question='GPU0降速在批次内部是否稳定？',rationale='有序运行窗口用折线图显示GPU0漂移，以三张参照卡的序列提供背景。',comparisonContext={'baseline':'whole-batch GPU1/2/3 exact-schedule mean','grain':'5 videos per GPU per sequential bin','unit':'%','semanticFamily':'generation time inflation'},type='line',dataset='temporal',sourceId='analysis',encodings={'x':{'field':'bin','type':'quantitative','label':'运行分箱（每箱5条）'},'y':{'field':'time_inflation_pct','type':'quantitative','label':'耗时偏差','format':'number','unit':'%'},'color':{'field':'series','type':'nominal','label':'GPU'},'tooltip':[{'field':'sample_start','type':'quantitative','label':'起始序号'},{'field':'sample_end','type':'quantitative','label':'结束序号'}]},palette={'kind':'categorical','name':'default'},labels={'values':'none'},valueFormat='number',unit='%',layout='full')]
    columns=[('order','序号','number'),('label','批次','text'),('gpu0_seconds','GPU0 秒','number'),('reference_seconds','参照 秒','number'),('inflation_percent','耗时增加 %','number'),('loss_percent','吞吐下降 %','number'),('coefficient','除数 r','number'),('ci95','耗时增幅95%区间','text')]
    tables=[dict(id='factors',title='用于分批次归一化的估计系数',subtitle='每批GPU0 n=50；GPU1/2/3 n=150；秒数为每视频均值',dataset='estimates',sourceId='analysis',defaultSort={'field':'order','direction':'asc'},density='spacious',columns=[dict(field=f,label=l,type=t) for f,l,t in columns]),
      dict(id='preview',title='完整VBench200加速比的修正预览',subtitle='仅为GPU0除以对应r后的算术估算，未替换原始实测',dataset='preview',sourceId='preview',defaultSort={'field':'cohort','direction':'asc'},density='spacious',columns=[dict(field=f,label=l,type=t) for f,l,t in [('cohort','批次','text'),('raw_speedup','原速度 ×','number'),('estimated_corrected_speedup','估算修正速度 ×','number'),('raw_candidate_mean','原候选秒','number'),('estimated_corrected_candidate_mean','修正候选秒','number')]])]
    blocks=[]
    def md(id,body,source='analysis'):
        block=dict(id=id,type='markdown',body=body)
        if source: block['sourceId']=source
        blocks.append(block)
    title='GPU0 Slowdown in Wan2.2 Benchmarks'
    md('title','# '+title,None)
    md('summary','## 技术摘要：GPU0平均多耗时10%–15%\n\n本次覆盖共享200条baseline与SeaCache、TeaCache各三档正式candidate，共1400条实测视频。相同方法、阈值及100次计算/复用序列匹配后，GPU0的完整generate耗时高出 **10.02%–15.27%**，对应吞吐下降 **9.11%–13.25%**。Baseline增幅为 **10.76%**。\n\n额外耗时几乎全部来自DiT，但批次内也存在漂移。应对每个批次分别估计修正，保留原始测量与估算结果的区别。')
    md('scope','## 比较口径：同方法、同计算序列，统一完整generate\n\n数据窗口为2026-08-30至09-05（CST）；共享baseline只计一次，六档candidate分别统计。每批200个固定prompt按四张物理GPU分片，每卡50条；物理卡号由generation config、生命周期元数据及baseline的CUDA_VISIBLE_DEVICES日志交叉确认。\n\n固定Wan2.2-T2V-A14B、832×480、45帧、50步DPM++、shift=12、CFG=(3,4)、seed=42、BF16、offload=True。Latency为CUDA同步的完整generate墙钟时间，含T5、DiT、VAE和迁移/调度，排除初始化、warmup、编码与评测。\n\n对每条GPU0视频，以GPU1/2/3中同批次且完整100-call F/R序列相同的样本均值作为参照。系数 **r=ΣGPU0秒数/Σ匹配参照秒数**；耗时增加=(r−1)×100%，吞吐下降=(1−1/r)×100%。所有350条GPU0视频均有参照，每个匹配层至少15条；不同卡的prompt并非完全配对。')
    md('difference','## 降速幅度因批次而异，不能统一套一个百分比\n\nSeaCache 0.38受影响最大，为15.27%；其余档位在10%–13%附近。SeaCache 0.55各卡实际full-call数量略有不同，直接比较原始均值只得到9.55%，控制完整动作序列后为10.52%。下表保留精确除数和不确定区间，以免将工作量差异误算为GPU降速。')
    blocks.extend([dict(id='slowdown-chart',type='chart',chartId='slowdown'),dict(id='factor-table',type='table',tableId='factors')])
    md('components',f"## 额外耗时来自DiT，三张参照卡表现接近\n\n七批次中，匹配后的DiT额外耗时占总generate额外耗时 **99.70%–101.63%**；略超过100%表示其余环节有小幅反向抵消。非DiT平均差异仅−1.74至+0.14秒。GPU1/2/3的匹配均值最大卡间差为 **{validation['max_peer_spread_pct']:.2f}%**，显著小于GPU0偏差。\n\n由此可把计时异常主要定位到DiT执行速度。热降频不改变实际full/reuse调用数与现有DiT TFLOPs统计；本次不修正运算量。组件结果以本文数值说明，避免在接近100%的比例上使用易误导的堆叠图。")
    md('variation','## 同一批次也会漂移：TeaCache 0.68从约8%升至约18%\n\n该档GPU0前10条耗时增加8.44%，后10条增加17.61%。图中每箱5条，各卡按自身运行顺序划分；其参照为该批次所有健康卡匹配均值。**13.11%是50条的平均系数，不能视作每条视频的真实降速。**\n\n95%区间来自每GPU独立循环分块bootstrap：块长5、2000次、seed=20260906。它描述本批次内的采样波动，不覆盖未来温度状态或因果识别误差；各批次去掉每卡首条后结论不变。')
    blocks.append(dict(id='temporal-chart',type='chart',chartId='temporal'))
    md('crosscheck','## 相同prompt复跑支持GPU0偏慢，独立SQL复算通过\n\n11条近期MagCache baseline在GPU1/2/3均有独立复跑。其中历史GPU0上的4条，相对新三卡同prompt均值多耗时13.00%；历史健康卡上的另7条仅差−0.07%。用后者控制日期差异后约为13.08%。这组交叉检查支持GPU0偏慢，但只有4条GPU0 prompt，不能替代完整50条的批次估计。\n\n独立SQLite从1400条CSV重新分层聚合，复现全部七个系数、组件分解和修正预览；1550个来源文件的SHA保持一致。复核notebook的代码已在wan2.2中顺序执行；环境缺少Jupyter内核，未声称完成Jupyter内核执行。')
    md('thermal','## 温度记录支持热降频解释，仍缺逐视频遥测\n\n同期保存的会话记录在9月2日、9月4日观察到GPU0约91–92°C及SW thermal slowdown；与本次DiT执行偏慢一致。正式结果目录未发现连续逐视频的温度、SM频率与限频标志记录，因此可以确认批次中的系统性计时偏差，不能仅凭这些汇总数字断言每条额外秒数都由温度造成。', 'raw')
    md('correction','## 修正预览：baseline与candidate分别处理\n\n估算公式为 **GPU0修正秒数=原秒数/r**，GPU1/2/3维持实测；baseline与candidate各用自己的批次系数，再计算ratio-of-sums加速比。这样共享200条baseline的均值由1017.20秒估计降至990.56秒；11条标定子集由1034.83秒估计降至995.40秒。\n\n原始计时和现有正式报告未覆盖；correction_factors.json与correction_preview.csv已保存。本次系数只适用于列明的七个冻结批次。后续计时修正应单独标注“归一化估算”，不可冒充重新实测，也不可把本次系数推广到其他日期或模型。')
    blocks.append(dict(id='preview-table',type='table',tableId='preview'))
    artifact=dict(surface='report',manifest=dict(version=1,surface='report',title=title,description='Batch-specific timing audit with exact computation-path matching',generatedAt=s['created_at'],charts=charts,tables=tables,sources=sources,blocks=blocks),snapshot=dict(version=1,status='ready',generatedAt=s['created_at'],datasets={'estimates':estimates,'temporal':temporal,'preview':s['correction_preview']}),sources=sources)
    materialize_chart_queries(out,artifact)
    dump(out/'artifact.json',artifact)
    (out/'source_notes.md').write_text('''# Evidence and report contracts

Technical audience: the user requests measurement bias diagnosis to support experiment timing correction. Structure: title and technical summary, metric/scope definitions, batch comparison, component attribution, temporal uncertainty, independent validation, thermal evidence gap, correction preview and implications. Further questions are merged into the thermal/uncertainty sections because continuous historical telemetry is absent. Source inventory is native, not a duplicate narrative appendix.

Chart 1 contract: batch grain, exact-schedule healthy reference, complete generate inflation in percentage points (10 means 10%), zero-origin bars; seven distinct conditions, single series, exact values and bootstrap CI in the adjacent table. Chart 2 contract: TeaCache 0.68 only, four GPU series, ordinal five-video windows, shared whole-batch matched reference, not simultaneous calendar windows. Snapshot preserves matching sample counts, seconds, intervals and adjacent metrics beyond visible encodings. Both charts use the canonical report reader. Supporting tables serve exact coefficient/speedup lookup. Component and same-prompt findings use inline quantities because their small count and causal qualifications are clearer in prose.

Data lineage: source_manifest.json records exact raw file identities and SHA256; videos.csv is one row per measured cohort/video; verify.sql independently reconstructs reference strata and coefficients from the CSV. No SQL database existed upstream; the independent query ran on an in-memory SQLite videos table populated by the notebook. No raw timing file was mutated. Intervals are within-batch circular block-bootstrap descriptions, not calibrated causal or future-temperature confidence bounds.

Notebook execution: wan2.2 lacks nbformat, nbclient and ipykernel. build_evidence.py emits a valid notebook-format JSON companion and executes its actual code cells top-to-bottom using Python exec with captured stdout. Jupyter-kernel execution is unavailable and is not claimed. No dependencies were installed.
''')
    print('Wrote verified notebook, VALIDATION.json and canonical artifact.json:',out)


if __name__ == '__main__':
    main()
