#!/usr/bin/env python3
"""Build the canonical technical-report artifact for validated benchmark results."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ours4wan21.contracts import dump, sha256


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in list(row.items()):
            if value == '':
                row[key] = None
            elif value in ('True', 'False'):
                row[key] = value == 'True'
            else:
                try:
                    number = float(value)
                    row[key] = int(number) if number.is_integer() else number
                except (TypeError, ValueError):
                    pass
    return rows


def register_rows(connection: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    fields = list(rows[0])
    kinds = []
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        if values and all(isinstance(value, bool) for value in values):
            kinds.append('INTEGER')
        elif values and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            kinds.append('INTEGER')
        elif values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            kinds.append('REAL')
        else:
            kinds.append('TEXT')
    connection.execute(f'CREATE TABLE "{table}" (' + ', '.join(
        f'"{field}" {kind}' for field, kind in zip(fields, kinds)) + ')')
    placeholders = ', '.join('?' for _ in fields)
    connection.executemany(f'INSERT INTO "{table}" VALUES ({placeholders})',
                           [[row[field] for field in fields] for row in rows])


def query_rows(connection: sqlite3.Connection, sql: str) -> list[dict]:
    cursor = connection.execute(sql)
    fields = [item[0] for item in cursor.description]
    return [dict(zip(fields, row)) for row in cursor.fetchall()]


def source(source_id: str, label: str, path: str, description: str,
           definitions: list[str], sql: str, tables_used: list[str]) -> dict:
    return dict(id=source_id, label=label, path=path, query=dict(
        engine='SQLite in-memory audit view', language='SQL', sql=sql,
        id=f'{source_id}_query', description=description, tables_used=tables_used,
        filters=['12 frozen modes', 'targets 1.8x, 2.4x, 3.0x',
                 'five frozen VBench200 prompts', 'same-physical-GPU matched baseline'],
        metric_definitions=definitions))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root.resolve(strict=True)
    analysis = root / 'analysis'
    validation = json.loads((analysis / 'VALIDATION.json').read_text())
    complete = json.loads((analysis / 'COMPLETE.json').read_text())
    if (validation.get('status') != 'pass' or complete.get('status') != 'complete'
            or sha256(analysis / 'results_long.csv') != complete['results_sha256']):
        raise ValueError('analysis did not pass its fail-closed validation')
    results = csv_rows(analysis / 'results_long.csv')
    mapping = csv_rows(analysis / 'target_k_mapping.csv')
    if len(results) != 36:
        raise ValueError('report requires all 36 result rows')

    connection = sqlite3.connect(':memory:')
    register_rows(connection, 'results', results)
    register_rows(connection, 'target_k_mapping', mapping)
    validation_record = [{key: validation[key] for key in (
        'status', 'modes', 'targets', 'conditions', 'prompt_count',
        'frames_per_video', 'official_full_vbench_score')}]
    register_rows(connection, 'validation_record', validation_record)
    results_sql = 'SELECT * FROM results ORDER BY target_speedup, label'
    mapping_sql = 'SELECT * FROM target_k_mapping ORDER BY target_speedup'
    overhead_sql = ('SELECT label, kind, AVG(latent_feature_wall_seconds_mean) AS '
        'latent_feature_wall_seconds_mean, AVG(predictor_decision_seconds_mean) AS '
        'predictor_decision_seconds_mean FROM results GROUP BY label, kind ORDER BY label')
    headline_sql = ('SELECT COUNT(*) AS conditions, COUNT(DISTINCT mode) AS modes, '
        'COUNT(DISTINCT target_speedup) AS targets, MAX(target_relative_error) AS '
        'max_target_relative_error, AVG(psnr_rgb_db) AS mean_psnr_rgb_db, '
        'AVG(vbench_custom_score) AS mean_vbench_custom_score FROM results')
    validation_sql = ('SELECT status, modes, targets, conditions, prompt_count AS prompts, '
        'frames_per_video, official_full_vbench_score FROM validation_record')
    results = query_rows(connection, results_sql)
    mapping = query_rows(connection, mapping_sql)
    mode_overhead = query_rows(connection, overhead_sql)
    headline = query_rows(connection, headline_sql)
    validation_headline = query_rows(connection, validation_sql)
    connection.close()
    max_error = float(headline[0]['max_target_relative_error'])
    mean_psnr = float(headline[0]['mean_psnr_rgb_db'])
    score_vbench = validation.get('vbench_enabled', True)
    mean_vbench = float(headline[0]['mean_vbench_custom_score']) if score_vbench else None
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    definitions = [
        'Achieved latency speedup: sum of native-baseline complete generate wall seconds divided by sum of candidate complete generate wall seconds over the same five prompts and physical GPU.',
        'Target relative error: absolute achieved-minus-requested speedup divided by requested speedup.',
        'PSNR/SSIM/LPIPS: RGB per-frame full-reference metrics aggregated within video and then equally across five videos.',
        'VBench custom score: unweighted arithmetic mean of ten raw official custom-input dimension implementations; not an official full VBench or VBench200 aggregate.',
        'DiT TFLOPs: estimated operation count from Calflops-observed modules plus analytic dense attention, using actual full/reuse calls; T5 and VAE are separate.',
        'Predictor TFLOPs include the FP32 policy MLP only. Latent feature FFT/quantile/reduction operations have measured wall time but no TFLOPs estimate.',
    ]
    sources = [
        source('results_source', 'Validated 36-condition results', 'results_long.csv',
               'One aggregate row per state mode and requested speed target.', definitions,
               results_sql, ['results_long.csv']),
        source('headline_source', 'Aggregate benchmark headline', 'results_long.csv',
               'Counts and descriptive aggregates over the validated 36-condition matrix.', definitions,
               headline_sql, ['results_long.csv']),
        source('overhead_source', 'Per-mode online-overhead aggregate', 'results_long.csv',
               'Three-target mean feature and predictor wall time for each state mode.', definitions,
               overhead_sql, ['results_long.csv']),
        source('validation_source', 'Fail-closed benchmark validation', 'VALIDATION.json',
               'Completeness, pairing, protocol, checkpoint, calibration and quality checks.', definitions,
               validation_sql, ['VALIDATION.json']),
        source('mapping_source', 'Prior-calibrated fixed target-to-K mapping',
               'target_k_mapping.csv', 'Three fixed K values derived from the saved SeaCache branch-reuse-count fit.', definitions,
               mapping_sql, ['target_k_mapping.csv']),
    ]
    charts = [
        dict(id='target_chart', title='Requested and achieved complete-generation speedup',
             subtitle='36 matched conditions; diagonal agreement is assessed numerically in the table',
             type='scatter', intent='relationship', dataset='results', sourceId='results_source',
             palette=dict(kind='categorical', name='blue-orange'), legend=dict(position='bottom'),
             encodings=dict(
                 x=dict(field='target_speedup', type='quantitative', label='Requested speedup (×)'),
                 y=dict(field='achieved_latency_speedup', type='quantitative', label='Achieved speedup (×)'),
                 color=dict(field='kind', type='nominal', label='Group type'),
                 tooltip=[dict(field='label', type='nominal', label='Mode'),
                          dict(field='skip_budget', type='quantitative', label='Exact K'),
                          dict(field='target_relative_error', type='quantitative', label='Relative error')]),
             comparisonContext=dict(grain='one mode-target condition', unit='speedup ratio')),
        dict(id='psnr_tradeoff', title='PSNR versus achieved complete-generation speedup',
             subtitle='Five-prompt RGB full-reference mean for each of 36 conditions',
             type='scatter', intent='relationship', dataset='results', sourceId='results_source',
             palette=dict(kind='categorical', name='blue-orange-neutral'), legend=dict(position='bottom'),
             encodings=dict(
                 x=dict(field='achieved_latency_speedup', type='quantitative', label='Achieved speedup (×)'),
                 y=dict(field='psnr_rgb_db', type='quantitative', label='PSNR (dB)'),
                 color=dict(field='target_label', type='nominal', label='Requested target'),
                 tooltip=[dict(field='label', type='nominal', label='Mode'),
                          dict(field='ssim_rgb', type='quantitative', label='SSIM'),
                          dict(field='lpips_alex_v0_1_spatial', type='quantitative', label='LPIPS')]),
             comparisonContext=dict(grain='one mode-target condition', unit='dB versus speedup')),
        dict(id='lpips_tradeoff', title='LPIPS versus achieved complete-generation speedup',
             subtitle='Lower LPIPS is closer to the matched native baseline',
             type='scatter', intent='relationship', dataset='results', sourceId='results_source',
             palette=dict(kind='categorical', name='blue-orange-neutral'), legend=dict(position='bottom'),
             encodings=dict(
                 x=dict(field='achieved_latency_speedup', type='quantitative', label='Achieved speedup (×)'),
                 y=dict(field='lpips_alex_v0_1_spatial', type='quantitative', label='AlexNet LPIPS'),
                 color=dict(field='target_label', type='nominal', label='Requested target'),
                 tooltip=[dict(field='label', type='nominal', label='Mode'),
                          dict(field='psnr_rgb_db', type='quantitative', label='PSNR (dB)'),
                          dict(field='ssim_rgb', type='quantitative', label='SSIM')]),
             comparisonContext=dict(grain='one mode-target condition', unit='LPIPS versus speedup')),
        dict(id='vbench_tradeoff', title='VBench custom diagnostic versus achieved speedup',
             subtitle='Non-official raw mean of ten custom-input dimensions on five prompts',
             type='scatter', intent='relationship', dataset='results', sourceId='results_source',
             palette=dict(kind='categorical', name='blue-orange-neutral'), legend=dict(position='bottom'),
             encodings=dict(
                 x=dict(field='achieved_latency_speedup', type='quantitative', label='Achieved speedup (×)'),
                 y=dict(field='vbench_custom_score', type='quantitative', label='VBench custom raw mean'),
                 color=dict(field='target_label', type='nominal', label='Requested target'),
                 tooltip=[dict(field='label', type='nominal', label='Mode'),
                          dict(field='vbench_custom_delta', type='quantitative', label='Delta vs matched baseline')]),
             comparisonContext=dict(grain='one mode-target condition', unit='raw score versus speedup')),
        dict(id='feature_overhead', title='Measured latent-feature wall time by state mode',
             subtitle='Mean per video across the three selected operating points; nested in complete latency',
             type='bar', intent='comparison', dataset='mode_overhead', sourceId='overhead_source',
             palette=dict(kind='sequential', name='blue'), labels=dict(values='auto'),
             encodings=dict(
                 x=dict(field='label', type='nominal', label='State mode'),
                 y=dict(field='latent_feature_wall_seconds_mean', type='quantitative', label='Seconds per video'),
                 tooltip=[dict(field='predictor_decision_seconds_mean', type='quantitative', label='Predictor decision seconds')]),
             comparisonContext=dict(grain='one state mode', unit='seconds per video')),
    ]
    cards = [
        dict(id='conditions_card', description='All requested mode-target cells passed artifact checks.',
             dataset='validation_headline', sourceId='validation_source', metrics=[
                 dict(label='Validated conditions', field='conditions', format='number'),
                 dict(label='Modes', field='modes', format='number'),
                 dict(label='Targets', field='targets', format='number')]),
        dict(id='error_card', description='Largest absolute requested-versus-achieved speedup error.',
             dataset='headline', sourceId='headline_source', metrics=[
                 dict(label='Maximum target error', field='max_target_relative_error', format='percent')]),
        dict(id='prompt_card', description='Frozen maximum-metadata-coverage subset; one seed per prompt.',
             dataset='validation_headline', sourceId='validation_source', metrics=[
                 dict(label='Prompts per condition', field='prompts', format='number')]),
        dict(id='psnr_card', description='Descriptive mean across 36 conditions; use target-level evidence below.',
             dataset='headline', sourceId='headline_source', metrics=[
                 dict(label='Mean PSNR', field='mean_psnr_rgb_db', format='number', unit='dB')]),
        dict(id='vbench_card', description='Non-official custom-input raw mean, not a leaderboard score.',
             dataset='headline', sourceId='headline_source', metrics=[
                 dict(label='Mean VBench custom', field='mean_vbench_custom_score', format='number')]),
    ]
    table = dict(id='results_table', title='Exact mode-by-target benchmark results',
        subtitle='36 conditions; sorted by requested target then diagnostic VBench score',
        dataset='results', sourceId='results_source', density='dense',
        defaultSort=dict(field='target_speedup', direction='asc'), columns=[
            dict(field='label', label='Mode', type='text'),
            dict(field='kind', label='Type', type='text'),
            dict(field='target_speedup', label='Target ×', format='number'),
            dict(field='skip_budget', label='K', format='number'),
            dict(field='achieved_latency_speedup', label='Actual ×', format='number'),
            dict(field='target_relative_error', label='Target error', format='percent'),
            dict(field='candidate_generate_seconds_mean', label='Generate s/video', format='number'),
            dict(field='candidate_dit_tflops_mean', label='DiT TFLOPs/video', format='number'),
            dict(field='psnr_rgb_db', label='PSNR dB', format='number'),
            dict(field='ssim_rgb', label='SSIM', format='number'),
            dict(field='lpips_alex_v0_1_spatial', label='LPIPS', format='number'),
            dict(field='vbench_custom_score', label='VBench custom', format='number'),
            dict(field='latent_feature_wall_seconds_mean', label='Feature s/video', format='number'),
            dict(field='selected_epoch', label='Checkpoint epoch', format='number')])
    components_table = dict(id='components_table', title='Component timing and operation counts',
        subtitle='Per-video means; CUDA component spans are not additive to complete generate wall time',
        dataset='results', sourceId='results_source', density='dense',
        defaultSort=dict(field='target_speedup', direction='asc'), columns=[
            dict(field='label', label='Mode', type='text'),
            dict(field='target_speedup', label='Target ×', format='number'),
            dict(field='candidate_generate_seconds_mean', label='Generate s', format='number'),
            dict(field='candidate_t5_cuda_seconds_mean', label='T5 CUDA s', format='number'),
            dict(field='candidate_dit_cuda_seconds_mean', label='DiT CUDA s', format='number'),
            dict(field='candidate_vae_cuda_seconds_mean', label='VAE CUDA s', format='number'),
            dict(field='candidate_dit_tflops_mean', label='DiT TFLOPs', format='number'),
            dict(field='t5_tflops_per_video', label='T5 TFLOPs', format='number'),
            dict(field='vae_tflops_per_video', label='VAE TFLOPs', format='number'),
            dict(field='predictor_tflops_mean', label='Predictor TFLOPs', format='number'),
            dict(field='predictor_decision_seconds_mean', label='Predictor wall s', format='number'),
            dict(field='latent_feature_wall_seconds_mean', label='Feature wall s', format='number')])

    title = 'Ours4Wan21 五提示词三档加速实验报告'
    blocks = [
        dict(id='title', type='markdown', body=f'# {title}'),
        dict(id='summary', type='markdown', sourceId='validation_source', body=(
            '## 技术摘要\n\n'
            f'- **完整性：** 12个state mode在1.8×、2.4×、3.0×三档均完成，共36个条件、每条件5个视频。\n'
            f'- **固定映射：** 既有标定给出1.8×→K23、2.4×→K29、3.0×→K35；按用户要求不做自适应补测。36个条件的最大名义目标相对误差为{max_error:.2%}。\n'
            f'- **质量读数：** 36条件平均PSNR为{mean_psnr:.3f} dB；'
            + (f'平均VBench custom原始均值为{mean_vbench:.4f}。' if score_vbench else '按用户要求未运行VBench评分。')
            + '应按目标档位和同GPU baseline逐组比较。\n'
            '- **结论边界：** 五提示词仅是小样本诊断集；VBench值不是完整VBench/VBench200官方分数，也没有多seed不确定性。')),
        dict(id='cards', type='metric-strip', cardIds=[card['id'] for card in cards]),
        dict(id='calibration_text', type='markdown', sourceId='mapping_source', body=(
            '## 三个K直接来自既有skip-count标定，不再自适应补测\n\n'
            '既有SeaCache拟合以100次CFG分支DiT调用中的reuse数计数，除以2换算成50-step Exact-K，并对三个目标取最近整数，得到K23/K29/K35。所有mode直接跑这三点；图中36点的完整generate实测值和目标偏差仅用于汇报，不会触发新的K。')),
        dict(id='target_chart_block', type='chart', chartId='target_chart'),
        dict(id='psnr_text', type='markdown', sourceId='results_source', body=(
            '## PSNR/SSIM/LPIPS共同刻画相对原生视频的失真\n\n'
            'PSNR和SSIM越高越接近baseline，LPIPS越低越接近baseline。所有三项均逐帧RGB比较且不重采样、不裁切；散点用于读取速度—质量权衡，不能把跨目标绝对值混成单一赢家。')),
        dict(id='psnr_chart_block', type='chart', chartId='psnr_tradeoff'),
        dict(id='lpips_chart_block', type='chart', chartId='lpips_tradeoff'),
        dict(id='vbench_text', type='markdown', sourceId='results_source', body=(
            '## VBench只作为五提示词custom-input诊断\n\n'
            '五条prompt无法覆盖官方16维。这里调用官方custom-input可用的10个维度实现，并取未加权原始均值；六个依赖benchmark辅助标签的维度不在该分数中。因此它适合内部配对比较，不可作为VBench排行榜或完整VBench200成绩。')),
        dict(id='vbench_chart_block', type='chart', chartId='vbench_tradeoff'),
        dict(id='overhead_text', type='markdown', sourceId='results_source', body=(
            '## 高维latent特征的在线成本已进入完整延迟\n\n'
            '特征提取wall time和predictor决策时间都嵌套在完整generate中，不应再次相加。Predictor MLP有独立TFLOPs；FFT、quantile与reduction目前仅报告实测wall time，TFLOPs明确留空。')),
        dict(id='overhead_chart_block', type='chart', chartId='feature_overhead'),
        dict(id='exact_text', type='markdown', body=(
            '## 36个条件的精确指标\n\n'
            '表格同时保留目标、Exact-K、完整延迟、DiT运算量、三项full-reference质量、VBench custom诊断和checkpoint epoch。T5、DiT、VAE分项CUDA时间及T5/VAE TFLOPs保存在源CSV。')),
        dict(id='results_table_block', type='table', tableId='results_table'),
        dict(id='components_text', type='markdown', sourceId='results_source', body=(
            '## T5、DiT、VAE与策略开销分项\n\n'
            '下表按条件列出完整generate wall time、三个模型组件的CUDA时间和分别估算的TFLOPs。Predictor MLP另外计数；latent特征只报告wall time，因为FFT、quantile与reduction尚无一致的算子级TFLOPs合同。组件CUDA事件嵌套于generate，不能与完整时间再次相加。')),
        dict(id='components_table_block', type='table', tableId='components_table'),
        dict(id='scope', type='markdown', sourceId='validation_source', body=(
            '## 数据、协议与指标口径\n\n'
            '推理固定为Wan2.1-T2V-1.3B、832×480、81帧、16fps、50-step UniPC、shift=5、CFG=5、seed=42、BF16、单卡resident、无offload。每组checkpoint仅由e300–e400验证状态的双侧actor一致率和归一化Q漂移选择，VBench/test不参与选择。Headline延迟是5个prompt完整`pipeline.generate`的ratio-of-sums。')),
        dict(id='method', type='markdown', sourceId='mapping_source', body=(
            '## 实验设计与稳健性检查\n\n'
            '每个mode固定到一张物理GPU，并只与该GPU生成的native baseline配对。Exact-K确保cond/uncond共享每步动作且两分支各自恰好reuse K步。K在运行前已经冻结；真实完整generate加速比按ratio-of-sums汇报，不用重新拟合或改写K。')),
        dict(id='limitations', type='markdown', body=(
            '## 限制与不确定性\n\n'
            '- 五prompt、单seed不足以估计总体质量或置信区间。\n'
            '- 各GPU虽采用本卡baseline归一化，但硬件时序噪声仍会影响目标误差。\n'
            '- VBench custom原始均值不是官方加权总分，且缺少六个辅助标签维度。\n'
            '- Feature计算未建立算子级TFLOPs模型；其影响已包含在完整延迟和wall-time分项中。')),
        dict(id='next', type='markdown', body=(
            '## 建议的后续验收\n\n'
            '若五prompt结果要升级为对外质量结论，应在预先选定的候选组上运行完整VBench200，并增加多seed重复；在此之前只把本报告用于筛选和定位速度—质量—特征开销的明显差异。')),
        dict(id='questions', type='markdown', body=(
            '## 仍需回答的问题\n\n'
            '- 五prompt上领先的mode在完整200 prompt和多seed下是否保持排序？\n'
            '- 哪些latent特征在质量收益与在线特征成本之间形成稳定Pareto优势？')),
    ]
    if not score_vbench:
        charts = [chart for chart in charts if chart['id'] != 'vbench_tradeoff']
        cards = [card for card in cards if card['id'] != 'vbench_card']
        table['columns'] = [column for column in table['columns'] if column['field'] != 'vbench_custom_score']
        table['subtitle'] = '36 conditions; sorted by requested target then mode label'
        blocks = [block for block in blocks if block['id'] not in ('vbench_text', 'vbench_chart_block')]
        for block in blocks:
            if block['id'] == 'cards':
                block['cardIds'] = [card['id'] for card in cards]
            if 'body' in block:
                block['body'] = block['body'].replace(
                    'VBench值不是完整VBench/VBench200官方分数，也没有多seed不确定性。',
                    '本次按用户要求跳过VBench评分，也没有多seed不确定性。').replace(
                    '、VBench custom诊断和checkpoint epoch', '和checkpoint epoch').replace(
                    '- VBench custom原始均值不是官方加权总分，且缺少六个辅助标签维度。',
                    '- VBench评分按用户要求未测量；不能由PSNR/SSIM/LPIPS推断VBench分数。')
        definitions[3] = 'VBench scoring skipped at user request; missing values indicate not measured, never zero.'
    artifact = dict(surface='report', manifest=dict(version=1, surface='report', title=title,
        description='十二个Ours4Wan21 checkpoint在三档目标速度下的配对技术评测。',
        generatedAt=now, cards=cards, charts=charts, tables=[table, components_table], sources=sources,
        blocks=blocks), snapshot=dict(version=1, generatedAt=now, status='ready',
        datasets=dict(results=results, target_k_mapping=mapping, mode_overhead=mode_overhead,
                      headline=headline, validation_headline=validation_headline), accessIssues=[]), sources=sources)
    dump(analysis / 'artifact.json', artifact)
    chart_map = [dict(section=chart['id'], question=chart.get('title'), family=chart['intent'],
                      type=chart['type'], dataset=chart['dataset'], source=chart['sourceId'])
                 for chart in charts]
    dump(analysis / 'chart_map.json', chart_map)
    print(json.dumps(dict(status='ready', blocks=len(blocks), charts=len(charts),
                          cards=len(cards), tables=2), indent=2))


if __name__ == '__main__':
    main()
