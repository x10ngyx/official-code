#!/usr/bin/env python3
"""Build the canonical portable-report artifact from a validated twelve-group readout."""
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
            if key in ('mode', 'label', 'kind', 'series'):
                continue
            try:
                number = float(value)
                row[key] = int(number) if number.is_integer() and key not in (
                    'selection_gate', 'selected_min_actor_agreement', 'selected_mean_actor_agreement') else number
            except (TypeError, ValueError):
                pass
    return rows


def register_rows(connection: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    """Register reviewed rows so the report provenance SQL is also the executed query."""
    fields = list(rows[0])
    types = []
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        if values and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            types.append('INTEGER')
        elif values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            types.append('REAL')
        else:
            types.append('TEXT')
    connection.execute(f'CREATE TABLE "{table}" (' + ', '.join(
        f'"{field}" {kind}' for field, kind in zip(fields, types)) + ')')
    placeholders = ', '.join('?' for _ in fields)
    connection.executemany(f'INSERT INTO "{table}" VALUES ({placeholders})',
                           [[row[field] for field in fields] for row in rows])


def query_rows(connection: sqlite3.Connection, sql: str) -> list[dict]:
    cursor = connection.execute(sql)
    fields = [item[0] for item in cursor.description]
    return [dict(zip(fields, row)) for row in cursor.fetchall()]


def source(source_id: str, label: str, path: str, description: str,
           definitions: list[str], sql: str, tables_used: list[str] | None = None) -> dict:
    return dict(id=source_id, label=label, path=path, query=dict(
        engine='SQLite in-memory audit view', language='SQL', sql=sql,
        id=f'{source_id}_query', description=description,
        tables_used=tables_used or [path], filters=['12 frozen state modes', 'epochs 1-400',
        'shared 3000-trajectory selection; original prompt split retained'],
        metric_definitions=definitions))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readout-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.readout_dir.resolve(strict=True)
    complete = json.loads((root / 'COMPLETE.json').read_text())
    validation = json.loads((root / 'VALIDATION.json').read_text())
    if complete.get('status') != 'complete' or validation.get('status') != 'pass':
        raise ValueError('training readout has not passed validation')
    if sha256(root / 'suite_summary.csv') != complete['summary_sha256']:
        raise ValueError('suite summary changed after validation')
    summary_rows = csv_rows(root / 'suite_summary.csv')
    aggregate_rows = csv_rows(root / 'epoch_aggregate.csv')
    if len(summary_rows) != 12 or len(aggregate_rows) != 1200:
        raise ValueError('unexpected report dataset size')
    connection = sqlite3.connect(':memory:')
    register_rows(connection, 'suite_summary', summary_rows)
    register_rows(connection, 'epoch_aggregate', aggregate_rows)
    summary_sql = 'SELECT * FROM suite_summary'
    aggregate_sql = 'SELECT * FROM epoch_aggregate ORDER BY epoch, series'
    checkpoint_sql = ('SELECT COUNT(*) AS selected_groups, MIN(selected_epoch) AS selected_epoch_min, '
        'MAX(selected_epoch) AS selected_epoch_max, 101 AS selection_window_checkpoints '
        'FROM suite_summary')
    runtime_sql = ('SELECT SUM(elapsed_hours) AS total_training_gpu_hours, '
        'MAX(elapsed_hours) AS longest_single_group_hours FROM suite_summary')
    summary = query_rows(connection, summary_sql)
    aggregate = query_rows(connection, aggregate_sql)
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    sea7 = next(row for row in summary if row['mode'] == 'sea7')
    best = min(summary, key=lambda row: row['tail50_val_pi_mean'])
    delta = (best['tail50_val_pi_mean'] / sea7['tail50_val_pi_mean'] - 1) * 100
    selected_epochs = [row['selected_epoch'] for row in summary]
    agreements = [row['selected_min_actor_agreement'] for row in summary]
    headline_rows = [dict(completed_groups=validation['groups'], planned_groups=12,
        epochs_per_group=validation['epochs_per_group'],
        total_epoch_records=validation['total_epoch_records'], trajectories=validation['trajectories'],
        train_trajectories=validation['trajectory_split_counts']['train'],
        val_trajectories=validation['trajectory_split_counts']['val'],
        test_trajectories=validation['trajectory_split_counts']['test'])]
    register_rows(connection, 'headline', headline_rows)
    headline_sql = 'SELECT * FROM headline'
    headline = query_rows(connection, headline_sql)
    selection_headline = query_rows(connection, checkpoint_sql)
    runtime_headline = query_rows(connection, runtime_sql)
    connection.close()

    definitions = [
        'Validation actor loss: advantage-weighted behavior-cloning loss on discretionary validation rows only; lower is descriptive, not rollout quality.',
        'Validation Q loss: sum of the two online-Q MSE losses against reward plus gamma times next-state V.',
        'Validation V loss: expectile loss against online Q; train V instead uses target Q, so train/validation V are not a direct gap.',
        'Offline policy skip rate: greedy actor action fraction on recorded validation states; not closed-loop reuse or speedup.',
        'Selected checkpoint: e301-e399 candidate chosen by the two-sided e300-e400 actor-agreement gate, then minimum adjacent Q drift normalized by Q IQR.',
    ]
    sources = [
        source('validation_source', 'Cross-group validation record', 'VALIDATION.json',
               'Fail-closed checks for run, epoch, selection, split, source and checkpoint completeness.',
               definitions, headline_sql),
        source('summary_source', 'Twelve-group training summary', 'suite_summary.csv',
               'One reviewed row per state mode with late-training, final and checkpoint-stability metrics.',
               definitions, summary_sql),
        source('epoch_source', 'Epoch-level aggregate trends', 'epoch_aggregate.csv',
               'Epoch 1-400 controls plus median of ten feature groups for validation trend charts.',
               definitions, aggregate_sql),
        source('checkpoint_source', 'Selected checkpoint summary', 'suite_summary.csv',
               'Selected epoch, SHA256 and stability fields for all twelve state modes.',
               definitions, checkpoint_sql),
        source('runtime_source', 'Training elapsed-time summary', 'suite_summary.csv',
               'Sum of single-GPU training elapsed hours and longest individual group runtime.',
               definitions, runtime_sql),
    ]
    title = 'Wan2.1 十二组 400-Epoch 训练指标报告'
    charts = [
        dict(id='tail_actor_chart', title='末 50 Epoch 验证集 Actor Loss 均值',
             subtitle='12组共享同一selection；数值较低只代表离线actor目标较低',
             type='bar', intent='comparison', question='各组在训练末50轮的验证actor目标如何比较？',
             rationale='单指标跨12个离散组的精确横向比较最适合零基线柱状图。',
             dataset='suite_summary', sourceId='summary_source', valueFormat='number',
             palette=dict(kind='sequential', name='blue'), labels=dict(values='auto'),
             encodings=dict(x=dict(field='label', type='nominal', label='State mode'),
                            y=dict(field='tail50_val_pi_mean', type='quantitative', label='Mean validation actor loss'),
                            tooltip=[dict(field='tail50_val_pi_std', type='quantitative', label='50-epoch std'),
                                     dict(field='input_dim', type='quantitative', label='Input dimensions'),
                                     dict(field='selected_epoch', type='quantitative', label='Selected epoch')]),
             comparisonContext=dict(baseline='SEA7', grain='one row per state mode', unit='loss')),
        dict(id='actor_trend_chart', title='验证集 Actor Loss 训练轨迹',
             subtitle='Scalar5、SEA7与十个feature组的逐epoch中位数；feature带宽未作为误差区间绘制',
             type='line', intent='trend', question='对照组和feature组整体的actor目标如何随epoch变化？',
             rationale='400个有序epoch足以呈现训练形态，三条系列避免12线过度拥挤。',
             dataset='epoch_aggregate', sourceId='epoch_source', valueFormat='number',
             palette=dict(kind='categorical', name='blue-orange-neutral'),
             legend=dict(position='bottom', sort='spec'), labels=dict(values='endpoints'),
             encodings=dict(x=dict(field='epoch', type='quantitative', label='Epoch'),
                            y=dict(field='val_pi_loss', type='quantitative', label='Validation actor loss'),
                            color=dict(field='series', type='nominal', label='Series'),
                            tooltip=[dict(field='val_pi_loss_q25', type='quantitative', label='Feature/control q25'),
                                     dict(field='val_pi_loss_q75', type='quantitative', label='Feature/control q75'),
                                     dict(field='group_count', type='quantitative', label='Groups represented')]),
             comparisonContext=dict(grain='epoch by series', unit='loss')),
        dict(id='q_trend_chart', title='验证集 Q Loss 训练轨迹',
             subtitle='Scalar5、SEA7与十个feature组中位数；与actor loss分图避免混合量纲',
             type='line', intent='trend', question='对照组和feature组整体的Q拟合误差如何随epoch变化？',
             rationale='Q loss使用独立纵轴，400个epoch展示是否持续漂移或稳定。',
             dataset='epoch_aggregate', sourceId='epoch_source', valueFormat='number',
             palette=dict(kind='categorical', name='blue-orange-neutral'),
             legend=dict(position='bottom', sort='spec'), labels=dict(values='endpoints'),
             encodings=dict(x=dict(field='epoch', type='quantitative', label='Epoch'),
                            y=dict(field='val_q_loss', type='quantitative', label='Validation Q loss'),
                            color=dict(field='series', type='nominal', label='Series'),
                            tooltip=[dict(field='val_q_loss_q25', type='quantitative', label='Feature/control q25'),
                                     dict(field='val_q_loss_q75', type='quantitative', label='Feature/control q75'),
                                     dict(field='group_count', type='quantitative', label='Groups represented')]),
             comparisonContext=dict(grain='epoch by series', unit='loss')),
        dict(id='v_trend_chart', title='验证集 V Loss 训练轨迹',
             subtitle='仅比较validation online-Q expectile目标；不与使用target-Q的train V直接作gap',
             type='line', intent='trend', question='各类状态输入的validation V拟合误差如何随epoch变化？',
             rationale='V loss具有独立定义和量纲，单图展示可以保留其漂移而不误导train/val gap解释。',
             dataset='epoch_aggregate', sourceId='epoch_source', valueFormat='number',
             palette=dict(kind='categorical', name='blue-orange-neutral'),
             legend=dict(position='bottom', sort='spec'), labels=dict(values='endpoints'),
             encodings=dict(x=dict(field='epoch', type='quantitative', label='Epoch'),
                            y=dict(field='val_v_loss', type='quantitative', label='Validation V loss'),
                            color=dict(field='series', type='nominal', label='Series'),
                            tooltip=[dict(field='val_v_loss_q25', type='quantitative', label='Feature/control q25'),
                                     dict(field='val_v_loss_q75', type='quantitative', label='Feature/control q75'),
                                     dict(field='group_count', type='quantitative', label='Groups represented')]),
             comparisonContext=dict(grain='epoch by series', unit='loss')),
        dict(id='skip_trend_chart', title='离线 Greedy Policy Skip Rate 训练轨迹',
             subtitle='记录状态上的动作比例，只用于诊断policy变化；不是闭环reuse率或speedup',
             type='line', intent='trend', question='训练过程中离线greedy动作比例是否发生系统性漂移？',
             rationale='按epoch展示控制组和feature中位数，可识别策略动作倾向变化而不冒充闭环性能。',
             dataset='epoch_aggregate', sourceId='epoch_source', valueFormat='percent',
             palette=dict(kind='categorical', name='blue-orange-neutral'),
             legend=dict(position='bottom', sort='spec'), labels=dict(values='endpoints'),
             encodings=dict(x=dict(field='epoch', type='quantitative', label='Epoch'),
                            y=dict(field='val_skip_rate_policy', type='quantitative', label='Offline greedy skip rate'),
                            color=dict(field='series', type='nominal', label='Series'),
                            tooltip=[dict(field='val_skip_rate_policy_q25', type='quantitative', label='Feature/control q25'),
                                     dict(field='val_skip_rate_policy_q75', type='quantitative', label='Feature/control q75'),
                                     dict(field='group_count', type='quantitative', label='Groups represented')]),
             comparisonContext=dict(grain='epoch by series', unit='fraction')),
        dict(id='stability_chart', title='Post-300 Checkpoint 稳定性',
             subtitle='每点为一组；越靠左代表相邻Q变化越小，越靠上代表actor动作一致率越高',
             type='scatter', intent='relationship', question='正式选中checkpoint的actor一致性与Q漂移是否同时稳定？',
             rationale='两个连续稳定性指标在12个同粒度组上的关系适合散点图，并保留组标签。',
             dataset='suite_summary', sourceId='summary_source',
             palette=dict(kind='categorical', name='blue-orange'), legend=dict(position='bottom', sort='spec'),
             encodings=dict(x=dict(field='selected_mean_normalized_dq', type='quantitative', label='Mean adjacent ΔQ / Q IQR'),
                            y=dict(field='selected_min_actor_agreement', type='quantitative', label='Minimum two-sided actor agreement'),
                            color=dict(field='kind', type='nominal', label='Group type'),
                            label=dict(field='label', type='nominal', label='State mode'),
                            tooltip=[dict(field='selected_epoch', type='quantitative', label='Selected epoch'),
                                     dict(field='selected_val_pi_loss', type='quantitative', label='Selected validation actor loss'),
                                     dict(field='input_dim', type='quantitative', label='Input dimensions')]),
             comparisonContext=dict(grain='one selected checkpoint per state mode', unit='agreement vs normalized Q drift')),
    ]
    cards = [
        dict(id='completion_card', description='完成标记、400条epoch记录和400个checkpoint均通过。',
             dataset='headline', sourceId='validation_source', metrics=[
                 dict(label='完整训练组', field='completed_groups', format='number'),
                 dict(label='计划组数', field='planned_groups', format='number')]),
        dict(id='epochs_card', description='每组完整400 epoch；总计4800个组×epoch记录。',
             dataset='headline', sourceId='validation_source', metrics=[
                 dict(label='Epoch / 组', field='epochs_per_group', format='number'),
                 dict(label='总Epoch记录', field='total_epoch_records', format='number')]),
        dict(id='selection_card', description='12组共享同一冻结selection和原prompt split。',
             dataset='headline', sourceId='validation_source', metrics=[
                 dict(label='训练轨迹', field='trajectories', format='number'),
                 dict(label='Train', field='train_trajectories', format='number'),
                 dict(label='Val', field='val_trajectories', format='number'),
                 dict(label='Test', field='test_trajectories', format='number')]),
        dict(id='window_card', description='每组仅在e300-e400验证自由状态上做稳定性普查。',
             dataset='selection_headline', sourceId='checkpoint_source', metrics=[
                 dict(label='选择窗口Checkpoint', field='selection_window_checkpoints', format='number'),
                 dict(label='最早选中Epoch', field='selected_epoch_min', format='number'),
                 dict(label='最晚选中Epoch', field='selected_epoch_max', format='number')]),
        dict(id='runtime_card', description='单卡训练elapsed之和是GPU-hours；最长组不等于四卡编排总wall time。',
             dataset='runtime_headline', sourceId='runtime_source', metrics=[
                 dict(label='训练GPU-hours', field='total_training_gpu_hours', format='number'),
                 dict(label='最长单组小时', field='longest_single_group_hours', format='number')]),
    ]
    tables = [dict(id='summary_table', title='十二组训练与Checkpoint摘要',
        subtitle='按末50轮验证actor loss升序；精确值用于审计，不代表闭环质量排名',
        dataset='suite_summary', sourceId='summary_source', defaultSort=dict(field='tail50_val_pi_mean', direction='asc'),
        density='dense', columns=[
            dict(field='label', label='组', type='text'), dict(field='kind', label='类型', type='text'),
            dict(field='input_dim', label='输入维数', format='number'),
            dict(field='trainable_parameters', label='可训练参数', format='number'),
            dict(field='tail50_val_pi_mean', label='末50 Actor Loss', format='number'),
            dict(field='tail50_val_q_mean', label='末50 Q Loss', format='number'),
            dict(field='tail50_val_v_mean', label='末50 V Loss', format='number'),
            dict(field='tail50_val_skip_policy_mean', label='末50离线Skip率', format='percent'),
            dict(field='selected_epoch', label='选中Epoch', format='number'),
            dict(field='selected_min_actor_agreement', label='最小动作一致率', format='percent'),
            dict(field='selected_mean_normalized_dq', label='平均ΔQ/IQR', format='number'),
            dict(field='elapsed_hours', label='训练小时', format='number')]),
        dict(id='diagnostics_table', title='末段漂移、Actor Weight与离线动作诊断',
            subtitle='cap统计覆盖400 epoch；斜率按末50 epoch折算为每100 epoch变化量',
            dataset='suite_summary', sourceId='summary_source',
            defaultSort=dict(field='tail50_val_q_slope_per_100_epochs', direction='asc'),
            density='dense', columns=[
                dict(field='label', label='组', type='text'),
                dict(field='best_val_pi_epoch', label='Best Actor Epoch', format='number'),
                dict(field='best_val_pi_loss', label='Best Actor Loss', format='number'),
                dict(field='final_val_pi_loss', label='Final Actor Loss', format='number'),
                dict(field='tail50_val_q_slope_per_100_epochs', label='Q末段斜率/100e', format='number'),
                dict(field='tail50_val_v_slope_per_100_epochs', label='V末段斜率/100e', format='number'),
                dict(field='logged_val_skip_rate', label='Logged Skip率', format='percent'),
                dict(field='tail50_val_skip_policy_mean', label='Policy Skip率', format='percent'),
                dict(field='train_actor_weight_cap_epochs', label='Train Cap Epochs', format='number'),
                dict(field='val_actor_weight_cap_epochs', label='Val Cap Epochs', format='number')])]
    blocks = [
        dict(id='title', type='markdown', body=f'# {title}'),
        dict(id='technical_summary', type='markdown', body=(
            '## 技术摘要\n\n'
            f'- **完整性通过：** 12/12组均完成400 epoch，共4800条组×epoch记录；全部核心指标有限，cache、source、split和checkpoint SHA验收通过。\n'
            f'- **离线actor目标的描述性最低值：** {best["label"]}末50轮验证actor loss均值为{best["tail50_val_pi_mean"]:.6g}，相对SEA7为{delta:+.2f}%。该结果不等于视频质量或推理速度优势。\n'
            f'- **稳定性选择已独立执行：** 12组选中epoch位于{min(selected_epochs)}–{max(selected_epochs)}，选中点两侧最小actor一致率范围为{min(agreements):.4f}–{max(agreements):.4f}；test与VBench均未参与选择。\n'
            '- **结论边界：** 每组只有seed42一次训练，且输入维数/参数量不同；本报告支持训练完整性和离线诊断比较，不支持显著性、因果或闭环性能结论。')),
        dict(id='metrics', type='metric-strip', cardIds=[card['id'] for card in cards]),
        dict(id='late_actor', type='markdown', sourceId='summary_source', body=(
            '## 末段Actor目标用于诊断，不用于直接选视频赢家\n\n'
            f'末50轮均值最低的是**{best["label"]}**（{best["tail50_val_pi_mean"]:.6g}），SEA7为{sea7["tail50_val_pi_mean"]:.6g}。'
            '柱图按同一验证集、同一损失定义比较；输入维度和模型第一层参数量差异仍会影响优化难度。')),
        dict(id='tail_actor_chart_block', type='chart', chartId='tail_actor_chart'),
        dict(id='actor_dynamics', type='markdown', sourceId='epoch_source', body=(
            '## Actor训练轨迹应与跨组离散比较一起阅读\n\n'
            '曲线保留两个对照组，并用十个feature组的逐epoch中位数表示整体形态；q25/q75仍保存在图表数据中。'
            '它能识别共同的训练阶段和漂移，但不会掩盖摘要表中的单组例外。')),
        dict(id='actor_trend_chart_block', type='chart', chartId='actor_trend_chart'),
        dict(id='q_dynamics', type='markdown', sourceId='epoch_source', body=(
            '## Q拟合误差单独展示，避免与Actor/V混合量纲\n\n'
            'Q loss的绝对值和末段斜率用于识别critic持续漂移或稳定；单纯变大不能证明策略失效，必须与动作一致率和归一化ΔQ同时判断。')),
        dict(id='q_trend_chart_block', type='chart', chartId='q_trend_chart'),
        dict(id='v_dynamics', type='markdown', sourceId='epoch_source', body=(
            '## Validation V Loss按其实际online-Q目标单独解释\n\n'
            'validation V loss可用于识别各组value网络的末段漂移；由于train V使用target Q，两者目标不同，报告不绘制伪装成普通泛化gap的train/validation叠加图。')),
        dict(id='v_trend_chart_block', type='chart', chartId='v_trend_chart'),
        dict(id='skip_dynamics', type='markdown', sourceId='epoch_source', body=(
            '## 离线动作比例揭示策略倾向，但不等于部署收益\n\n'
            'greedy skip率只在记录的validation状态上计算，可检查训练是否整体偏向reuse或recompute；真实Exact-K闭环动作由硬预算约束，部署speedup仍需生成实验。')),
        dict(id='skip_trend_chart_block', type='chart', chartId='skip_trend_chart'),
        dict(id='stability', type='markdown', sourceId='summary_source', body=(
            '## Post-300选择同时约束Actor动作和Critic变化\n\n'
            '正式选择先要求候选checkpoint左右两侧actor agreement通过0.96门槛（必要时按合同回退），再最小化平均ΔQ/IQR。'
            '散点越靠左上表示这两个离线稳定性条件同时更强，但仍不是收敛或视频质量证明。')),
        dict(id='stability_chart_block', type='chart', chartId='stability_chart'),
        dict(id='exact_values', type='markdown', body='## 精确值与模型规模\n\n下表保留12组的输入维数、可训练参数、末段指标、选择稳定性和耗时，便于逐组复核。'),
        dict(id='summary_table_block', type='table', tableId='summary_table'),
        dict(id='diagnostics_values', type='markdown', body=(
            '## 优化诊断补充\n\n'
            '下表补充best/final actor、末50轮Q/V斜率、logged与policy skip差异，以及actor weight触顶epoch数。完整逐epoch advantage、actor weight、自由/强制样本数和reward仍保存在每组`training_metrics.csv`与跨组`epoch_metrics_long.csv`。')),
        dict(id='diagnostics_table_block', type='table', tableId='diagnostics_table'),
        dict(id='scope', type='markdown', body=(
            '## 数据、指标与比较范围\n\n'
            '所有组使用同一份3000条随机SeaCache轨迹，原prompt级split保持为2400/300/300条轨迹；训练器仅用train拟合normalizer和更新参数，val逐epoch评估，test不参与训练或checkpoint选择。'
            'Actor loss只统计Exact-K自由状态；Q critic仍使用自由与强制行。logged/policy skip率是在离线记录状态上的动作比例，不是闭环reuse率。')),
        dict(id='methodology', type='markdown', sourceId='validation_source', body=(
            '## 方法与验收\n\n'
            '固定设置为400 epoch、batch 256、seed 42、3×256 MLP、AdamW 1e-4、IQL tau=.6/beta=1/gamma=1。'
            '跨组审计复核400条有限训练/验证记录、400个checkpoint、共同source与prompt split、cache SHA、final SHA和selected SHA。'
            '每组post-300普查覆盖e300–e400，候选仅为e301–e399。')),
        dict(id='limitations', type='markdown', body=(
            '## 限制与稳健性边界\n\n'
            '- 每个state mode只有一个seed，无法估计训练随机性或显著性。\n'
            '- Feature组输入维数不同，第一层及总参数量不同，不是严格等参数消融。\n'
            '- Train V使用target Q，validation V使用online Q，两条V曲线不可当作普通泛化gap。\n'
            '- Actor/Q/V loss、离线skip率和相邻checkpoint稳定性都不能替代真实闭环latency、TFLOPs、PSNR/SSIM/LPIPS或VBench。')),
        dict(id='next_steps', type='markdown', body=(
            '## 建议的后续验收\n\n'
            '1. 使用每组`selected_model.pt`和相同物理GPU/同prompt baseline，在显式Exact-K预算下运行匹配VBench200。\n'
            '2. 按完整generate latency、DiT/T5/VAE分项、DiT与predictor TFLOPs、特征提取wall time及PSNR/SSIM/LPIPS/VBench联合比较。\n'
            '3. 若要声称feature带来稳定收益，应为候选与SEA7增加多seed重复，并预先定义跨seed聚合与不确定性口径。')),
        dict(id='questions', type='markdown', body=(
            '## 仍需回答的问题\n\n'
            '- 训练指标较优的组在相同Exact-K下是否保持或提升视频质量？\n'
            '- 高维FFT/quantile特征的在线开销是否抵消缓存带来的完整推理收益？\n'
            '- 单seed下观察到的loss与稳定性差异在重复训练中是否保持？')),
    ]
    artifact = dict(surface='report', manifest=dict(version=1, surface='report', title=title,
        description='十二组共享selection的400-epoch离线IQL训练完整性、损失轨迹与post-300稳定性技术报告。',
        generatedAt=now, cards=cards, charts=charts, tables=tables, sources=sources, blocks=blocks),
        snapshot=dict(version=1, generatedAt=now, status='ready', datasets=dict(
            headline=headline, suite_summary=summary, epoch_aggregate=aggregate,
            selection_headline=selection_headline, runtime_headline=runtime_headline), accessIssues=[]),
        sources=sources)
    dump(root / 'artifact.json', artifact)
    print(json.dumps(dict(status='ready', artifact=str(root / 'artifact.json'),
        groups=len(summary), epoch_aggregate_rows=len(aggregate)), ensure_ascii=False))


if __name__ == '__main__':
    main()
