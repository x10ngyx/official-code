"""Audit existing e391 K29 traces and export lossless action matrices (CPU only)."""
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

ROOT = Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_dynamics128_e391_vbench50_random42_4gpu_v1')
OUT = ROOT / 'analysis/trace_k29_e391_50prompts'
SOURCES = {}


def read(path):
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data.decode()


def js(path):
    return json.loads(read(path))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def longest(values):
    best = run = 0
    for value in values:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def main():
    config = js(ROOT / 'config.json')
    complete = js(ROOT / 'COMPLETE.json')
    results = list(csv.DictReader(read(ROOT / 'analysis/results.csv').splitlines()))
    assert SOURCES['analysis/results.csv'] == complete['results_sha256']
    summary = next(r for r in results if int(r['k']) == 29)
    assert config['selected_epoch'] == int(summary['checkpoint_epoch']) == 391
    prompts = [json.loads(s) for s in read(ROOT / 'prompts/selected.jsonl').splitlines()]
    ids = [p['sample_id'] for p in prompts]
    assert ids == config['selected_ids'] == sorted(set(ids)) and len(ids) == 50
    quality = [r for r in csv.DictReader(read(ROOT / 'analysis/per_video.csv').splitlines()) if int(r['k']) == 29]
    assert len(quality) == 50 and {r['sample_id'] for r in quality} == set(ids)
    quality = {r['sample_id']: r for r in quality}
    raw, rows, matrix = [], [], []
    for prompt in prompts:
        sid = prompt['sample_id']
        q = quality[sid]
        folder = ROOT / f"shards/gpu{q['gpu']}/K29"
        run = js(folder / 'run.json')
        assert run['policy_sha256'] == config['checkpoint_sha256'] and run['skip_budget'] == 29
        baseline_run = js(folder.parent / 'baseline/run.json')
        assert run['gpu_uuid'] == baseline_run['gpu_uuid']
        trace = js(folder / f'traces/{sid}.json')
        timing = js(folder / f'timings/{sid}.json')
        assert timing['status'] == 'success'
        assert trace['total_steps'] == 50 and len(trace['decisions']) == 100
        assert len(timing['calls']) == 100
        assert abs(timing['pipeline_generate_wall_seconds'] - float(q['candidate_seconds'])) < 1e-8
        original_quality = list(csv.DictReader(read(folder / 'quality/per_video.csv').splitlines()))
        oq = next(r for r in original_quality if r['video_id'] == sid)
        assert abs(float(oq['psnr_rgb_db_mean']) - float(q['psnr_rgb_db'])) < 1e-10
        branches = []
        for offset, (branch, timing_branch) in enumerate((('cond', 'condition'), ('uncond', 'uncondition'))):
            decisions = [d for d in trace['decisions'] if d['branch'] == branch]
            assert [d['step_index'] for d in decisions] == list(range(50))
            assert all(d['action'] in ('reuse', 'recompute') for d in decisions)
            values = [int(d['action'] == 'reuse') for d in decisions]
            assert sum(values) == 29
            assert trace['per_branch'][branch]['reuse_path'] == [i for i, v in enumerate(values) if v]
            assert trace['per_branch'][branch]['recompute_path'] == [i for i, v in enumerate(values) if not v]
            for step, (value, decision) in enumerate(zip(values, decisions)):
                call = timing['calls'][2 * step + offset]
                assert call['cfg_branch'] == timing_branch
                assert call['blocks_executed'] == (0 if value else 30)
                raw.append(dict(sample_id=sid, prompt_en=prompt['prompt_en'], gpu=q['gpu'],
                                branch=branch, step_index=step, action=decision['action'],
                                reason=decision.get('reason', ''), blocks_executed=call['blocks_executed'],
                                psnr_rgb_db=q['psnr_rgb_db'], source_trace=str(folder / f'traces/{sid}.json')))
            branches.append(values)
        assert branches[0] == branches[1], f'{sid}: CFG actions differ; cannot collapse'
        values = branches[0]
        matrix.append(values)
        rows.append(dict(sample_id=sid, prompt_en=prompt['prompt_en'], psnr_rgb_db=q['psnr_rgb_db'],
                         reuse_steps=29, recompute_steps=21, first_reuse=values.index(1),
                         longest_reuse_run=longest(values), trace_0_to_49=''.join('R' if v else 'C' for v in values)))
    assert len(raw) == 5000
    psnr_mean = statistics.mean(float(r['psnr_rgb_db']) for r in rows)
    assert abs(psnr_mean - float(summary['psnr_rgb_db'])) < 1e-10
    actual = statistics.mean(float(q['baseline_seconds']) for q in quality.values()) / statistics.mean(float(q['candidate_seconds']) for q in quality.values())
    assert abs(actual - float(summary['latency_speedup'])) < 1e-10
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv('branch_step_traces.csv', raw)
    write_csv('prompt_traces.csv', rows)
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'svg.fonttype': 'path', 'text.color': '#20252B'})
    matrix = np.array(matrix)
    for start, stop, name in [(0, 50, 'trace_all50'), (0, 25, 'trace_01_25'), (25, 50, 'trace_26_50')]:
        n = stop - start
        fig, ax = plt.subplots(figsize=(17, 2.5 + n * .34))
        fig.subplots_adjust(left=.13, right=.91, top=1 - 1.7 / fig.get_figheight(), bottom=.7 / fig.get_figheight())
        ax.pcolormesh(np.arange(51)-.5, np.arange(n+1)-.5, matrix[start:stop],
                      cmap=ListedColormap(['#FFFFFF', '#2563A6']), vmin=0, vmax=1,
                      edgecolors='#D3D7DD', linewidth=.45)
        ax.set(xlim=(-.5, 49.5), ylim=(n-.5, -.5))
        ax.set_xticks(range(50))
        ax.set_yticks(range(n), [r['sample_id'] for r in rows[start:stop]])
        ax.tick_params(axis='x', top=True, labeltop=True, bottom=True, labelbottom=True, length=0, pad=5, labelsize=8)
        ax.tick_params(axis='y', length=0, pad=9, labelsize=10)
        ax.set_xlabel('采样 step（0–49）', fontsize=11, labelpad=10)
        for i, row in enumerate(rows[start:stop]):
            ax.text(50.2, i, f"{float(row['psnr_rgb_db']):.2f}", va='center', fontsize=10, clip_on=False)
        ax.text(50.2, -1.35, 'PSNR\n(dB)', va='bottom', fontsize=10, clip_on=False)
        fig.text(.13, 1-.32/fig.get_figheight(), 'Dynamics128 e391：50 条 prompt 的逐步 trace', fontsize=18, va='top')
        fig.text(.13, 1-.76/fig.get_figheight(), f'名义 2.4× · 固定 K29 · 实测 {actual:.4f}× · seed 42 · 当前显示第 {start+1}–{stop} 条', fontsize=11, va='top')
        fig.legend(handles=[Patch(facecolor='#2563A6', edgecolor='#D3D7DD', label='复用（每条 29 步）'),
                            Patch(facecolor='white', edgecolor='#A0A6AF', label='重算（每条 21 步）')],
                   loc='upper left', bbox_to_anchor=(.125, 1-1.08/fig.get_figheight()), frameon=False, ncol=2, fontsize=11)
        fig.text(.13, .12/fig.get_figheight(), 'cond / uncond 的动作逐步一致，合并显示；已核对实际执行 blocks。右侧为相对同卡 baseline 的视频 PSNR。', fontsize=9)
        for ext in ('png', 'svg'):
            fig.savefig(OUT / f'{name}.{ext}', dpi=170, facecolor='white')
        plt.close(fig)
    qa = dict(status='pass', prompts=50, steps_per_prompt=50, raw_branch_steps=5000, plotted_cells=2500,
              checkpoint_epoch=391, skip_budget=29, nominal_speedup=2.4, actual_speedup=actual,
              mean_psnr_db=psnr_mean, unique_traces=len({tuple(v) for v in matrix}),
              first_reuse_range=[min(r['first_reuse'] for r in rows), max(r['first_reuse'] for r in rows)],
              longest_reuse_run_range=[min(r['longest_reuse_run'] for r in rows), max(r['longest_reuse_run'] for r in rows)],
              cfg_actions_identical=True, actual_blocks_match=True, source_sha256=SOURCES)
    (OUT / 'VALIDATION.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in qa.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2))
    print(OUT)


if __name__ == '__main__':
    main()
