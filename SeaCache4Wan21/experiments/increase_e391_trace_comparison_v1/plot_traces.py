"""Compare existing measured increase/e391 traces; CPU-only, no inference."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Patch
import numpy as np

BASE = Path('/mnt/hdd/xiongyuxiang/tmp/exp')
INC = BASE / 'seacache_wan21_linear_increase_matched_5x4_v1'
DYN = BASE / 'ours21_dynamics128_e391_vbench50_random42_4gpu_v1'
D38 = BASE / 'wan21_vbench50_speed36_seacache_dynamics128_e391_v1'
OUT = BASE / 'wan21_increase_e391_trace_comparison_v1'
SOURCES = {}
BLUE, ORANGE, INK, GRAY, GRID = '#376CB1', '#C87524', '#202B38', '#647181', '#CDD5DF'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    data = p.read_bytes()
    SOURCES[str(p)] = hashlib.sha256(data).hexdigest()
    return data.decode()


def js(p):
    return json.loads(read(p))


def csvrows(p):
    return list(csv.DictReader(read(p).splitlines()))


def writecsv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    cfg = {p: js(p / 'config.json') for p in (INC, DYN, D38)}
    evidence = {}
    quality = {}
    for p in cfg:
        complete = js(p / 'COMPLETE.json')
        assert complete['status'] == 'complete'
        validation = js(p / 'analysis/VALIDATION.json')
        assert sha(p / 'analysis/VALIDATION.json') == complete['validation_sha256']
        read(p / 'analysis/results.csv')
        assert sha(p / 'analysis/results.csv') == complete['results_sha256']
        evidence[p] = validation.get('evidence_sha256', {})
        quality[p] = csvrows(p / 'analysis/per_video.csv')
        assert cfg[p]['protocol'] == cfg[INC]['protocol']
    assert cfg[DYN]['selected_epoch'] == cfg[D38]['selected_epoch'] == 391
    assert cfg[DYN]['checkpoint_sha256'] == cfg[D38]['checkpoint_sha256']
    prompts = sorted(cfg[INC]['prompts'].values(), key=lambda p: p['sample_id'])
    originals = [json.loads(s) for s in read(DYN / 'prompts/selected.jsonl').splitlines()]
    originals = {p['sample_id']: p for p in originals}
    for p in prompts:
        assert p['prompt_en'] == originals[p['sample_id']]['prompt_en']
    rows, raw = [], []
    for root, method, levels in ((INC, 'increase', range(1, 6)), (DYN, 'dynamic e391', (23, 29, 35)), (D38, 'dynamic e391', (38,))):
        for level in levels:
            for prompt in prompts:
                sid = prompt['sample_id']
                candidates = [q for q in quality[root] if q['sample_id'] == sid]
                if root == INC:
                    q = next(q for q in candidates if q['phase'] == 'increase' and int(q['strategy']) == level)
                    folder = root / f"conditions/increase_{level}/gpu{q['gpu']}"
                elif root == DYN:
                    q = next(q for q in candidates if int(q['k']) == level)
                    folder = root / f"shards/gpu{q['gpu']}/K{level}"
                else:
                    q = next(q for q in candidates if q['method'] == 'dynamics128')
                    folder = root / f"shards/gpu{q['gpu']}/dynamics128"
                run = js(folder / 'run.json')
                tracepath = folder / f'traces/{sid}.json'
                timingpath = folder / f'timings/{sid}.json'
                tr, tm = js(tracepath), js(timingpath)
                for file in (folder / 'run.json', tracepath, timingpath):
                    rel = str(file.relative_to(root))
                    expected = evidence[root].get(rel, evidence[root].get(str(file)))
                    if expected is not None:
                        assert sha(file) == expected
                gpu = q['gpu']
                assert run['gpu_uuid'] == cfg[INC]['gpu_uuids'][gpu]
                assert run['protocol'] == cfg[INC]['protocol']
                baseline = next(r for r in js(DYN / f'shards/gpu{gpu}/baseline/components.json')['rows'] if r['sample_id'] == sid)
                assert float(q['baseline_seconds']) == baseline['generate_seconds'] == cfg[INC]['baselines'][gpu]['generate_seconds']
                if root == INC:
                    strategy = cfg[INC]['strategies'][level - 1]
                    assert run['threshold_path'] == tr['threshold_path'] == strategy['path']
                    label = f"Increase {level}  ({strategy['start']:.2f}→{strategy['end']:.2f})"
                else:
                    assert run['policy_sha256'] == cfg[root]['checkpoint_sha256']
                    assert run['skip_budget'] == tr['skip_budget'] == level
                    label = f'Dynamic e391  K{level}'
                assert tr['total_steps'] == 50 and len(tr['decisions']) == len(tm['calls']) == 100
                assert tm['status'] == 'success'
                seconds = float(q['generate_seconds'] if root == INC else q['candidate_seconds'])
                assert abs(seconds - tm['pipeline_generate_wall_seconds']) < 1e-8
                branches = []
                for offset, (branch, timing_branch) in enumerate((('cond', 'condition'), ('uncond', 'uncondition'))):
                    ds = [d for d in tr['decisions'] if d['branch'] == branch]
                    assert [d['step_index'] for d in ds] == list(range(50))
                    assert all(d['action'] in ('reuse', 'recompute') for d in ds)
                    values = [int(d['action'] == 'reuse') for d in ds]
                    for step, (d, skip) in enumerate(zip(ds, values)):
                        call = tm['calls'][step * 2 + offset]
                        assert call['cfg_branch'] == timing_branch
                        assert call['blocks_executed'] == (0 if skip else 30)
                        assert d['action'] == d['execution']
                        if root == INC:
                            assert d['requested_threshold'] == strategy['path'][step]
                        raw.append(dict(method=method, level=level, sample_id=sid, branch=branch,
                                        step=step+1, action=d['action'], blocks_executed=call['blocks_executed'],
                                        source_trace=str(tracepath)))
                    branches.append(values)
                assert branches[0] == branches[1]
                skip = branches[0]
                assert sum(skip) * 2 == tr['reuse'] and (50-sum(skip)) * 2 == tr['recompute']
                if root == INC:
                    assert sum(skip) == float(q['reuse_steps'])
                else:
                    assert sum(skip) == level
                rows.append(dict(method=method, level=level, label=label, sample_id=sid,
                                 prompt_en=prompt['prompt_en'], gpu=int(gpu), skip_steps=sum(skip),
                                 recompute_steps=50-sum(skip), skip_first25=sum(skip[:25]),
                                 skip_last25=sum(skip[25:]), generate_seconds=seconds,
                                 latency_speedup=float(q['baseline_seconds'])/seconds,
                                 psnr_rgb_db=float(q['psnr_rgb_db']), source_trace=str(tracepath),
                                 skip_path=''.join(map(str,skip))))
    rows.sort(key=lambda r: (r['skip_steps'], r['method'], r['level'], r['sample_id']))
    assert len(rows) == 36 and len(raw) == 3600
    assert set(Counter((r['method'], r['level']) for r in rows).values()) == {4}
    assert len({r['source_trace'] for r in rows}) == 36
    OUT.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        r['display_rank'] = i + 1
    writecsv('trace_summary_sorted.csv', rows)
    writecsv('branch_steps.csv', raw)
    writecsv('step_actions.csv', [dict(display_rank=r['display_rank'], method=r['method'],level=r['level'],
        sample_id=r['sample_id'],prompt_en=r['prompt_en'],skip_steps=r['skip_steps'],step=i+1,
        action='skip' if v=='1' else 'recompute') for r in rows for i,v in enumerate(r['skip_path'])])
    writecsv('prompts.csv', [dict(sample_id=p['sample_id'],prompt_en=p['prompt_en']) for p in prompts])
    draw(rows)
    SOURCES[str(Path(__file__).resolve())] = sha(Path(__file__))
    assert all(sha(p) == h for p,h in SOURCES.items())
    validation = dict(status='pass', trajectories=36, levels=9, traces_per_level=4,
        display_cells=1800, branch_records=3600, cfg_actions_identical=True, actual_blocks_match=True,
        same_four_prompts=True, selection='frozen increase prompts, no outcome selection',
        sort='actual skip_steps ascending, method, level, sample_id',
        source_sha256=SOURCES, figures_sha256={p.name:sha(p) for p in OUT.glob('trace_comparison.*')})
    (OUT/'VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    (OUT/'README.md').write_text('# Increase / dynamic e391 实测trace对比\n\n'
        '同四条prompt，每档4条：increase五档和e391 K23/K29/K35/K38，共36条。\n'
        '按实测skip步数升序；同数时按方法、档位、prompt ID排序。每条完整显示50步。\n'
        'skip按采样step计数；两个CFG分支动作核对一致后合并，原始3600次调用均核对实际blocks。\n\n'
        '- `trace_comparison.png` / `.svg`：完整排序图，实心格重算、白色格skip。\n'
        '- `trace_summary_sorted.csv`：36条汇总，含速度、PSNR、路径与来源。\n'
        '- `step_actions.csv` / `branch_steps.csv`：1800个显示格 / 3600次CFG调用。\n'
        '- `prompts.csv`：完整prompt。`VALIDATION.json`：校验与来源哈希。\n\n'
        '使用既有实测数据，没有新增推理或质量评测；各档skip/速度不同，不视为等速质量比较。\n')
    print(json.dumps(dict(status='pass',trajectories=len(rows),skip_counts=sorted({r['skip_steps'] for r in rows}),output=str(OUT))))


def draw(rows):
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    fontManager.addfont(font)
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'svg.fonttype':'path',
        'font.size':11,'text.color':INK,'axes.labelcolor':INK,'xtick.color':GRAY,'ytick.color':INK})
    fig=plt.figure(figsize=(19,14))
    ax=fig.add_axes([.255,.092,.615,.765])
    metrics=fig.add_axes([.887,.092,.096,.765])
    fig.text(.025,.972,'手工 Increase 与 Dynamic e391：实测 trace 对比',fontsize=23,weight='bold',va='top')
    fig.text(.025,.934,'按实际 skip 数升序排列 · 每档同 4 条 prompt · Increase 5 档 + e391 4 档 = 36 条 trace',fontsize=12,color=GRAY)
    fig.legend(handles=[Patch(facecolor=BLUE,edgecolor=BLUE,label='Increase：重算'),
        Patch(facecolor=ORANGE,edgecolor=ORANGE,label='Dynamic e391：重算'),
        Patch(facecolor='white',edgecolor=GRID,label='白色：skip / 复用')],
        loc='upper left',bbox_to_anchor=(.025,.913),frameon=False,ncol=3,fontsize=12)
    matrix=np.array([[0 if v=='1' else (1 if r['method']=='increase' else 2) for v in r['skip_path']] for r in rows])
    ax.pcolormesh(np.arange(51),np.arange(37),matrix,cmap=ListedColormap(['white',BLUE,ORANGE]),
        vmin=0,vmax=2,edgecolors=GRID,linewidth=.45,antialiased=True)
    ax.set(xlim=(0,50),ylim=(36,0))
    ticks=[1,5,10,15,20,25,30,35,40,45,50]
    ax.set_xticks([t-.5 for t in ticks],ticks)
    ax.tick_params(axis='both',length=0,pad=7)
    ax.set_yticks(np.arange(36)+.5,[r['label']+'  |  '+r['sample_id'].split('_')[-1] for r in rows])
    ax.set_xlabel('采样步（按实际执行顺序，1 → 50）',labelpad=12)
    metrics.set(xlim=(0,1),ylim=(36,0));metrics.axis('off')
    for x,label in ((.22,'skip'),(.76,'重算')):
        metrics.text(x,-.72,label,ha='center',fontsize=12,weight='bold')
    for i,r in enumerate(rows):
        metrics.text(.22,i+.5,str(r['skip_steps']),ha='center',va='center',weight='bold',fontsize=12)
        metrics.text(.76,i+.5,str(r['recompute_steps']),ha='center',va='center',fontsize=12)
        if i and r['skip_steps']!=rows[i-1]['skip_steps']:
            ax.axhline(i,color=INK,lw=1.1);metrics.axhline(i,color=GRID,lw=.7)
    ax.spines[:].set_visible(False)
    fig.text(.025,.039,'skip 按 50 个采样 step 计数；双 CFG 动作一致后合并。全部 3,600 次调用已核对实际 blocks。',fontsize=11,color=GRAY)
    fig.text(.025,.019,'Prompt ID：040 / 041 / 144 / 155（各档相同，未按质量或路径挑选）；seed 42。完整 prompt 与逐步 CSV 随图保存。',fontsize=10.5,color=GRAY)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    clipped=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:clipped.append(t.get_text())
    assert not clipped,clipped
    for ext in ('png','svg'):fig.savefig(OUT/f'trace_comparison.{ext}',dpi=180,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    main()
