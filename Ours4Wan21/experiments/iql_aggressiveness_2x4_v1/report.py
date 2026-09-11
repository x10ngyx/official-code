"""Audit 240 paired videos, training/Q evidence and render the final suite report."""
import statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from ours4wan21.online_common import verified
from generate_worker import job_identity

COLORS={'Original':'#414141','a1':'#95b9d4','a2':'#5697bd','a3':'#246b98','a4':'#123c5c'}
STYLES={'Original':':','a1':'--','a2':'-.','a3':'-','a4':(0,(5,1,1,1))}

def savefig(fig,name):
    fig.savefig(ROOT/'figures'/(name+'.png'),dpi=170)
    fig.savefig(ROOT/'figures'/(name+'.svg'));plt.close(fig)

def plots(config,qrows,results):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    for metric in ('mean_Q','Q_IQR'):
        fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
        for feature,ax in zip(FEATURES,axes):
            for level in ('Original',*LEVELS):
                rr=[r for r in qrows if r['feature']==feature and r['level']==level]
                assert len(rr)==400
                ax.plot([int(r['epoch']) for r in rr],[float(r[metric]) for r in rr],color=COLORS[level],
                    ls=STYLES[level],label=level.upper() if level!='Original' else level,lw=1.5)
            ax.set(title=feature,xlabel='Epoch',ylabel=metric,xlim=(1,400));ax.grid(alpha=.16);ax.legend(fontsize=9,ncol=3)
        fig.suptitle(f'{metric} | same 13,738 validation states x 2 actions per feature')
        savefig(fig,metric)
    for feature in FEATURES:
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        groups=[g for g in config['groups'] if g['feature']==feature]
        series=[('Original',Path(groups[0]['baseline']))]+[(g['level'],Path(g['training'])) for g in groups]
        for level,path in series:
            rows=[json.loads(s) for s in (path/'epoch_metrics.jsonl').read_text().splitlines()]
            for i,split in enumerate(('train','val')):
                for j,key in enumerate(('q_loss','v_loss','pi_loss')):
                    ax=axes[i,j];ax.plot(range(1,401),[r[split][key] for r in rows],
                        color=COLORS[level],ls=STYLES[level],label=level,lw=1.1)
                    ax.set(title=split+' '+key,xlabel='Epoch');ax.grid(alpha=.15)
        axes[0,0].legend(fontsize=9)
        fig.suptitle(feature+' | 400 epochs; changed loss objectives do not rank video quality')
        savefig(fig,feature+'_training_comparison')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for i,feature in enumerate(FEATURES):
        for j,(key,label) in enumerate(zip(METRICS,['PSNR (dB, higher better)','SSIM (higher better)','LPIPS (lower better)'])):
            ax=axes[i,j]
            for level,marker in zip(LEVELS,['o','s','^','D']):
                rr=sorted([r for r in results if r['feature']==feature and r['level']==level],key=lambda r:r['k'])
                ax.plot([r['k'] for r in rr],[r[key] for r in rr],color=COLORS[level],ls=STYLES[level],
                    marker=marker,label=level.upper(),markerfacecolor='white' if level in ('a1','a3') else COLORS[level])
            ax.set(title=feature+' | '+label,xlabel='Fixed skip budget K',xticks=BUDGETS);ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=9,ncol=2)
    fig.suptitle('Same 10 VBench prompts | three fixed budgets | seed 42 | no VBench score')
    savefig(fig,'quality_comparison')

def main():
    config=read(ROOT/'config.json');verify_sources(config)
    reference=load_evaluation_bundle(REFERENCE)
    assert config['prompts']==freeze_prompts(reference,config['gpu_uuids'])
    evidence={};detail=[];qrows=[];selections=[]
    for gpu in range(4):
        jobs=read(ROOT/'jobs'/f'gpu{gpu}.json')
        qp=ROOT/'evaluation/quality'/f'gpu{gpu}'/'per_video.csv'
        qs=read(qp.with_name('summary.json'))
        assert qs['video_count']==len(jobs) and qs['frame_count_total']==81*len(jobs)
        quality={r['video_id']:r for r in csv.DictReader(qp.open())}
        assert set(quality)=={f"{j['group']}_K{j['skip_budget']}_{j['sample_id']}" for j in jobs}
        evidence[str(qp)]=sha256(qp)
        for job in jobs:
            out=Path(job['output']);assert verified(out,job_identity(job,config))
            generation=read(out/'generation.json');c=read(out/'measurement.json')
            assert generation['protocol']==PROTOCOL and generation['gpu_uuid']==job['expected_gpu_uuid']==config['gpu_uuids'][gpu]
            q=quality[f"{job['group']}_K{job['skip_budget']}_{job['sample_id']}"]
            assert (int(q['frames']),int(q['width']),int(q['height']))==(81,832,480)
            for side in ('reference','candidate'):assert sha256(q[side])==q[side+'_sha256']
            assert Path(q['candidate']).resolve()==(out/'video.mp4').resolve()
            assert Path(q['reference']).resolve()==(REFERENCE/'baselines'/job['sample_id']/'video.mp4').resolve()
            baseline=read(REFERENCE/'baselines'/job['sample_id']/'measurement.json')
            trace=audit_trace(read(out/'trace.json'),read(out/'timing.json'),job['skip_budget'])
            assert all(math.isfinite(c[key]) and c[key]>=0 for key in FIELDS)
            assert all(math.isfinite(float(q[key+'_mean'])) for key in METRICS)
            g=next(g for g in config['groups'] if g['name']==job['group'])
            selected=read(Path(g['analysis'])/'checkpoint_selection.json')
            assert job['state_mode']==g['mode']
            assert job['checkpoint']==dict(path=selected['checkpoint'],sha256=selected['checkpoint_sha256'])
            detail.append(dict(group=job['group'],feature=g['feature'],level=g['level'],k=job['skip_budget'],
                target=TARGETS[BUDGETS.index(job['skip_budget'])],sample_id=job['sample_id'],gpu=gpu,
                baseline_seconds=baseline['generate_seconds'],**{key:c[key] for key in FIELDS},
                **{key:float(q[key+'_mean']) for key in METRICS},**trace))
            evidence[str(out/'COMPLETE.json')]=sha256(out/'COMPLETE.json')
    assert len(detail)==240
    results=[]
    for g in config['groups']:
        verify_training(g)
        selection_path=Path(g['analysis'])/'checkpoint_selection.json';s=read(selection_path)
        assert 300<s['checkpoint_epoch']<400 and sha256(s['checkpoint'])==s['checkpoint_sha256']
        assert s['selected']['epoch']==s['checkpoint_epoch']
        selections.append(dict(group=g['name'],feature=g['feature'],level=g['level'],gate=s['gate'],**s['selected']))
        evidence[str(selection_path)]=sha256(selection_path)
        extra=Path(g['analysis'])/'extra';done=read(extra/'COMPLETE.json')
        assert done['status']=='complete' and done['epochs_per_method']==400 and done['validation_states']==13738
        assert sha256(extra/'q_statistics.csv')==done['statistics_sha256']
        for r in csv.DictReader((extra/'q_statistics.csv').open()):
            if r['method']=='Original' and g['level']!='a1':continue
            qrows.append(dict(feature=g['feature'],level='Original' if r['method']=='Original' else g['level'],**r))
        for k in BUDGETS:
            rr=[r for r in detail if r['group']==g['name'] and r['k']==k]
            assert len(rr)==10 and {r['sample_id'] for r in rr}=={r['sample_id'] for r in config['prompts']}
            result=dict(group=g['name'],feature=g['feature'],level=g['level'],epoch=s['checkpoint_epoch'],
                tau=g['training_config']['tau'],beta=g['training_config']['beta'],weight_max=g['training_config']['weight_max'],
                k=k,target=TARGETS[BUDGETS.index(k)],n=10,
                latency_speedup=sum(r['baseline_seconds'] for r in rr)/sum(r['generate_seconds'] for r in rr),
                **{key:st.fmean(r[key] for r in rr) for key in (*FIELDS,*METRICS,'late25','longest_late25')})
            results.append(result)
    assert len(qrows)==4000 and len(results)==24
    writecsv(ROOT/'per_video.csv',detail);writecsv(ROOT/'results.csv',results)
    writecsv(ROOT/'q_statistics.csv',qrows);writecsv(ROOT/'checkpoint_selection.csv',selections)
    plots(config,qrows,results)
    lines=['# SEA7 / Dynamics128：四档激进IQL结果','',
        '8×400轮；每组仅按e301–399验证actor/Q双侧稳定性选点。全部组共用同10条提示词、seed42和同GPU原native baseline。',
        '名义1.8/2.4/3.0×固定K23/29/35；PSNR/SSIM/LPIPS相对原native视频，VBench分数按用户此前要求跳过。','',
        '| 方案 | 档位 | epoch | 名义档 | 实测加速 | 推理秒 | PSNR | SSIM | LPIPS | DiT TFLOPs | 后25步skip | 后段最长连续skip |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['feature']} | {r['level']} | {r['epoch']} | {r['target']:.1f}× | {r['latency_speedup']:.4f}× | {r['generate_seconds']:.3f} | {r['psnr_rgb_db']:.4f} | {r['ssim_rgb']:.5f} | {r['lpips_alex_v0_1_spatial']:.5f} | {r['dit_tflops']:.3f} | {r['late25']:.2f} | {r['longest_late25']:.2f} |")
    lines+=['','## Q统计与训练诊断','',
        'figures/mean_Q.png和Q_IQR.png覆盖完整400轮，每方案含四档及原组。Q=min(Q1,Q2)，同13738个验证自由状态×两动作；IQR使用midpoint经验分位数。',
        'τ与优势权重改变了训练目标，mean_Q升高、Q-IQR降低或原始loss更低均不能单独证明策略更好。对比策略需联合固定K下的质量与后段skip。','',
        '| 方案 | 档位 | 选定epoch | actor稳定门槛 | 选定mean_Q | 选定Q-IQR | e351–400 mean_Q | e351–400 Q-IQR |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in selections:
        rr=[r for r in qrows if r['feature']==s['feature'] and r['level']==s['level']]
        chosen=next(r for r in rr if int(r['epoch'])==s['epoch']);late=[r for r in rr if int(r['epoch'])>=351]
        lines.append(f"| {s['feature']} | {s['level']} | {s['epoch']} | {s['gate']:.5f} | {float(chosen['mean_Q']):.6f} | {float(chosen['Q_IQR']):.6f} | {st.fmean(float(r['mean_Q']) for r in late):.6f} | {st.fmean(float(r['Q_IQR']) for r in late):.6f} |")
    lines+=['','results.csv保留T5/DiT/VAE及predictor/feature时间和组件TFLOPs，per_video.csv保留全部240条质量、50步skip path及后段集中程度。',
        '原baseline时间来自此前同GPU测试，不是同时重测；质量仅10prompt单seed描述性比较。四档是三参数联合改变，不能分离单参数因果贡献。',
        '当选择门槛低于.95时表示使用最大可得actor一致率回退，只能称相对稳定选点，不能称达到预设稳定阈值。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    verify_sources(config)
    for item in config['inputs'].values():assert sha256(item['path'])==item['sha256']
    dump(ROOT/'VALIDATION.json',dict(status='pass',groups=8,epochs=3200,candidates=240,quality_frames=19440,
        cfg_actions=24000,q_statistics_rows=4000,evidence_sha256=evidence,visual_qa='exported; human/model image review pending',
        vbench_status='skipped_by_user'))
    dump(ROOT/'COMPLETE.json',dict(status='complete',groups=8,epochs=3200,candidates=240,prompts=10,
        results_sha256=sha256(ROOT/'results.csv'),report_sha256=sha256(ROOT/'REPORT.md'),
        validation_sha256=sha256(ROOT/'VALIDATION.json'),vbench_status='skipped_by_user'))

if __name__=='__main__':main()
